import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniGPTConfig:
    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=256,
        n_layer=6,
        n_head=4,
        block_size=16,
        bias=True
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.block_size = block_size
        self.bias = bias


def write_to_paged_cache(k_cache, v_cache, key, value, block_table, context_lens, block_size):
    """
    Writes newly generated keys and values for a batch of sequences to the physical KV cache.
    
    Args:
        k_cache (Tensor): Key cache tensor, shape [num_blocks, num_heads, block_size, head_dim]
        v_cache (Tensor): Value cache tensor, shape [num_blocks, num_heads, block_size, head_dim]
        key (Tensor): Key tensor for new tokens, shape [batch_size, num_heads, num_new_tokens, head_dim]
        value (Tensor): Value tensor for new tokens, shape [batch_size, num_heads, num_new_tokens, head_dim]
        block_table (Tensor): Block mapping table, shape [batch_size, max_blocks_per_seq]
        context_lens (Tensor): 1D tensor containing the sequence lengths *before* writing these new tokens.
        block_size (int): Size of each physical block (number of tokens).
    """
    batch_size, num_heads, num_new_tokens, head_dim = key.shape
    
    for i in range(batch_size):
        start_idx = context_lens[i].item()
        phys_blocks = block_table[i]
        
        for t in range(num_new_tokens):
            token_idx = start_idx + t
            logical_blk = token_idx // block_size
            slot = token_idx % block_size
            phys_blk = phys_blocks[logical_blk].item()
            
            # Write key and value into the corresponding slot of the physical block
            k_cache[phys_blk, :, slot, :] = key[i, :, t, :]
            v_cache[phys_blk, :, slot, :] = value[i, :, t, :]


def read_from_paged_cache(k_cache, v_cache, block_table, context_lens, total_len, block_size):
    """
    Reads the full keys and values for a batch of sequences from the physical KV cache up to total_len.
    
    Args:
        k_cache (Tensor): Key cache tensor, shape [num_blocks, num_heads, block_size, head_dim]
        v_cache (Tensor): Value cache tensor, shape [num_blocks, num_heads, block_size, head_dim]
        block_table (Tensor): Block mapping table, shape [batch_size, max_blocks_per_seq]
        context_lens (Tensor): 1D tensor of active sequence lengths (after the current step).
        total_len (int): Maximum length of context to retrieve.
        block_size (int): Size of each physical block.
        
    Returns:
        K_seq (Tensor): Gathered key tensor, shape [batch_size, num_heads, total_len, head_dim]
        V_seq (Tensor): Gathered value tensor, shape [batch_size, num_heads, total_len, head_dim]
    """
    batch_size = block_table.size(0)
    num_heads = k_cache.size(1)
    head_dim = k_cache.size(3)
    device = k_cache.device
    
    K_seq = torch.zeros(batch_size, num_heads, total_len, head_dim, device=device)
    V_seq = torch.zeros(batch_size, num_heads, total_len, head_dim, device=device)
    
    for i in range(batch_size):
        length = context_lens[i].item()
        phys_blocks = block_table[i]
        
        for t in range(length):
            logical_blk = t // block_size
            slot = t % block_size
            phys_blk = phys_blocks[logical_blk].item()
            
            K_seq[i, :, t, :] = k_cache[phys_blk, :, t % block_size, :]
            V_seq[i, :, t, :] = v_cache[phys_blk, :, t % block_size, :]
            
    return K_seq, V_seq


class PagedTreeSelfAttention(nn.Module):
    """
    Self-attention layer reading and writing from a global paged KV cache tensor.
    Supports Tree-Attention masking for speculative decoding verification.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.block_size = config.block_size
        
        # QKV projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
    def forward(
        self,
        x,
        block_table,
        context_lens,
        k_cache=None,
        v_cache=None,
        attention_mask=None,
        is_prefill=False
    ):
        B, T, C = x.size()
        
        # Compute Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape to [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        if k_cache is not None and v_cache is not None:
            # Paged KV Cache execution pathway
            if is_prefill:
                # Prefill: write brand new sequence keys and values to cache
                write_to_paged_cache(k_cache, v_cache, k, v, block_table, context_lens, self.block_size)
                
                # Retrieve sequence keys and values up to prompt size
                total_lens = context_lens + T
                K_seq, V_seq = read_from_paged_cache(
                    k_cache, v_cache, block_table, total_lens, total_lens.max().item(), self.block_size
                )
            else:
                # Decode (T=1) or Speculative Verification (T > 1)
                # Write current generated/speculated keys and values to cache starting at slot context_lens
                write_to_paged_cache(k_cache, v_cache, k, v, block_table, context_lens, self.block_size)
                
                # Fetch full keys/values from start up to (context_lens + T)
                total_lens = context_lens + T
                K_seq, V_seq = read_from_paged_cache(
                    k_cache, v_cache, block_table, total_lens, total_lens.max().item(), self.block_size
                )
        else:
            # Non-cached execution pathway (standard self-attention)
            K_seq, V_seq = k, v
            total_lens = torch.full((B,), T, dtype=torch.long, device=x.device)
            
        # Attention computation: Q @ K^T / sqrt(d)
        # shape: [B, n_head, T, total_len]
        att = (q @ K_seq.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Apply attention mask
        if attention_mask is not None:
            # attention_mask shape should broadcast to [B, 1, T, total_len]
            att = att.masked_fill(attention_mask == 0, float('-inf'))
        else:
            # Fallback dynamic causal mask
            if context_lens is not None:
                mask = torch.zeros(B, 1, T, K_seq.size(-2), dtype=torch.bool, device=x.device)
                for i in range(B):
                    start = context_lens[i].item()
                    if is_prefill:
                        # Causal masking for prompt prefill tokens
                        for t_idx in range(T):
                            mask[i, 0, t_idx, :start + t_idx + 1] = True
                    else:
                        # Decode masking: can attend to all past tokens plus current step
                        t_len = total_lens[i].item()
                        mask[i, 0, :, :t_len] = True
                att = att.masked_fill(~mask, float('-inf'))
            else:
                # Standard causal mask for non-cached pathway
                mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
                att = att.masked_fill(~mask[None, None, :, :], float('-inf'))
            
        att = F.softmax(att, dim=-1)
        y = att @ V_seq # [B, n_head, T, head_dim]
        
        # Reassemble channels
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        
        # Projection output
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = PagedTreeSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(
        self,
        x,
        block_table,
        context_lens,
        k_cache=None,
        v_cache=None,
        attention_mask=None,
        is_prefill=False
    ):
        x = x + self.attn(
            self.ln_1(x),
            block_table,
            context_lens,
            k_cache,
            v_cache,
            attention_mask,
            is_prefill
        )
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniGPT(nn.Module):
    """
    Custom GPT model supporting Paged KV cache and custom attention routing.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.n_positions, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie embeddings
        self.transformer.wte.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(
        self,
        input_ids,
        position_ids,
        block_table,
        context_lens,
        k_caches=None,
        v_caches=None,
        attention_mask=None,
        is_prefill=False
    ):
        device = input_ids.device
        B, T = input_ids.size()
        
        # Embed tokens and positions
        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(position_ids)
        x = tok_emb + pos_emb
        
        # Pass through transformer layers
        for i, block in enumerate(self.transformer.h):
            # Extract layer-specific key and value caches if provided
            k_c = k_caches[i] if k_caches is not None else None
            v_c = v_caches[i] if v_caches is not None else None
            
            x = block(
                x,
                block_table,
                context_lens,
                k_cache=k_c,
                v_cache=v_c,
                attention_mask=attention_mask,
                is_prefill=is_prefill
            )
            
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
