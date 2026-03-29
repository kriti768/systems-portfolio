import time
import asyncio
import torch
from typing import List, Dict, Tuple, Optional
from model import MiniGPTConfig, MiniGPT
from radix_cache import RadixCacheManager, RadixNode
from scheduler import Request, ContinuousScheduler
from speculative_engine import SpeculativeVerifier

class LLMEngine:
    """
    Coordinates the models, scheduler, radix cache manager, and speculative verifier.
    Runs iteration-level scheduling steps to generate tokens for active requests.
    """
    def __init__(
        self,
        target_config: MiniGPTConfig,
        draft_config: MiniGPTConfig,
        num_blocks: int = 64,
        chunk_size: int = 16,
        enable_speculative: bool = True
    ):
        self.target_config = target_config
        self.draft_config = draft_config
        self.block_size = target_config.block_size
        self.enable_speculative = enable_speculative
        self.chunk_size = chunk_size
        
        # Instantiate models
        self.target_model = MiniGPT(target_config)
        self.draft_model = MiniGPT(draft_config)
        
        # Disable gradients
        self.target_model.eval()
        self.draft_model.eval()
        for p in self.target_model.parameters():
            p.requires_grad = False
        for p in self.draft_model.parameters():
            p.requires_grad = False
            
        # Cache manager & scheduler
        self.cache_manager = RadixCacheManager(num_blocks=num_blocks, block_size=self.block_size)
        self.scheduler = ContinuousScheduler(self.cache_manager, chunk_size=chunk_size)
        self.verifier = SpeculativeVerifier(self.target_model, self.draft_model)
        
        # Pre-allocate physical KV caches for Target model (one pair per layer)
        self.head_dim = target_config.n_embd // target_config.n_head
        self.k_caches = [
            torch.zeros(num_blocks, target_config.n_head, self.block_size, self.head_dim)
            for _ in range(target_config.n_layer)
        ]
        self.v_caches = [
            torch.zeros(num_blocks, target_config.n_head, self.block_size, self.head_dim)
            for _ in range(target_config.n_layer)
        ]
        
        # Keep track of request metadata for dashboard telemetry
        self.completed_requests: List[Request] = []
        self.step_count = 0
        self.speculative_accepted_tokens = 0
        self.speculative_total_tokens = 0

    def add_request(self, prompt_tokens: List[int], max_gen_len: int = 64) -> Request:
        req = Request(prompt_tokens, max_gen_len)
        self.scheduler.add_request(req)
        return req

    def step(self) -> List[Tuple[Request, List[int]]]:
        """
        Executes one batch iteration step (iteration-level scheduling).
        Runs context prefilling or speculative decoding validation.
        
        Returns:
            outputs (List[Tuple[Request, List[int]]]): New tokens generated for each request in this step.
        """
        self.step_count += 1
        prefill_requests, decode_requests = self.scheduler.schedule()
        
        # If nothing is scheduled, return
        if not prefill_requests and not decode_requests:
            return []
            
        outputs: List[Tuple[Request, List[int]]] = []
        device = next(self.target_model.parameters()).device
        
        # --- Handle Prefill Tasks ---
        for req in prefill_requests:
            start_pos = req.num_prefilled_tokens
            prompt_len = len(req.prompt_tokens)
            chunk_len = min(self.chunk_size, prompt_len - start_pos)
            chunk_tokens = req.prompt_tokens[start_pos : start_pos + chunk_len]
            
            # Format inputs
            input_ids = torch.tensor([chunk_tokens], dtype=torch.long, device=device)
            position_ids = torch.arange(start_pos, start_pos + chunk_len, device=device)[None, :]
            block_table = torch.tensor([req.allocated_block_ids + [-1] * (16 - len(req.allocated_block_ids))], dtype=torch.long, device=device)
            context_lens = torch.tensor([start_pos], dtype=torch.long, device=device)
            
            with torch.no_grad():
                logits = self.target_model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    block_table=block_table,
                    context_lens=context_lens,
                    k_caches=self.k_caches,
                    v_caches=self.v_caches,
                    is_prefill=True
                )
                
            # Update prefill progress
            req.num_prefilled_tokens += chunk_len
            
            # If prompt is fully prefilled, transition to decode and yield the first token
            if req.num_prefilled_tokens == prompt_len:
                req.state = "running"
                first_token = torch.argmax(logits[0, -1, :]).item()
                req.generated_tokens.append(first_token)
                req.first_token_time = time.time()
                
                # Push first token to SSE output queue
                req.output_queue.put_nowait(first_token)
                outputs.append((req, [first_token]))
                
                # Insert completed blocks into Radix Tree to share with future requests
                num_complete_blocks = prompt_len // self.block_size
                curr_node = self.cache_manager.root
                for b_idx in range(num_complete_blocks):
                    block_tokens = tuple(req.prompt_tokens[b_idx * self.block_size : (b_idx + 1) * self.block_size])
                    phys_blk_id = req.allocated_block_ids[b_idx]
                    
                    # Insert in tree (uses phys_blk_id, increments ref_count)
                    curr_node = self.cache_manager.insert_block(curr_node, block_tokens, phys_block_id=phys_blk_id)
                    if curr_node not in req.referenced_nodes:
                        req.referenced_nodes.append(curr_node)
                        curr_node.ref_count += 1
                        
                req.current_node = curr_node
                
        # --- Handle Decode Tasks ---
        for req in decode_requests:
            total_len = req.get_total_tokens()
            
            # Check completion criteria
            if req.is_finished():
                self.scheduler.complete_request(req)
                self.completed_requests.append(req)
                continue
                
            # Full token sequence representation
            full_seq = req.prompt_tokens + req.generated_tokens
            
            if self.enable_speculative:
                # 1. Speculative Decoding Path
                # Use Draft model to generate the candidate tree
                draft_tensor = torch.tensor([full_seq], dtype=torch.long, device=device)
                flat_tokens, paths, draft_ancestors = self.verifier.generate_draft_tree(draft_tensor)
                
                # Assemble 7-token query: [root_token] + flat_tokens
                root_token = full_seq[-1]
                query_tokens = [root_token] + flat_tokens
                tree_size = len(query_tokens)
                
                # Setup target model execution
                input_ids = torch.tensor([query_tokens], dtype=torch.long, device=device)
                # Position ids start at total_len - 1
                position_ids = torch.arange(total_len - 1, total_len - 1 + tree_size, device=device)[None, :]
                
                # Temporary allocate blocks to cover tree length (which now is total_len - 1 + tree_size)
                future_tokens = total_len - 1 + tree_size
                required_blocks = (future_tokens + self.block_size - 1) // self.block_size
                blocks_to_allocate = required_blocks - len(req.allocated_block_ids)
                
                # Dynamically request temp blocks from manager
                temp_blocks = []
                for _ in range(blocks_to_allocate):
                    temp_blk = self.cache_manager.allocate_block()
                    temp_blocks.append(temp_blk)
                
                eval_block_ids = req.allocated_block_ids + temp_blocks
                block_table = torch.tensor([eval_block_ids + [-1] * (32 - len(eval_block_ids))], dtype=torch.long, device=device)
                
                # Construct 7-token query ancestors map
                # Idx 0: root (attends to 0)
                # Idx 1: T1 (attends to 0, 1)
                # Idx 2: T2 (attends to 0, 2)
                # Idx 3: T11 (attends to 0, 1, 3)
                # Idx 4: T12 (attends to 0, 1, 4)
                # Idx 5: T21 (attends to 0, 2, 5)
                # Idx 6: T22 (attends to 0, 2, 6)
                ancestors = [
                    [0],
                    [0, 1],
                    [0, 2],
                    [0, 1, 3],
                    [0, 1, 4],
                    [0, 2, 5],
                    [0, 2, 6]
                ]
                
                # Compile Tree-Attention Mask
                # Note: context_len is total_len - 1 (since index total_len-1 is now query index 0)
                attention_mask = self.verifier.construct_tree_mask(
                    batch_size=1,
                    context_len=total_len - 1,
                    tree_size=tree_size,
                    mask_ancestors=ancestors,
                    device=device
                )
                
                context_lens = torch.tensor([total_len - 1], dtype=torch.long, device=device)
                
                # Run target model validation pass
                with torch.no_grad():
                    logits = self.target_model(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        block_table=block_table,
                        context_lens=context_lens,
                        k_caches=self.k_caches,
                        v_caches=self.v_caches,
                        attention_mask=attention_mask,
                        is_prefill=False
                    )
                    
                # Free temp blocks used during speculative forward pass
                for temp_blk in temp_blocks:
                    self.cache_manager.free_block(temp_blk)
                    
                # Run greedy path verification
                accepted = self.verifier.verify_greedy(
                    flat_tokens=flat_tokens,
                    paths=paths,
                    target_logits=logits
                )
                
                # Update speculative generation metrics
                # accepted can be e.g. [T1, T12, Target_Prediction] (length 3, meaning 2 speculated tokens accepted)
                # or [root_greedy] (length 1, meaning 0 speculated tokens accepted)
                self.speculative_total_tokens += 2  # we speculated 2 tokens deep on the accepted path
                self.speculative_accepted_tokens += (len(accepted) - 1)
                
                # Append all accepted new tokens and write their KV caches contiguously
                accepted_additions = accepted
                
                for idx, token_id in enumerate(accepted_additions):
                    req.generated_tokens.append(token_id)
                    req.output_queue.put_nowait(token_id)
                    
                    # Contiguous KV cache update forward pass for accepted additions
                    t_pos = total_len + idx
                    req_req_blocks = (t_pos + 1 + self.block_size - 1) // self.block_size
                    if req_req_blocks > len(req.allocated_block_ids):
                        # Allocate block permanently for the request
                        new_blk = self.cache_manager.allocate_block()
                        req.allocated_block_ids.append(new_blk)
                        
                    input_ids_acc = torch.tensor([[token_id]], dtype=torch.long, device=device)
                    position_ids_acc = torch.tensor([[t_pos]], dtype=torch.long, device=device)
                    block_table_acc = torch.tensor([req.allocated_block_ids + [-1] * (32 - len(req.allocated_block_ids))], dtype=torch.long, device=device)
                    context_lens_acc = torch.tensor([t_pos], dtype=torch.long, device=device)
                    
                    with torch.no_grad():
                        self.target_model(
                            input_ids=input_ids_acc,
                            position_ids=position_ids_acc,
                            block_table=block_table_acc,
                            context_lens=context_lens_acc,
                            k_caches=self.k_caches,
                            v_caches=self.v_caches,
                            is_prefill=False
                        )
                        
                    # Insert block into Radix Tree if it is now complete
                    if (t_pos + 1) % self.block_size == 0:
                        block_idx = t_pos // self.block_size
                        block_tokens = tuple((req.prompt_tokens + req.generated_tokens)[block_idx * self.block_size : (block_idx + 1) * self.block_size])
                        phys_blk_id = req.allocated_block_ids[block_idx]
                        
                        new_node = self.cache_manager.insert_block(req.current_node, block_tokens, phys_block_id=phys_blk_id)
                        if new_node not in req.referenced_nodes:
                            req.referenced_nodes.append(new_node)
                            new_node.ref_count += 1
                        req.current_node = new_node
                        
                outputs.append((req, accepted_additions))
                
            else:
                # 2. Standard Decode Path (non-speculative)
                last_token = req.generated_tokens[-1]
                
                # Check block boundaries
                req_req_blocks = (total_len + 1 + self.block_size - 1) // self.block_size
                if req_req_blocks > len(req.allocated_block_ids):
                    new_blk = self.cache_manager.allocate_block()
                    req.allocated_block_ids.append(new_blk)
                    
                input_ids = torch.tensor([[last_token]], dtype=torch.long, device=device)
                position_ids = torch.tensor([[total_len]], dtype=torch.long, device=device)
                block_table = torch.tensor([req.allocated_block_ids + [-1] * (32 - len(req.allocated_block_ids))], dtype=torch.long, device=device)
                context_lens = torch.tensor([total_len], dtype=torch.long, device=device)
                
                with torch.no_grad():
                    logits = self.target_model(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        block_table=block_table,
                        context_lens=context_lens,
                        k_caches=self.k_caches,
                        v_caches=self.v_caches,
                        is_prefill=False
                    )
                    
                next_token = torch.argmax(logits[0, -1, :]).item()
                req.generated_tokens.append(next_token)
                req.output_queue.put_nowait(next_token)
                outputs.append((req, [next_token]))
                
                # Cache insertion on block completion
                if (total_len + 1) % self.block_size == 0:
                    block_idx = total_len // self.block_size
                    block_tokens = tuple((req.prompt_tokens + req.generated_tokens)[block_idx * self.block_size : (block_idx + 1) * self.block_size])
                    phys_blk_id = req.allocated_block_ids[block_idx]
                    
                    new_node = self.cache_manager.insert_block(req.current_node, block_tokens, phys_block_id=phys_blk_id)
                    if new_node not in req.referenced_nodes:
                        req.referenced_nodes.append(new_node)
                        new_node.ref_count += 1
                    req.current_node = new_node
                    
        return outputs
