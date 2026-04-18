import unittest
import torch
from model import MiniGPTConfig, MiniGPT, write_to_paged_cache, read_from_paged_cache

class TestMiniGPT(unittest.TestCase):
    def setUp(self):
        # Configure a small model for testing
        self.config = MiniGPTConfig(
            vocab_size=1000,
            n_positions=128,
            n_embd=64,
            n_layer=2,
            n_head=2,
            block_size=4,  # small block size for easier testing
            bias=True
        )
        self.model = MiniGPT(self.config)
        self.model.eval()
        
        # Dimensions
        self.num_blocks = 10
        self.head_dim = self.config.n_embd // self.config.n_head
        
        # Allocate global KV cache tensors for 2 layers
        self.k_caches = [
            torch.zeros(self.num_blocks, self.config.n_head, self.config.block_size, self.head_dim)
            for _ in range(self.config.n_layer)
        ]
        self.v_caches = [
            torch.zeros(self.num_blocks, self.config.n_head, self.config.block_size, self.head_dim)
            for _ in range(self.config.n_layer)
        ]
        
    def test_cache_write_and_read(self):
        batch_size = 2
        num_new_tokens = 3
        
        # Allocate key and value tensors
        key = torch.randn(batch_size, self.config.n_head, num_new_tokens, self.head_dim)
        value = torch.randn(batch_size, self.config.n_head, num_new_tokens, self.head_dim)
        
        # Block table mapping:
        # seq 0 uses physical blocks 1 and 3
        # seq 1 uses physical blocks 2 and 5
        block_table = torch.tensor([
            [1, 3, -1],
            [2, 5, -1]
        ], dtype=torch.long)
        
        # Context lens (both start at 0)
        context_lens = torch.tensor([0, 0], dtype=torch.long)
        
        # Write to physical cache
        write_to_paged_cache(
            self.k_caches[0], self.v_caches[0], key, value, block_table, context_lens, self.config.block_size
        )
        
        # Verify block writes:
        # For sequence 0: tokens 0, 1, 2 should go to block 1 slots 0, 1, 2.
        # For sequence 1: tokens 0, 1, 2 should go to block 2 slots 0, 1, 2.
        for t in range(num_new_tokens):
            torch.testing.assert_close(self.k_caches[0][1, :, t, :], key[0, :, t, :])
            torch.testing.assert_close(self.v_caches[0][1, :, t, :], value[0, :, t, :])
            
            torch.testing.assert_close(self.k_caches[0][2, :, t, :], key[1, :, t, :])
            torch.testing.assert_close(self.v_caches[0][2, :, t, :], value[1, :, t, :])
            
        # Read back from cache
        # If we query with context_lens = 3, we should get shape [batch_size, n_head, 3, head_dim]
        read_lens = torch.tensor([3, 3], dtype=torch.long)
        K_read, V_read = read_from_paged_cache(
            self.k_caches[0], self.v_caches[0], block_table, read_lens, 3, self.config.block_size
        )
        
        self.assertEqual(K_read.shape, (batch_size, self.config.n_head, 3, self.head_dim))
        torch.testing.assert_close(K_read[0, :, :3, :], key[0])
        torch.testing.assert_close(K_read[1, :, :3, :], key[1])
        
    def test_model_forward_prefill_and_decode(self):
        # 1. PREFILL phase
        # Batch of 2 sequences, prompt lengths 3 and 2
        # Input IDs padded to max length 3
        input_ids = torch.tensor([
            [101, 102, 103],
            [201, 202, 0]      # 0 is padding
        ], dtype=torch.long)
        
        position_ids = torch.tensor([
            [0, 1, 2],
            [0, 1, 0]          # pos ids for valid tokens
        ], dtype=torch.long)
        
        # Block table
        block_table = torch.tensor([
            [0, 1],
            [2, 3]
        ], dtype=torch.long)
        
        # Prefill starting offsets (always 0 at start)
        context_lens = torch.tensor([0, 0], dtype=torch.long)
        
        # Run prefill forward pass
        logits_prefill = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            block_table=block_table,
            context_lens=context_lens,
            k_caches=self.k_caches,
            v_caches=self.v_caches,
            is_prefill=True
        )
        
        # Logits shape should be [batch_size, seq_len, vocab_size]
        self.assertEqual(logits_prefill.shape, (2, 3, 1000))
        
        # 2. DECODE phase (T=1)
        # Next token inputs (single token)
        next_input_ids = torch.tensor([
            [104],
            [203]
        ], dtype=torch.long)
        
        next_position_ids = torch.tensor([
            [3],
            [2]
        ], dtype=torch.long)
        
        # Context lens are now the prompt lengths: 3 and 2
        context_lens_decode = torch.tensor([3, 2], dtype=torch.long)
        
        # Run decode step
        logits_decode = self.model(
            input_ids=next_input_ids,
            position_ids=next_position_ids,
            block_table=block_table,
            context_lens=context_lens_decode,
            k_caches=self.k_caches,
            v_caches=self.v_caches,
            is_prefill=False
        )
        
        self.assertEqual(logits_decode.shape, (2, 1, 1000))

if __name__ == '__main__':
    unittest.main()
