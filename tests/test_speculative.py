import unittest
import torch
from model import MiniGPTConfig, MiniGPT
from speculative_engine import SpeculativeVerifier

class TestSpeculativeEngine(unittest.TestCase):
    def setUp(self):
        # Config small models
        self.target_config = MiniGPTConfig(vocab_size=100, n_positions=64, n_embd=32, n_layer=1, n_head=1)
        self.draft_config = MiniGPTConfig(vocab_size=100, n_positions=64, n_embd=16, n_layer=1, n_head=1)
        
        self.target_model = MiniGPT(self.target_config)
        self.draft_model = MiniGPT(self.draft_config)
        
        self.verifier = SpeculativeVerifier(self.target_model, self.draft_model)
        
    def test_draft_tree_generation_and_mask(self):
        input_ids = torch.randint(0, 100, (1, 5), dtype=torch.long)
        
        flat_tokens, paths, ancestors = self.verifier.generate_draft_tree(input_ids)
        
        # Verify Flat representation size
        self.assertEqual(len(flat_tokens), 6)
        self.assertEqual(len(paths), 4)
        
        # Construct tree mask
        context_len = 5
        tree_size = 6
        mask = self.verifier.construct_tree_mask(
            batch_size=1,
            context_len=context_len,
            tree_size=tree_size,
            mask_ancestors=ancestors,
            device=torch.device("cpu")
        )
        
        # Expected shape [B, 1, tree_size, context_len + tree_size] = [1, 1, 6, 11]
        self.assertEqual(list(mask.shape), [1, 1, 6, 11])
        
        # Column 0 to 4 (context) must be all True
        self.assertTrue(mask[0, 0, :, :context_len].all())
        
        # Row 2 (T11): represents index 2, ancestors are 0 (T1) and 2 (T11)
        # So column context_len + 0 (which is 5) must be True, column context_len + 1 (6) must be False, column context_len + 2 (7) must be True.
        self.assertTrue(mask[0, 0, 2, 5].item())   # T1 (index 0)
        self.assertFalse(mask[0, 0, 2, 6].item())  # T2 (index 1)
        self.assertTrue(mask[0, 0, 2, 7].item())   # T11 (index 2)
        self.assertFalse(mask[0, 0, 2, 8].item())  # T12 (index 3)

    def test_verify_greedy(self):
        # Mock flat tokens
        flat_tokens = [10, 20, 11, 12, 21, 22]
        # Paths:
        # [0, 2] -> [10, 11]
        # [0, 3] -> [10, 12]
        # [1, 4] -> [20, 21]
        # [1, 5] -> [20, 22]
        paths = [[0, 2], [0, 3], [1, 4], [1, 5]]
        
        # Mock target logits [1, 7, vocab_size]
        # Index 0 is root token prediction.
        # Index 1 is T1 prediction.
        # Index 2 is T2 prediction.
        # Index 3 is T11 prediction.
        # Index 4 is T12 prediction.
        # Index 5 is T21 prediction.
        # Index 6 is T22 prediction.
        vocab_size = 100
        
        # Case 1: Root predicts T1 (10). T1 predicts T12 (12). T12 predicts 99.
        # Target logits should reflect these predictions.
        target_logits = torch.zeros(1, 7, vocab_size)
        target_logits[0, 0, 10] = 10.0  # Root predicts T1 (10)
        target_logits[0, 1, 12] = 10.0  # T1 predicts T12 (12)
        target_logits[0, 4, 99] = 10.0  # T12 predicts 99
        
        accepted_1 = self.verifier.verify_greedy(
            flat_tokens=flat_tokens,
            paths=paths,
            target_logits=target_logits
        )
        self.assertEqual(accepted_1, [10, 12, 99])
        
        # Case 2: Root predicts T2 (20). T2 predicts T21 (21). T21 predicts 88.
        target_logits = torch.zeros(1, 7, vocab_size)
        target_logits[0, 0, 20] = 10.0  # Root predicts T2 (20)
        target_logits[0, 2, 21] = 10.0  # T2 predicts T21 (21)
        target_logits[0, 5, 88] = 10.0  # T21 predicts 88
        
        accepted_2 = self.verifier.verify_greedy(
            flat_tokens=flat_tokens,
            paths=paths,
            target_logits=target_logits
        )
        self.assertEqual(accepted_2, [20, 21, 88])
        
        # Case 3: Root predicts 50 (does not match T1 or T2)
        # We reject the tree and accept just the root greedy token [50]
        target_logits = torch.zeros(1, 7, vocab_size)
        target_logits[0, 0, 50] = 10.0  # Root predicts 50
        
        accepted_3 = self.verifier.verify_greedy(
            flat_tokens=flat_tokens,
            paths=paths,
            target_logits=target_logits
        )
        self.assertEqual(accepted_3, [50])

if __name__ == '__main__':
    unittest.main()
