import unittest
from model import MiniGPTConfig
from engine import LLMEngine

class TestLLMEngine(unittest.TestCase):
    def setUp(self):
        # Configure tiny configurations for test models
        self.target_config = MiniGPTConfig(
            vocab_size=100, n_positions=32, n_embd=16, n_layer=1, n_head=1, block_size=2
        )
        self.draft_config = MiniGPTConfig(
            vocab_size=100, n_positions=32, n_embd=8, n_layer=1, n_head=1, block_size=2
        )
        
    def test_engine_execution_flow(self):
        # Instantiate engine with speculative decoding enabled
        engine = LLMEngine(
            target_config=self.target_config,
            draft_config=self.draft_config,
            num_blocks=10,
            chunk_size=2,
            enable_speculative=True
        )
        
        # Add request with prompt length 4, expecting 3 tokens generated
        req = engine.add_request(prompt_tokens=[10, 20, 30, 40], max_gen_len=3)
        self.assertEqual(req.state, "waiting")
        
        # Iteration 1: Prefills first chunk of length 2 [10, 20]
        outputs_1 = engine.step()
        self.assertEqual(len(outputs_1), 0)
        self.assertEqual(req.state, "prefilling")
        self.assertEqual(req.num_prefilled_tokens, 2)
        
        # Iteration 2: Prefills second chunk of length 2 [30, 40].
        # Completes prefill phase and generates the 1st token.
        outputs_2 = engine.step()
        self.assertEqual(len(outputs_2), 1)
        self.assertEqual(outputs_2[0][0].request_id, req.request_id)
        self.assertEqual(len(outputs_2[0][1]), 1)  # first token
        self.assertEqual(req.state, "running")
        self.assertEqual(len(req.generated_tokens), 1)
        
        # Iteration 3: Runs speculative decoding step for active generation.
        # Should execute draft tree generation, target verification, and return accepted tokens.
        outputs_3 = engine.step()
        self.assertEqual(len(outputs_3), 1)
        self.assertEqual(outputs_3[0][0].request_id, req.request_id)
        # Accepted output can have length >= 1 (since it accepts path additions + next target prediction)
        self.assertTrue(len(outputs_3[0][1]) >= 1)
        
    def test_engine_without_speculative(self):
        # Instantiate engine with speculative decoding disabled
        engine = LLMEngine(
            target_config=self.target_config,
            draft_config=self.draft_config,
            num_blocks=10,
            chunk_size=2,
            enable_speculative=False
        )
        
        req = engine.add_request(prompt_tokens=[10, 20, 30, 40], max_gen_len=2)
        
        # Step 1: prefill first chunk
        engine.step()
        # Step 2: prefill second chunk, generate first token
        engine.step()
        # Step 3: decode next token
        outputs_3 = engine.step()
        self.assertEqual(len(outputs_3), 1)
        self.assertEqual(len(outputs_3[0][1]), 1)  # standard decode generates exactly 1 token
        self.assertEqual(len(req.generated_tokens), 2)

if __name__ == '__main__':
    unittest.main()
