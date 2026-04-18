import unittest
from radix_cache import RadixCacheManager
from scheduler import Request, ContinuousScheduler

class TestScheduler(unittest.TestCase):
    def test_chunked_prefill_and_transition(self):
        # 4 physical blocks, block_size = 2
        manager = RadixCacheManager(num_blocks=4, block_size=2)
        # chunk_size = 2
        scheduler = ContinuousScheduler(manager, chunk_size=2)
        
        # Request with prompt of length 4 (requires 2 chunks of size 2)
        req = Request(prompt_tokens=[1, 2, 3, 4], max_gen_len=2)
        scheduler.add_request(req)
        
        # 1. First iteration: Prefill first chunk [1, 2]
        prefill, decode = scheduler.schedule()
        self.assertEqual(len(prefill), 1)
        self.assertEqual(len(decode), 0)
        self.assertEqual(prefill[0].request_id, req.request_id)
        self.assertEqual(req.state, "prefilling")
        
        # Update progress simulation (representing model execution)
        req.num_prefilled_tokens += 2
        self.assertEqual(len(req.allocated_block_ids), 1)
        
        # 2. Second iteration: Prefill second chunk [3, 4]
        prefill, decode = scheduler.schedule()
        self.assertEqual(len(prefill), 1)
        self.assertEqual(len(decode), 0)
        
        # Complete prompt prefill
        req.num_prefilled_tokens += 2
        req.state = "running"  # transition to decode
        self.assertEqual(len(req.allocated_block_ids), 2)
        
        # 3. Third iteration: Decode step
        prefill, decode = scheduler.schedule()
        self.assertEqual(len(prefill), 0)
        self.assertEqual(len(decode), 1)
        self.assertEqual(decode[0].request_id, req.request_id)
        
    def test_preemption(self):
        # Only 2 blocks, block_size = 2
        manager = RadixCacheManager(num_blocks=2, block_size=2)
        scheduler = ContinuousScheduler(manager, chunk_size=2)
        
        # Request A: prompt length 2, needs 1 block
        req_A = Request(prompt_tokens=[10, 20], max_gen_len=5, request_id="A")
        # Request B: prompt length 2, needs 1 block
        req_B = Request(prompt_tokens=[30, 40], max_gen_len=5, request_id="B")
        
        scheduler.add_request(req_A)
        
        # Prefill Request A
        prefill, _ = scheduler.schedule()
        self.assertEqual(len(prefill), 1)
        req_A.num_prefilled_tokens += 2
        req_A.state = "running"
        self.assertEqual(manager.get_num_free_blocks(), 1)
        
        # Now add Request B and prefill it
        scheduler.add_request(req_B)
        prefill, _ = scheduler.schedule()
        self.assertEqual(len(prefill), 1)
        req_B.num_prefilled_tokens += 2
        req_B.state = "running"
        self.assertEqual(manager.get_num_free_blocks(), 0)
        
        # Both req_A and req_B are in "running" queue.
        # Both have sequence length 2 (which is multiple of block_size 2).
        # In the next step, both need to generate a token, crossing block boundary,
        # requiring 1 new block each (2 blocks total).
        # However, 0 blocks are free!
        # Scheduler must preempt requests to reclaim memory.
        # Since running list has [req_A, req_B], the scheduler will preempt req_B (newest/LIFO).
        
        prefill, decode = scheduler.schedule()
        
        # Verify req_B was preempted and req_A remains scheduled for decode
        self.assertEqual(req_B.state, "preempted")
        self.assertEqual(len(req_B.allocated_block_ids), 0)
        self.assertEqual(req_A.state, "running")
        self.assertEqual(len(decode), 1)
        self.assertEqual(decode[0].request_id, "A")
        
        # Reclaimed block from B allows A to run
        self.assertEqual(manager.get_num_free_blocks(), 1)

if __name__ == '__main__':
    unittest.main()
