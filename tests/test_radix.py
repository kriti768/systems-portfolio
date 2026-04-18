import unittest
import time
from radix_cache import RadixCacheManager, RadixNode

class TestRadixCache(unittest.TestCase):
    def test_prefix_matching_and_insertion(self):
        # 3 physical blocks, block_size = 2
        manager = RadixCacheManager(num_blocks=3, block_size=2)
        
        # Match empty tree
        prompt = [10, 20, 30, 40, 50]
        matched_nodes, matched_block_ids, remaining = manager.match_prefix(prompt)
        
        self.assertEqual(len(matched_nodes), 0)
        self.assertEqual(len(matched_block_ids), 0)
        self.assertEqual(remaining, [10, 20, 30, 40, 50])
        
        # Insert first block: (10, 20) under root
        n1 = manager.insert_block(manager.root, (10, 20))
        self.assertIsNotNone(n1.phys_block_id)
        self.assertEqual(manager.get_num_free_blocks(), 2)
        
        # Insert second block: (30, 40) under n1
        n2 = manager.insert_block(n1, (30, 40))
        self.assertIsNotNone(n2.phys_block_id)
        self.assertEqual(manager.get_num_free_blocks(), 1)
        
        # Now try to match prompt again
        matched_nodes, matched_block_ids, remaining = manager.match_prefix(prompt)
        self.assertEqual(len(matched_nodes), 2)
        self.assertEqual(matched_block_ids, [n1.phys_block_id, n2.phys_block_id])
        self.assertEqual(remaining, [50])  # incomplete block token
        
    def test_lru_eviction(self):
        # 3 physical blocks, block_size = 2
        manager = RadixCacheManager(num_blocks=3, block_size=2)
        
        # Fill the cache completely
        # Tree Structure:
        # root -> block A (10, 20) [phys_block_id = A] -> block B (30, 40) [phys_block_id = B]
        # root -> block C (50, 60) [phys_block_id = C]
        
        node_A = manager.insert_block(manager.root, (10, 20))
        node_A.last_accessed = 1000.0  # mock timestamp
        
        node_B = manager.insert_block(node_A, (30, 40))
        node_B.last_accessed = 1002.0  # mock timestamp
        
        node_C = manager.insert_block(manager.root, (50, 60))
        node_C.last_accessed = 1001.0  # mock timestamp
        
        # Free blocks should be empty now
        self.assertEqual(manager.get_num_free_blocks(), 0)
        
        # Nodes B and C are leaves. Node A is a parent (non-leaf).
        # Set all ref_counts to 0
        node_A.ref_count = 0
        node_B.ref_count = 0
        node_C.ref_count = 0
        
        # Node C is a leaf and is older than node B (1001.0 < 1002.0)
        # Node A has children, so it should NOT be evicted despite having the lowest timestamp (1000.0)
        # So Node C should be evicted first.
        
        evicted = manager.evict_lru_node()
        self.assertEqual(evicted, node_C.phys_block_id)
        self.assertNotIn((50, 60), manager.root.children)
        self.assertEqual(manager.get_num_free_blocks(), 1)
        
        # Now Node B is the only leaf left with ref_count 0 (since C is gone, and A still has child B).
        # If we evict again, it should evict Node B.
        evicted_2 = manager.evict_lru_node()
        self.assertEqual(evicted_2, node_B.phys_block_id)
        self.assertNotIn((30, 40), node_A.children)
        
        # Now Node A has no children left (it is now a leaf node with ref_count 0).
        # If we evict again, it should evict A.
        evicted_3 = manager.evict_lru_node()
        self.assertEqual(evicted_3, node_A.phys_block_id)
        self.assertNotIn((10, 20), manager.root.children)
        self.assertEqual(manager.get_num_free_blocks(), 3)

if __name__ == '__main__':
    unittest.main()
