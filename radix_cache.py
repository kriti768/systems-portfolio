import time
from typing import Dict, List, Tuple, Optional

class RadixNode:
    """
    A node in the Radix Tree representing a completed block of tokens in the KV cache.
    """
    def __init__(self, tokens: Tuple[int, ...], phys_block_id: Optional[int], parent: Optional['RadixNode'] = None):
        self.tokens = tokens  # The tokens contained in this block (size <= block_size)
        self.phys_block_id = phys_block_id  # The index of the physical block allocated in the cache
        self.parent = parent
        self.children: Dict[Tuple[int, ...], 'RadixNode'] = {}
        self.ref_count = 0  # Number of active sequences currently holding a reference to this node
        self.last_accessed = 0.0  # Unix timestamp of last access for LRU eviction

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class RadixCacheManager:
    """
    Manages the global pool of physical KV cache blocks and the Radix Tree for prefix caching.
    """
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = set(range(num_blocks))
        
        # Root node does not hold tokens and does not consume a physical block
        self.root = RadixNode(tokens=(), phys_block_id=None)
        
        # Keep a mapping from physical block ID to its RadixNode for debugging/telemetry
        self.block_to_node: Dict[int, RadixNode] = {}
        
    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def allocate_block(self) -> int:
        """
        Allocates a physical block from the free list. If empty, evicts an LRU node.
        """
        if len(self.free_blocks) == 0:
            evicted_block = self.evict_lru_node()
            if evicted_block is None:
                raise RuntimeError("KV Cache Memory Exhausted: No free blocks and no evictable nodes (all blocks are active).")
            return evicted_block
            
        return self.free_blocks.pop()

    def free_block(self, block_id: int):
        """
        Returns a physical block back to the free list.
        """
        self.free_blocks.add(block_id)
        if block_id in self.block_to_node:
            del self.block_to_node[block_id]

    def match_prefix(self, token_ids: List[int]) -> Tuple[List[RadixNode], List[int], List[int]]:
        """
        Finds the longest matched prefix of blocks in the Radix Tree.
        
        Args:
            token_ids (List[int]): The full prompt sequence.
            
        Returns:
            matched_nodes (List[RadixNode]): List of nodes matching the prompt block-by-block.
            matched_block_ids (List[int]): The physical block IDs of the matched nodes.
            remaining_tokens (List[int]): The suffix of tokens that couldn't be matched as completed blocks.
        """
        matched_nodes = []
        matched_block_ids = []
        
        # Chunk prompt into block_size pieces
        num_blocks = len(token_ids) // self.block_size
        blocks = [
            tuple(token_ids[i * self.block_size : (i + 1) * self.block_size])
            for i in range(num_blocks)
        ]
        
        # Suffix tokens that don't make up a full block or come after match failure
        remaining_start_idx = 0
        
        curr = self.root
        for block in blocks:
            if block in curr.children:
                curr = curr.children[block]
                curr.last_accessed = time.time()
                matched_nodes.append(curr)
                matched_block_ids.append(curr.phys_block_id)
                remaining_start_idx += self.block_size
            else:
                break
                
        remaining_tokens = token_ids[remaining_start_idx:]
        return matched_nodes, matched_block_ids, remaining_tokens

    def insert_block(self, parent_node: RadixNode, block_tokens: Tuple[int, ...], phys_block_id: Optional[int] = None) -> RadixNode:
        """
        Inserts a new block of tokens into the Radix Tree under the parent node.
        Uses the provided physical block ID or allocates a new one if not provided.
        """
        assert len(block_tokens) == self.block_size, f"Can only cache full blocks, got size {len(block_tokens)}"
        
        if block_tokens in parent_node.children:
            # Already exists, just return it
            child = parent_node.children[block_tokens]
            child.last_accessed = time.time()
            return child
            
        # Allocate physical block if not provided
        if phys_block_id is None:
            phys_block_id = self.allocate_block()
        
        # Create node
        new_node = RadixNode(tokens=block_tokens, phys_block_id=phys_block_id, parent=parent_node)
        new_node.last_accessed = time.time()
        
        # Link in tree
        parent_node.children[block_tokens] = new_node
        self.block_to_node[phys_block_id] = new_node
        
        return new_node

    def evict_lru_node(self) -> Optional[int]:
        """
        Finds and evicts the Least Recently Used (LRU) leaf node with ref_count == 0.
        Returns the physical block ID of the evicted node.
        """
        best_candidate: Optional[RadixNode] = None
        
        # Helper DFS function to traverse the tree and locate evictable leaves
        def find_candidates(node: RadixNode):
            if node.phys_block_id is not None and node.ref_count == 0 and node.is_leaf():
                nonlocal best_candidate
                if best_candidate is None or node.last_accessed < best_candidate.last_accessed:
                    best_candidate = node
                    
            for child in node.children.values():
                find_candidates(child)
                
        find_candidates(self.root)
        
        if best_candidate is None:
            return None
            
        # Evict candidate
        evicted_block = best_candidate.phys_block_id
        parent = best_candidate.parent
        
        # Remove from parent's children
        if parent is not None:
            del parent.children[best_candidate.tokens]
            
        # Free physical block back to pool
        self.free_block(evicted_block)
        return evicted_block

    def increment_ref(self, nodes: List[RadixNode]):
        """
        Increments reference count of a list of nodes (called when sequence begins processing).
        """
        for node in nodes:
            node.ref_count += 1
            node.last_accessed = time.time()

    def decrement_ref(self, nodes: List[RadixNode]):
        """
        Decrements reference count of a list of nodes (called when sequence completes generation).
        """
        for node in nodes:
            node.ref_count -= 1
            assert node.ref_count >= 0, "Reference count went negative!"
            node.last_accessed = time.time()
            
    def get_tree_structure(self) -> dict:
        """
        Returns a dictionary representation of the Radix Tree for telemetry visualization.
        """
        def serialize_node(node: RadixNode) -> dict:
            return {
                "tokens": list(node.tokens),
                "phys_block_id": node.phys_block_id,
                "ref_count": node.ref_count,
                "last_accessed": node.last_accessed,
                "children": [serialize_node(child) for child in node.children.values()]
            }
        return serialize_node(self.root)
