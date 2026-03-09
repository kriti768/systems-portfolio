import time
import uuid
import math
import asyncio
from typing import List, Dict, Tuple, Optional
from radix_cache import RadixNode, RadixCacheManager

class Request:
    """
    Represents an inference request managed by the serving engine.
    """
    def __init__(self, prompt_tokens: List[int], max_gen_len: int = 64, request_id: Optional[str] = None):
        self.request_id = request_id or str(uuid.uuid4())
        self.prompt_tokens = prompt_tokens
        self.max_gen_len = max_gen_len
        self.generated_tokens: List[int] = []
        
        # State machine: "waiting", "prefilling", "running", "preempted", "completed"
        self.state = "waiting"
        
        # Telemetry metrics
        self.arrival_time = time.time()
        self.start_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self.completion_time: Optional[float] = None
        
        # Prefill tracking (for chunked prefill)
        self.num_prefilled_tokens = 0
        
        # KV Cache tracking
        self.allocated_block_ids: List[int] = []
        # List of RadixNodes in the prefix tree that this request holds references to (and increments ref_count)
        self.referenced_nodes: List[RadixNode] = []
        # Pointer to the last completed block in the Radix Tree
        self.current_node: Optional[RadixNode] = None
        
        # Output queue for streaming tokens (used by API server)
        self.output_queue = asyncio.Queue()

    def get_total_tokens(self) -> int:
        return self.num_prefilled_tokens + len(self.generated_tokens)

    def is_finished(self) -> bool:
        return len(self.generated_tokens) >= self.max_gen_len or (
            len(self.generated_tokens) > 0 and self.generated_tokens[-1] == 50256 # end-of-text token for GPT-2
        )


class ContinuousScheduler:
    """
    Manages request queues and makes iteration-level scheduling decisions based on KV cache budget.
    Supports continuous batching, preemption, and chunked prefill.
    """
    def __init__(self, cache_manager: RadixCacheManager, chunk_size: int = 16):
        self.cache_manager = cache_manager
        self.chunk_size = chunk_size
        self.block_size = cache_manager.block_size
        
        # Queues
        self.waiting_queue: List[Request] = []
        self.running_queue: List[Request] = []
        self.preempted_queue: List[Request] = []

    def add_request(self, request: Request):
        self.waiting_queue.append(request)

    def preempt_request(self, request: Request):
        """
        Preempts an active request, freeing its blocks and returning it to the preempted queue.
        """
        request.state = "preempted"
        
        # Decrement references for all nodes in the Radix Tree
        self.cache_manager.decrement_ref(request.referenced_nodes)
        
        # Free private block IDs (those not matching physical block IDs in referenced nodes)
        referenced_block_ids = {node.phys_block_id for node in request.referenced_nodes}
        for block_id in request.allocated_block_ids:
            if block_id not in referenced_block_ids:
                self.cache_manager.free_block(block_id)
                
        # Clear allocations for recomputation-based preemption
        request.allocated_block_ids = []
        request.referenced_nodes = []
        request.current_node = None
        request.num_prefilled_tokens = 0
        
        # Move request back to the preempted queue (at the front to maintain priority)
        self.running_queue.remove(request)
        self.preempted_queue.insert(0, request)

    def complete_request(self, request: Request):
        """
        Finalizes a request, freeing its cache references and cleaning up.
        """
        request.state = "completed"
        request.completion_time = time.time()
        
        # Decrement references for all shared cache blocks in Radix Tree
        self.cache_manager.decrement_ref(request.referenced_nodes)
        
        # Free any private trailing block IDs
        referenced_block_ids = {node.phys_block_id for node in request.referenced_nodes}
        for block_id in request.allocated_block_ids:
            if block_id not in referenced_block_ids:
                self.cache_manager.free_block(block_id)
                
        self.running_queue.remove(request)

    def schedule(self) -> Tuple[List[Request], List[Request]]:
        """
        Determines which requests run in the next batch step.
        
        Returns:
            prefill_requests (List[Request]): Requests scheduled to run prefill chunks.
            decode_requests (List[Request]): Requests scheduled to run decode steps.
        """
        # Step 1: Manage running queue memory overhead.
        # Check if active decode requests will exceed the available block count.
        decode_requests = [req for req in self.running_queue if req.state == "running"]
        prefilling_requests = [req for req in self.running_queue if req.state == "prefilling"]
        
        # Count blocks that will be needed by active decoders
        blocks_needed = 0
        for req in decode_requests:
            total_tokens = req.get_total_tokens()
            # If generating one more token crosses a block boundary, we need a new block
            if total_tokens % self.block_size == 0:
                blocks_needed += 1
                
        # If block limit is exceeded, preempt running requests (LIFO: newest preempted first)
        preempted_in_this_step = False
        while blocks_needed > self.cache_manager.get_num_free_blocks() and len(decode_requests) > 0:
            preempt_target = decode_requests.pop()
            self.preempt_request(preempt_target)
            preempted_in_this_step = True
            
            # Recalculate blocks needed after preemption
            blocks_needed = 0
            for req in decode_requests:
                total_tokens = req.get_total_tokens()
                if total_tokens % self.block_size == 0:
                    blocks_needed += 1
                    
        # Step 2: Schedule new, preempted, or active prefilling requests if no preemption occurred.
        prefill_requests: List[Request] = []
        
        # If we had to preempt active decoders to save memory, do not schedule any prefill tasks in this iteration.
        if preempted_in_this_step:
            return prefill_requests, decode_requests
            
        # Prioritize active prefilling requests, then preempted, then waiting
        candidates = prefilling_requests + self.preempted_queue + self.waiting_queue
        
        for req in list(candidates):
            if req.state == "waiting" or req.state == "preempted":
                # Check for prefix cache matching
                matched_nodes, matched_block_ids, remaining_tokens = self.cache_manager.match_prefix(req.prompt_tokens)
                
                # Setup request with matched blocks
                req.referenced_nodes = list(matched_nodes)
                self.cache_manager.increment_ref(req.referenced_nodes)
                req.allocated_block_ids = list(matched_block_ids)
                req.current_node = matched_nodes[-1] if len(matched_nodes) > 0 else self.cache_manager.root
                req.num_prefilled_tokens = len(matched_nodes) * self.block_size
                
                # Move from wait to prefilling
                req.state = "prefilling"
                req.start_time = time.time()
                
                if req in self.waiting_queue:
                    self.waiting_queue.remove(req)
                elif req in self.preempted_queue:
                    self.preempted_queue.remove(req)
                self.running_queue.append(req)
                
            if req.state == "prefilling":
                # Determine how many tokens are in the next chunk
                prompt_len = len(req.prompt_tokens)
                chunk_len = min(self.chunk_size, prompt_len - req.num_prefilled_tokens)
                
                if chunk_len > 0:
                    # Calculate block overhead for this prefill chunk
                    future_tokens = req.num_prefilled_tokens + chunk_len
                    required_blocks = math.ceil(future_tokens / self.block_size)
                    blocks_to_allocate = required_blocks - len(req.allocated_block_ids)
                    
                    if blocks_to_allocate <= self.cache_manager.get_num_free_blocks():
                        # Allocate the physical blocks
                        for _ in range(blocks_to_allocate):
                            blk_id = self.cache_manager.allocate_block()
                            req.allocated_block_ids.append(blk_id)
                            
                        prefill_requests.append(req)
                    else:
                        # Insufficient memory to schedule this prefill chunk.
                        # Revert its prefill state to waiting/preempted to try again in the next iteration.
                        self.preempt_request(req)
                        break
                else:
                    # No more tokens to prefill, transition to decoding state
                    req.state = "running"
                        
        return prefill_requests, decode_requests
