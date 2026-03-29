import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from model import MiniGPT

class SpeculativeVerifier:
    r"""
    Implements Tree-based Speculative Decoding.
    Generates a tree of candidate tokens using the Draft model, compiles a Tree Attention Mask,
    runs target verification, and selects the longest accepted token path.
    """
    def __init__(self, target_model: MiniGPT, draft_model: MiniGPT, temp: float = 0.0):
        self.target_model = target_model
        self.draft_model = draft_model
        self.temp = temp
        
    def generate_draft_tree(self, input_ids: torch.Tensor) -> Tuple[List[int], List[List[int]], List[List[int]]]:
        r"""
        Generates a fixed-topology candidate token tree of width 2, depth 2.
        Tree Topology:
                     [Root (Last Confirmed Token)]
                             /             \
                       T1 (Idx 0)       T2 (Idx 1)
                        /      \         /      \
                      T1,1    T1,2     T2,1    T2,2
                     (Idx 2) (Idx 3)  (Idx 4) (Idx 5)
                     
        Returns:
            flat_tokens (List[int]): The 6 speculated token IDs in tree order.
            paths (List[List[int]]): The token paths through the tree (excluding root).
            mask_ancestors (List[List[int]]): The indices of ancestor tokens for each of the query tokens.
        """
        device = input_ids.device
        
        # 1. Draft step 1: Get top-2 tokens from root
        with torch.no_grad():
            draft_logits = self.draft_model(
                input_ids=input_ids,
                position_ids=torch.arange(input_ids.size(1), device=device)[None, :],
                block_table=None,
                context_lens=None
            )
            probs = F.softmax(draft_logits[0, -1, :], dim=-1)
            top_probs, top_ids = torch.topk(probs, 2)
            t1, t2 = top_ids[0].item(), top_ids[1].item()
            
        # 2. Draft step 2: From t1 and t2, get top-2 child tokens
        with torch.no_grad():
            # Run draft on sequence extended with t1
            ids_t1 = torch.cat([input_ids, torch.tensor([[t1]], device=device)], dim=1)
            draft_logits_t1 = self.draft_model(
                input_ids=ids_t1,
                position_ids=torch.arange(ids_t1.size(1), device=device)[None, :],
                block_table=None,
                context_lens=None
            )
            probs_t1 = F.softmax(draft_logits_t1[0, -1, :], dim=-1)
            top_probs_t1, top_ids_t1 = torch.topk(probs_t1, 2)
            t11, t12 = top_ids_t1[0].item(), top_ids_t1[1].item()
            
            # Run draft on sequence extended with t2
            ids_t2 = torch.cat([input_ids, torch.tensor([[t2]], device=device)], dim=1)
            draft_logits_t2 = self.draft_model(
                input_ids=ids_t2,
                position_ids=torch.arange(ids_t2.size(1), device=device)[None, :],
                block_table=None,
                context_lens=None
            )
            probs_t2 = F.softmax(draft_logits_t2[0, -1, :], dim=-1)
            top_probs_t2, top_ids_t2 = torch.topk(probs_t2, 2)
            t21, t22 = top_ids_t2[0].item(), top_ids_t2[1].item()

        # Compile flat representation
        flat_tokens = [t1, t2, t11, t12, t21, t22]
        
        # Paths mapping index sequence
        paths = [
            [0, 2],
            [0, 3],
            [1, 4],
            [1, 5]
        ]
        
        # Ancestors map query indices (0 to 5) to other query indices they can attend to
        mask_ancestors = [
            [0],
            [1],
            [0, 2],
            [0, 3],
            [1, 4],
            [1, 5]
        ]
        
        return flat_tokens, paths, mask_ancestors

    def construct_tree_mask(
        self,
        batch_size: int,
        context_len: int,
        tree_size: int,
        mask_ancestors: List[List[int]],
        device: torch.device
    ) -> torch.Tensor:
        """
        Constructs the 3D Tree-Attention Mask tensor.
        Shape: [B, 1, tree_size, context_len + tree_size]
        
        The query tokens can attend to:
        1. All context tokens (past history: columns 0 to context_len - 1).
        2. Query tokens that are their valid tree ancestors (columns context_len to context_len + tree_size - 1).
        """
        mask = torch.zeros(batch_size, 1, tree_size, context_len + tree_size, dtype=torch.bool, device=device)
        
        # 1. Allow all query tokens to attend to all context (history) tokens
        mask[:, :, :, :context_len] = True
        
        # 2. Allow query tokens to attend only to their ancestors in the tree query portion
        for q_idx in range(tree_size):
            ancestors = mask_ancestors[q_idx]
            for a_idx in ancestors:
                mask[:, :, q_idx, context_len + a_idx] = True
                
        return mask

    def verify_greedy(
        self,
        flat_tokens: List[int],
        paths: List[List[int]],
        target_logits: torch.Tensor
    ) -> List[int]:
        """
        Validates draft tree paths greedily.
        
        Args:
            flat_tokens (List[int]): The 6 speculated token IDs in tree order.
            paths (List[List[int]]): The 4 paths through the tree.
            target_logits (Tensor): Logits from target model, shape [1, 7, vocab_size]
                                   where index 0 is the prediction after the root token.
                                   
        Returns:
            accepted_tokens (List[int]): The accepted path tokens plus the next target token.
        """
        t1, t2 = flat_tokens[0], flat_tokens[1]
        
        # Logit at index 0 corresponds to the prediction after the root token (position L-1)
        root_logits = target_logits[0, 0]
        root_greedy_token = torch.argmax(root_logits).item()
        
        accepted_path = []
        next_step_node = None
        
        # Check depth 1
        if root_greedy_token == t1:
            accepted_path.append(t1)
            next_step_node = 1  # T1 is query index 1 in target_logits
        elif root_greedy_token == t2:
            accepted_path.append(t2)
            next_step_node = 2  # T2 is query index 2 in target_logits
        else:
            # Reject entire tree, return target prediction at root
            return [root_greedy_token]
            
        # Check depth 2
        # Prediction at next_step_node (T1 or T2)
        node_logits = target_logits[0, next_step_node]
        node_greedy_token = torch.argmax(node_logits).item()
        
        # Children mapping:
        # In query: [root, T1, T2, T11, T12, T21, T22]
        # Indices:   0,    1,  2,   3,   4,   5,   6
        if next_step_node == 1:
            child1, child2 = flat_tokens[2], flat_tokens[3]
            child1_idx, child2_idx = 3, 4
        else:
            child1, child2 = flat_tokens[4], flat_tokens[5]
            child1_idx, child2_idx = 5, 6
            
        if node_greedy_token == child1:
            accepted_path.append(child1)
            last_logits = target_logits[0, child1_idx]
            accepted_path.append(torch.argmax(last_logits).item())
        elif node_greedy_token == child2:
            accepted_path.append(child2)
            last_logits = target_logits[0, child2_idx]
            accepted_path.append(torch.argmax(last_logits).item())
        else:
            # Reject depth 2, append prediction at depth 1
            accepted_path.append(node_greedy_token)
            
        return accepted_path
