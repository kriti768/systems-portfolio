import json
import time
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from model import MiniGPTConfig, MiniGPT
import torch

# Fallback tokenizer
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
except Exception:
    class FallbackTokenizer:
        def encode(self, text: str):
            return [min(ord(c), 50255) for c in text]
        def decode(self, tokens):
            return "".join([chr(t) for t in tokens if 0 <= t < 0x110000])
    tokenizer = FallbackTokenizer()

# Create target model configuration (identical to api_server target)
config = MiniGPTConfig(
    vocab_size=50257,
    n_positions=512,
    n_embd=128,
    n_layer=4,
    n_head=4,
    block_size=8,
    bias=True
)

model = MiniGPT(config)
model.eval()

# Lock to ensure strict sequential execution (naive FIFO baseline)
lock = asyncio.Lock()

app = FastAPI(title="Naive HF-Style Serving Server")

@app.get("/generate")
async def generate(prompt: str, max_tokens: int = 64):
    prompt_tokens = tokenizer.encode(prompt)
    device = next(model.parameters()).device
    
    async def sse_generator():
        # Acquire lock to simulate sequential single-threaded generation
        async with lock:
            seq = list(prompt_tokens)
            start_time = time.time()
            first_token_sent = False
            
            for t in range(max_tokens):
                input_ids = torch.tensor([seq], dtype=torch.long, device=device)
                position_ids = torch.arange(len(seq), device=device)[None, :]
                
                # Standard forward pass (recomputes full context every step without caching!)
                with torch.no_grad():
                    logits = model(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        block_table=None,
                        context_lens=None,
                        k_caches=None,
                        v_caches=None,
                        is_prefill=False
                    )
                
                next_token = torch.argmax(logits[0, -1, :]).item()
                seq.append(next_token)
                
                # Metric capture simulation
                if not first_token_sent:
                    ttft = time.time() - start_time
                    first_token_sent = True
                    
                word = tokenizer.decode([next_token])
                yield f"data: {json.dumps({'token': word})}\n\n"
                
                # Yield control briefly to keep server responsive
                await asyncio.sleep(0.005)
                
                if next_token == 50256: # End of text
                    break
                    
            yield f"data: {json.dumps({'status': 'done'})}\n\n"
            
    return StreamingResponse(sse_generator(), media_type="text/event-stream")
