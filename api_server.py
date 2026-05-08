import json
import time
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Set
from model import MiniGPTConfig
from engine import LLMEngine
from scheduler import Request

# Fallback tokenizer to ensure zero internet dependency
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("[Tokenizer] Loaded Hugging Face GPT-2 tokenizer.")
except Exception as e:
    print(f"[Tokenizer] Hugging Face loading failed ({e}). Falling back to Character tokenizer.")
    class FallbackTokenizer:
        def encode(self, text: str) -> List[int]:
            # Map chars to ASCII/unicode values, clamp to vocab size 50257
            return [min(ord(c), 50255) for c in text]
        def decode(self, tokens: List[int]) -> str:
            return "".join([chr(t) for t in tokens if 0 <= t < 0x110000])
    tokenizer = FallbackTokenizer()

# Create model configs
# Small sizes suitable for fast CPU generation
target_config = MiniGPTConfig(
    vocab_size=50257,
    n_positions=512,
    n_embd=128,
    n_layer=4,
    n_head=4,
    block_size=8,
    bias=True
)

draft_config = MiniGPTConfig(
    vocab_size=50257,
    n_positions=512,
    n_embd=64,
    n_layer=1,
    n_head=2,
    block_size=8,
    bias=True
)

# Instantiate engine
engine = LLMEngine(
    target_config=target_config,
    draft_config=draft_config,
    num_blocks=32,
    chunk_size=8,
    enable_speculative=True
)

app = FastAPI(title="Mini-vLLM Serving Engine")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Websocket connections manager for real-time telemetry streaming
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()

# Background asyncio engine execution loop
async def serving_loop():
    print("[Engine] Starting serving execution loop...")
    while True:
        try:
            # Execute one step of scheduling & generation
            outputs = engine.step()
            
            # Send live telemetry updates to dashboard if connections exist
            if manager.active_connections:
                await send_telemetry_update()
                
            # Dynamic batch speed: sleep 10ms if requests are active, 100ms when idle
            active_requests = len(engine.scheduler.running_queue)
            if active_requests > 0:
                await asyncio.sleep(0.01)
            else:
                await asyncio.sleep(0.1)
        except Exception as e:
            print(f"[Engine Loop Error] {e}")
            await asyncio.sleep(0.5)

async def send_telemetry_update():
    # Gather state descriptions
    scheduler = engine.scheduler
    cache = engine.cache_manager
    
    # 1. KV cache blocks status
    # We want to categorize each of the num_blocks blocks:
    # - Free: in cache.free_blocks
    # - Active: allocated_block_ids of running/prefilling requests
    # - Shared/Cached: occupied by Radix Tree nodes (ref_count == 0 but not evicted, or ref_count > 0)
    block_status = ["free"] * cache.num_blocks
    
    # Radix cached nodes
    for blk_id, node in cache.block_to_node.items():
        if node.ref_count > 0:
            block_status[blk_id] = "active_shared"
        else:
            block_status[blk_id] = "cached_shared"
            
    # Private blocks in active requests
    for req in scheduler.running_queue:
        shared_ids = {node.phys_block_id for node in req.referenced_nodes}
        for blk_id in req.allocated_block_ids:
            if blk_id not in shared_ids:
                block_status[blk_id] = "active_private"
                
    # 2. Telemetry metrics
    completed = engine.completed_requests
    avg_ttft = 0.0
    avg_itl = 0.0
    
    if len(completed) > 0:
        ttfts = [req.first_token_time - req.arrival_time for req in completed if req.first_token_time]
        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
        
        itls = []
        for req in completed:
            if req.first_token_time and req.completion_time and len(req.generated_tokens) > 1:
                total_gen_time = req.completion_time - req.first_token_time
                itls.append(total_gen_time / (len(req.generated_tokens) - 1))
        avg_itl = sum(itls) / len(itls) if itls else 0.0

    spec_acc_rate = 0.0
    if engine.speculative_total_tokens > 0:
        spec_acc_rate = engine.speculative_accepted_tokens / engine.speculative_total_tokens
        
    payload = {
        "step": engine.step_count,
        "queues": {
            "waiting": len(scheduler.waiting_queue),
            "running": len(scheduler.running_queue),
            "preempted": len(scheduler.preempted_queue),
            "completed": len(completed)
        },
        "cache": {
            "num_blocks": cache.num_blocks,
            "free_blocks": len(cache.free_blocks),
            "block_status": block_status,
            "radix_tree": cache.get_tree_structure()
        },
        "metrics": {
            "avg_ttft_seconds": round(avg_ttft, 4),
            "avg_itl_seconds": round(avg_itl, 4),
            "spec_acceptance_rate": round(spec_acc_rate, 4),
            "spec_total_tokens": engine.speculative_total_tokens,
            "spec_accepted_tokens": engine.speculative_accepted_tokens
        },
        "requests": [
            {
                "id": req.request_id[:8],
                "state": req.state,
                "prompt_len": len(req.prompt_tokens),
                "generated_len": len(req.generated_tokens),
                "allocated_blocks": len(req.allocated_block_ids)
            }
            for req in scheduler.running_queue
        ]
    }
    
    await manager.broadcast(json.dumps(payload))

# SSE (Server-Sent Events) Streaming Generation Endpoint
@app.get("/generate")
def generate(prompt: str, max_tokens: int = 64):
    prompt_tokens = tokenizer.encode(prompt)
    req = engine.add_request(prompt_tokens, max_tokens)
    
    async def sse_generator():
        while not req.is_finished() or not req.output_queue.empty():
            try:
                # Poll queue for new tokens
                token = await asyncio.wait_for(req.output_queue.get(), timeout=1.0)
                word = tokenizer.decode([token])
                yield f"data: {json.dumps({'token': word})}\n\n"
                req.output_queue.task_done()
            except asyncio.TimeoutError:
                if req.is_finished():
                    break
        # Final status
        yield f"data: {json.dumps({'status': 'done'})}\n\n"
        
    return StreamingResponse(sse_generator(), media_type="text/event-stream")

# Prometheus Metrics Scraping Endpoint (Zero-cost, custom-formatted plain text)
@app.get("/metrics")
def metrics():
    scheduler = engine.scheduler
    cache = engine.cache_manager
    
    # Calculate averages
    completed = engine.completed_requests
    avg_ttft = 0.0
    if len(completed) > 0:
        ttfts = [req.first_token_time - req.arrival_time for req in completed if req.first_token_time]
        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
        
    spec_acc_rate = 0.0
    if engine.speculative_total_tokens > 0:
        spec_acc_rate = engine.speculative_accepted_tokens / engine.speculative_total_tokens

    lines = [
        "# HELP vllm_request_queue_size Number of requests in the queues",
        "# TYPE vllm_request_queue_size gauge",
        f'vllm_request_queue_size{{state="waiting"}} {len(scheduler.waiting_queue)}',
        f'vllm_request_queue_size{{state="running"}} {len(scheduler.running_queue)}',
        f'vllm_request_queue_size{{state="preempted"}} {len(scheduler.preempted_queue)}',
        f'vllm_request_queue_size{{state="completed"}} {len(completed)}',
        "",
        "# HELP vllm_kv_cache_usage_ratio Ratio of allocated KV blocks to total blocks",
        "# TYPE vllm_kv_cache_usage_ratio gauge",
        f"vllm_kv_cache_usage_ratio {(cache.num_blocks - len(cache.free_blocks)) / cache.num_blocks}",
        "",
        "# HELP vllm_time_to_first_token_seconds Average time to first token of completed requests",
        "# TYPE vllm_time_to_first_token_seconds gauge",
        f"vllm_time_to_first_token_seconds {avg_ttft}",
        "",
        "# HELP vllm_speculative_acceptance_rate Speculative draft acceptance rate",
        "# TYPE vllm_speculative_acceptance_rate gauge",
        f"vllm_speculative_acceptance_rate {spec_acc_rate}",
    ]
    return HTMLResponse(content="\n".join(lines), media_type="text/plain")

from fastapi import Request as APIRequest

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    try:
        with open("templates/dashboard.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), media_type="text/html")
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Dashboard HTML template not found.</h1>", status_code=404)

@app.post("/submit_prompt")
async def submit_prompt(request: APIRequest):
    data = await request.json()
    prompt = data.get("prompt", "Hello AI")
    max_tokens = int(data.get("max_tokens", 64))
    prompt_tokens = tokenizer.encode(prompt)
    req = engine.add_request(prompt_tokens, max_tokens)
    return {"request_id": req.request_id, "status": "enqueued"}

@app.post("/toggle_speculative")
async def toggle_speculative(request: APIRequest):
    data = await request.json()
    enable = bool(data.get("enable", True))
    engine.enable_speculative = enable
    return {"speculative_enabled": engine.enable_speculative}

@app.post("/clear_completed")
def clear_completed():
    engine.completed_requests = []
    engine.speculative_accepted_tokens = 0
    engine.speculative_total_tokens = 0
    return {"status": "cleared"}

@app.websocket("/api/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Keep connection open
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Start background engine loop on startup
@app.on_event("startup")
def startup_event():
    asyncio.create_task(serving_loop())
