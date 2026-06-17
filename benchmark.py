import time
import json
import asyncio
import argparse
import random
from typing import List, Dict
import urllib.request
import urllib.parse

# List of synthetic prompts with varying lengths
PROMPTS = [
    "Write a short story about an astronaut who gets lost in a space station.",
    "Explain the theory of general relativity in simple terms for a child.",
    "What is the difference between supervised and unsupervised learning?",
    "How does a database transaction work and what are the ACID properties?",
    "Write a python script to implement a depth-first search on a graph.",
    "What are the main benefits of using a containerized microservice architecture?",
    "Explain how the internet works from the moment I type a URL into a browser.",
    "Describe the process of photosynthesis and how plants convert light to energy."
]

async def send_request(url: str, prompt: str, max_tokens: int) -> Dict:
    """
    Sends a request to the server and parses the SSE stream to measure latency metrics.
    """
    start_time = time.perf_counter()
    ttft = None
    first_token_time = None
    tokens_generated = 0
    
    # Query parameters
    params = urllib.parse.urlencode({"prompt": prompt, "max_tokens": max_tokens})
    full_url = f"{url}?{params}"
    
    loop = asyncio.get_event_loop()
    
    def fetch():
        try:
            req = urllib.request.Request(full_url)
            with urllib.request.urlopen(req) as response:
                return response.read().decode('utf-8')
        except Exception as e:
            return f"Error: {e}"
            
    # Run the blocking urllib request in an executor
    response_text = await loop.run_in_executor(None, fetch)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Parse SSE output to reconstruct tokens and times
    lines = response_text.split('\n')
    token_times = []
    
    for line in lines:
        if line.startswith('data: '):
            try:
                data = json.loads(line[6:])
                if 'token' in data:
                    tokens_generated += 1
                    token_times.append(time.perf_counter())
            except Exception:
                pass
                
    if len(token_times) > 0:
        # Estimate TTFT as time to first token event
        ttft = token_times[0] - start_time
        # Inter-token latency (average of step differences)
        if len(token_times) > 1:
            itl = (token_times[-1] - token_times[0]) / (len(token_times) - 1)
        else:
            itl = 0.0
    else:
        ttft = total_time
        itl = 0.0
        
    return {
        "prompt_len": len(prompt.split()),
        "tokens_generated": tokens_generated,
        "ttft": ttft,
        "itl": itl,
        "total_latency": total_time,
        "throughput": tokens_generated / total_time if total_time > 0 else 0
    }

async def run_benchmark(server_url: str, num_requests: int, concurrency: int) -> List[Dict]:
    """
    Runs concurrent client workload requests against the serving endpoint.
    """
    sem = asyncio.Semaphore(concurrency)
    results = []
    
    async def worker(prompt: str):
        async with sem:
            res = await send_request(server_url, prompt, max_tokens=32)
            results.append(res)
            
    tasks = []
    for _ in range(num_requests):
        prompt = random.choice(PROMPTS)
        tasks.append(asyncio.create_task(worker(prompt)))
        
    await asyncio.gather(*tasks)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port-opt", type=int, default=8000, help="Port of optimized engine server")
    parser.add_argument("--port-naive", type=int, default=8001, help="Port of naive server")
    parser.add_argument("--requests", type=int, default=15, help="Total requests to send")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrency limit")
    args = parser.parse_args()
    
    print("=== STARTING BENCHMARK RUN ===")
    print(f"Workload: {args.requests} requests, Concurrency: {args.concurrency}")
    
    # 1. Benchmark Naive Server
    print("\n[Naive Server] Running benchmark...")
    url_naive = f"http://localhost:{args.port_naive}/generate"
    try:
        results_naive = asyncio.run(run_benchmark(url_naive, args.requests, args.concurrency))
        avg_ttft_n = sum(r['ttft'] for r in results_naive) / len(results_naive)
        avg_itl_n = sum(r['itl'] for r in results_naive) / len(results_naive)
        avg_lat_n = sum(r['total_latency'] for r in results_naive) / len(results_naive)
        total_tokens_n = sum(r['tokens_generated'] for r in results_naive)
        throughput_n = total_tokens_n / sum(r['total_latency'] for r in results_naive)
        print(f"  Avg TTFT: {avg_ttft_n:.4f}s")
        print(f"  Avg ITL:  {avg_itl_n:.4f}s")
        print(f"  Avg Latency: {avg_lat_n:.4f}s")
        print(f"  Throughput: {throughput_n:.2f} tokens/s")
    except Exception as e:
        print(f"  Failed to run naive benchmark: {e}")
        results_naive = []
        
    # 2. Benchmark Optimized Engine
    print("\n[Optimized Engine] Running benchmark...")
    url_opt = f"http://localhost:{args.port_opt}/generate"
    try:
        results_opt = asyncio.run(run_benchmark(url_opt, args.requests, args.concurrency))
        avg_ttft_o = sum(r['ttft'] for r in results_opt) / len(results_opt)
        avg_itl_o = sum(r['itl'] for r in results_opt) / len(results_opt)
        avg_lat_o = sum(r['total_latency'] for r in results_opt) / len(results_opt)
        total_tokens_o = sum(r['tokens_generated'] for r in results_opt)
        throughput_o = total_tokens_o / sum(r['total_latency'] for r in results_opt)
        print(f"  Avg TTFT: {avg_ttft_o:.4f}s")
        print(f"  Avg ITL:  {avg_itl_o:.4f}s")
        print(f"  Avg Latency: {avg_lat_o:.4f}s")
        print(f"  Throughput: {throughput_o:.2f} tokens/s")
    except Exception as e:
        print(f"  Failed to run optimized benchmark: {e}")
        results_opt = []
        
    # Save results
    summary = {
        "naive": results_naive,
        "optimized": results_opt
    }
    with open("benchmark_results.json", "w") as f:
        json.dump(summary, f, indent=4)
    print("\nResults saved to benchmark_results.json")

if __name__ == "__main__":
    main()
