import json
import os

def main():
    if not os.path.exists("benchmark_results.json"):
        print("Error: benchmark_results.json not found. Run benchmark.py first.")
        return

    with open("benchmark_results.json", "r") as f:
        data = json.load(f)

    naive = data.get("naive", [])
    optimized = data.get("optimized", [])

    if not naive or not optimized:
        print("Error: Empty benchmark results for either naive or optimized servers.")
        return

    # Helper calculations
    def calculate_stats(results):
        count = len(results)
        tot_tokens = sum(r['tokens_generated'] for r in results)
        tot_time = sum(r['total_latency'] for r in results)
        avg_ttft = sum(r['ttft'] for r in results) / count
        avg_itl = sum(r['itl'] for r in results) / count
        avg_lat = sum(r['total_latency'] for r in results) / count
        throughput = tot_tokens / tot_time if tot_time > 0 else 0
        return {
            "count": count,
            "tokens": tot_tokens,
            "ttft": avg_ttft,
            "itl": avg_itl,
            "latency": avg_lat,
            "throughput": throughput
        }

    stats_naive = calculate_stats(naive)
    stats_opt = calculate_stats(optimized)

    # Percentage improvements (reduction for latency/TTFT/ITL, increase for throughput)
    ttft_imp = ((stats_naive['ttft'] - stats_opt['ttft']) / stats_naive['ttft']) * 100
    itl_imp = ((stats_naive['itl'] - stats_opt['itl']) / stats_naive['itl']) * 100
    lat_imp = ((stats_naive['latency'] - stats_opt['latency']) / stats_naive['latency']) * 100
    thru_imp = ((stats_opt['throughput'] - stats_naive['throughput']) / stats_naive['throughput']) * 100

    report = f"""
================================================================================
                    LLM INFERENCE SERVING ENGINE BENCHMARK REPORT
================================================================================

Comparative summary under concurrency:

| Metric                          | Naive Serving (FIFO) | Mini-vLLM Engine | Improvement % |
|---------------------------------|----------------------|------------------|---------------|
| Request Count                   | {stats_naive['count']:<20} | {stats_opt['count']:<16} | -             |
| Total Tokens Generated          | {stats_naive['tokens']:<20} | {stats_opt['tokens']:<16} | -             |
| Average TTFT (Time-To-First)    | {stats_naive['ttft']:.4f}s            | {stats_opt['ttft']:.4f}s         | {ttft_imp:.1f}% reduction   |
| Average ITL (Inter-Token)       | {stats_naive['itl']:.4f}s            | {stats_opt['itl']:.4f}s         | {itl_imp:.1f}% reduction   |
| Average Request Latency        | {stats_naive['latency']:.4f}s            | {stats_opt['latency']:.4f}s         | {lat_imp:.1f}% reduction   |
| System Throughput (Tokens/sec)  | {stats_naive['throughput']:.2f} tokens/s       | {stats_opt['throughput']:.2f} tokens/s   | {thru_imp:+.1f}% increase   |

--------------------------------------------------------------------------------
Key Observations:
1. **Continuous Batching (Iteration Scheduling)**: Reclaims pipeline bubbles by
   mixing prefill and decode tasks in parallel, boosting throughput significantly.
2. **Paged KV Cache (Prefix caching)**: Speeds up Time-to-First-Token (TTFT) for
   requests sharing common prompt prefixes, minimizing duplicate prompt compute.
3. **Speculative Decoding**: Employs a lightweight draft model to generate
   candidate tokens, verified in batch by the target model, lowering latency.
================================================================================
"""
    print(report)

    # Attempt to plot with matplotlib
    try:
        import matplotlib.pyplot as plt
        
        # 1. Bar chart comparison of Latency & TTFT
        metrics = ['Avg TTFT (s)', 'Avg ITL (s)', 'Avg Latency (s)']
        naive_vals = [stats_naive['ttft'], stats_naive['itl'], stats_naive['latency']]
        opt_vals = [stats_opt['ttft'], stats_opt['itl'], stats_opt['latency']]
        
        x = range(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Latency Subplot
        ax[0].bar([i - width/2 for i in x], naive_vals, width, label='Naive Serving (FIFO)', color='#f87171')
        ax[0].bar([i + width/2 for i in x], opt_vals, width, label='Mini-vLLM Engine', color='#60a5fa')
        ax[0].set_ylabel('Seconds (Lower is Better)')
        ax[0].set_title('Latency Metrics Comparison')
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(metrics)
        ax[0].legend()
        ax[0].grid(axis='y', linestyle='--', alpha=0.5)
        
        # Throughput Subplot
        ax[1].bar(['Naive Serving (FIFO)', 'Mini-vLLM Engine'], [stats_naive['throughput'], stats_opt['throughput']], 
                   width=0.5, color=['#f87171', '#60a5fa'])
        ax[1].set_ylabel('Tokens per Second (Higher is Better)')
        ax[1].set_title('System Generation Throughput')
        ax[1].grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.suptitle("Serving Engine Benchmarking Analysis", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Create plots folder if it doesn't exist
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/benchmark_comparison.png", dpi=150)
        print("Generated benchmark comparison chart: plots/benchmark_comparison.png")
    except ImportError:
        print("Matplotlib not installed. Skipping chart image generation (printed ASCII text report above).")

if __name__ == "__main__":
    main()
