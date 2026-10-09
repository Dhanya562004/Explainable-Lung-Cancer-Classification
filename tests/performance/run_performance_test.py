import os
import time
import argparse
import concurrent.futures
import numpy as np
import httpx
from PIL import Image
import io

def generate_sample_image_bytes():
    img = Image.new('RGB', (299, 299), color='gray')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

def send_prediction_request(client, base_url, image_bytes):
    start = time.perf_counter()
    try:
        response = client.post(
            f"{base_url}/predict",
            files={"file": ("sample_ct.png", image_bytes, "image/png")},
            timeout=10.0
        )
        latency = (time.perf_counter() - start) * 1000.0
        return response.status_code == 200, latency, response.status_code
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000.0
        return False, latency, 500

def run_performance_suite(base_url: str, concurrency: int = 3, total_requests: int = 15):
    print(f"[INFO] Starting Performance Load Test against: {base_url}")
    print(f"[INFO] Concurrency: {concurrency} workers | Total Requests: {total_requests}")

    image_bytes = generate_sample_image_bytes()
    client = httpx.Client()

    # 1. Warm-up Request
    print("[INFO] Executing warm-up request...")
    success, lat, status_code = send_prediction_request(client, base_url, image_bytes)
    print(f"[INFO] Warm-up finished: status={status_code}, latency={lat:.1f}ms")

    # 2. Concurrent Load Test Execution
    latencies = []
    success_count = 0
    failure_count = 0
    start_total_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_prediction_request, client, base_url, image_bytes)
            for _ in range(total_requests)
        ]
        for f in concurrent.futures.as_completed(futures):
            is_ok, latency, st_code = f.result()
            latencies.append(latency)
            if is_ok:
                success_count += 1
            else:
                failure_count += 1

    total_duration_sec = time.perf_counter() - start_total_time
    rps = total_requests / total_duration_sec if total_duration_sec > 0 else 0

    p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    p99 = float(np.percentile(latencies, 99)) if latencies else 0.0
    avg_lat = float(np.mean(latencies)) if latencies else 0.0
    min_lat = float(np.min(latencies)) if latencies else 0.0
    max_lat = float(np.max(latencies)) if latencies else 0.0

    report_md = f"""# Load Testing & Performance Benchmark Report

- **Target Endpoint:** `{base_url}/predict`
- **Execution Timestamp:** `{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}`
- **Test Concurrency:** `{concurrency}` parallel workers
- **Total Requests Executed:** `{total_requests}`
- **Total Test Duration:** `{total_duration_sec:.2f} seconds`

---

## Summary Metrics

| Metric | Value |
| :--- | :--- |
| **Successful Requests** | `{success_count}` |
| **Failed Requests** | `{failure_count}` |
| **Error Rate** | `{(failure_count / total_requests) * 100:.2f}%` |
| **Throughput (RPS)** | `{rps:.2f} req/sec` |
| **Min Latency** | `{min_lat:.2f} ms` |
| **Average Latency** | `{avg_lat:.2f} ms` |
| **p50 Latency (Median)** | `{p50:.2f} ms` |
| **p95 Latency** | `{p95:.2f} ms` |
| **p99 Latency** | `{p99:.2f} ms` |
| **Max Latency** | `{max_lat:.2f} ms` |

---

## Test Conditions & Environment
- **Backbone Framework:** TensorFlow / Keras (Xception Deep Transfer Learning)
- **Input Dimension:** 299 x 299 x 3 RGB PNG Image payload
- **Validation Scope:** Warm-up request, concurrent load, error handling, latency distribution
"""

    os.makedirs("results", exist_ok=True)
    report_path = os.path.join("results", "performance_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print("[SUCCESS] Load Test Completed Successfully!")
    print(f"[RESULTS] RPS={rps:.2f} | p50={p50:.1f}ms | p95={p95:.1f}ms | p99={p99:.1f}ms")
    print(f"[INFO] Full Markdown Report saved to: {report_path}")
    return report_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run performance load tests on Explainable Lung Cancer API")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Base API URL")
    parser.add_argument("--concurrency", type=int, default=3, help="Number of parallel request threads")
    parser.add_argument("--requests", type=int, default=15, help="Total number of prediction requests")
    args = parser.parse_args()

    run_performance_suite(args.url, args.concurrency, args.requests)
