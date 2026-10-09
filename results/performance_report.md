# Load Testing & Performance Benchmark Report

- **Target Endpoint:** `http://127.0.0.1:8000/predict`
- **Execution Timestamp:** `2026-10-09 04:54:26 UTC`
- **Test Concurrency:** `3` parallel workers
- **Total Requests Executed:** `15`
- **Total Test Duration:** `5.36 seconds`

---

## Summary Metrics

| Metric | Value |
| :--- | :--- |
| **Successful Requests** | `15` |
| **Failed Requests** | `0` |
| **Error Rate** | `0.00%` |
| **Throughput (RPS)** | `2.80 req/sec` |
| **Min Latency** | `330.93 ms` |
| **Average Latency** | `1014.57 ms` |
| **p50 Latency (Median)** | `975.04 ms` |
| **p95 Latency** | `1464.12 ms` |
| **p99 Latency** | `1513.50 ms` |
| **Max Latency** | `1525.85 ms` |

---

## Test Conditions & Environment
- **Backbone Framework:** TensorFlow / Keras (Xception Deep Transfer Learning)
- **Input Dimension:** 299 x 299 x 3 RGB PNG Image payload
- **Validation Scope:** Warm-up request, concurrent load, error handling, latency distribution
