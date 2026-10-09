import time
import threading
from typing import Dict, Any

class PrometheusMetricsManager:
    """
    Thread-safe Prometheus metrics collector and exporter for API operational monitoring.
    Maintains bounded metric labels to prevent cardinality explosion.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.request_count = {"success": 0, "failure": 0}
        self.predictions_by_confidence = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        self.total_latency_seconds = 0.0
        self.latency_samples = []

    def record_prediction(self, status: str, latency_ms: float, confidence_level: str = None):
        with self._lock:
            if status in self.request_count:
                self.request_count[status] += 1
            else:
                self.request_count["failure"] += 1

            if confidence_level and confidence_level in self.predictions_by_confidence:
                self.predictions_by_confidence[confidence_level] += 1

            latency_sec = latency_ms / 1000.0
            self.total_latency_seconds += latency_sec
            self.latency_samples.append(latency_sec)

            # Keep latency samples array bounded to last 5000 requests
            if len(self.latency_samples) > 5000:
                self.latency_samples = self.latency_samples[-5000:]

    def generate_prometheus_format(self) -> str:
        with self._lock:
            total_reqs = sum(self.request_count.values())
            success_reqs = self.request_count["success"]
            fail_reqs = self.request_count["failure"]
            high_conf = self.predictions_by_confidence["HIGH"]
            med_conf = self.predictions_by_confidence["MEDIUM"]
            low_conf = self.predictions_by_confidence["LOW"]
            total_lat = self.total_latency_seconds

        lines = [
            "# HELP api_predictions_total Total count of prediction requests processed.",
            "# TYPE api_predictions_total counter",
            f'api_predictions_total{{status="success"}} {success_reqs}',
            f'api_predictions_total{{status="failure"}} {fail_reqs}',
            "",
            "# HELP api_predictions_by_confidence_total Count of predictions by confidence level.",
            "# TYPE api_predictions_by_confidence_total counter",
            f'api_predictions_by_confidence_total{{level="HIGH"}} {high_conf}',
            f'api_predictions_by_confidence_total{{level="MEDIUM"}} {med_conf}',
            f'api_predictions_by_confidence_total{{level="LOW"}} {low_conf}',
            "",
            "# HELP api_prediction_latency_seconds_total Total latency spent processing predictions in seconds.",
            "# TYPE api_prediction_latency_seconds_total counter",
            f'api_prediction_latency_seconds_total {total_lat:.4f}',
            "",
            "# HELP api_requests_total Overall HTTP requests processed.",
            "# TYPE api_requests_total counter",
            f'api_requests_total {total_reqs}'
        ]
        return "\n".join(lines) + "\n"

# Global singleton instance
metrics_manager = PrometheusMetricsManager()
