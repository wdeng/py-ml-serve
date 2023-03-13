from prometheus_client import Histogram, Counter


class CustomMetrics:
    def __init__(self):
        self.latency_histogram = Histogram(
            "endpoint_latency_seconds",
            "Latency for endpoints",
            ["method", "endpoint"]
        )
        self.click_counter = Counter(
            "endpoint_click_count",
            "Endpoint count",
            ["method", "endpoint"]
        )
        self.status_counter = Counter(
            "endpoint_status_count",
            "Endpoint status count",
            ["method", "endpoint", "status_code"]
        )

    def label(self, req, resp, latency_time):
        self.latency_histogram.labels(method=req.method, endpoint=req.path).observe(latency_time)
        self.click_counter.labels(method=req.method, endpoint=req.path).inc()
        self.status_counter.labels(method=req.method, endpoint=req.path, status_code=resp.status_code).inc()
