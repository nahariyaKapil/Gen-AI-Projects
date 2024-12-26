# Simplified metrics collector
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self):
        self.metrics = {}
        
    def collect_metric(self, name, value):
        """Collect a metric"""
        self.metrics[name] = {
            "value": value,
            "timestamp": time.time()
        }
        logger.info(f"Collected metric {name}: {value}")
        
    def get_metrics(self):
        """Get all collected metrics"""
        return self.metrics
        
    def get_metric(self, name):
        """Get a specific metric"""
        return self.metrics.get(name, {"value": 0, "timestamp": time.time()})
