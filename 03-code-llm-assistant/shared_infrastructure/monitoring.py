# Simplified monitoring module
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonitoringSystem:
    def track_metric(self, name, value):
        logger.info(f"Metric {name}: {value}")
        
    def log_event(self, event):
        logger.info(f"Event: {event}")
