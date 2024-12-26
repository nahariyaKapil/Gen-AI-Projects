"""
PerformanceMonitor: Real-time performance tracking and alerting
"""

import logging
import time
from typing import Dict, List, Any
from collections import defaultdict, deque

class PerformanceMonitor:
    """Real-time performance monitoring with alerting capabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_buffer = deque(maxlen=1000)
        self.alert_thresholds = {
            'response_time': 10.0,
            'error_rate': 0.1,
            'cost_per_request': 0.05
        }
        self.active_alerts = []
    
    def record_metric(self, metric_name: str, value: float, model: str):
        """Record a performance metric"""
        self.metrics_buffer.append({
            'metric': metric_name,
            'value': value,
            'model': model,
            'timestamp': time.time()
        })
        
        # Check for alerts
        self._check_alerts(metric_name, value, model)
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics"""
        recent_metrics = [m for m in self.metrics_buffer 
                         if time.time() - m['timestamp'] < 300]  # Last 5 minutes
        
        if not recent_metrics:
            return {}
        
        # Aggregate metrics
        by_model = defaultdict(list)
        for metric in recent_metrics:
            by_model[metric['model']].append(metric)
        
        aggregated = {}
        for model, metrics in by_model.items():
            response_times = [m['value'] for m in metrics if m['metric'] == 'response_time']
            
            aggregated[model] = {
                'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
                'request_count': len(metrics),
                'alerts': [a for a in self.active_alerts if a.get('model') == model]
            }
        
        return aggregated
    
    def _check_alerts(self, metric_name: str, value: float, model: str):
        """Check if metric exceeds alert thresholds"""
        if metric_name in self.alert_thresholds:
            threshold = self.alert_thresholds[metric_name]
            
            if value > threshold:
                alert = {
                    'type': f'high_{metric_name}',
                    'model': model,
                    'value': value,
                    'threshold': threshold,
                    'timestamp': time.time()
                }
                self.active_alerts.append(alert)
                self.logger.warning(f"Alert: {model} {metric_name} = {value} exceeds threshold {threshold}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        recent_alerts = [a for a in self.active_alerts 
                        if time.time() - a['timestamp'] < 3600]  # Last hour
        
        health_score = 1.0 - min(len(recent_alerts) * 0.1, 0.5)
        
        return {
            'health_score': health_score,
            'status': 'healthy' if health_score > 0.8 else 'warning' if health_score > 0.5 else 'critical',
            'active_alerts': len(recent_alerts),
            'total_requests': len(self.metrics_buffer)
        } 