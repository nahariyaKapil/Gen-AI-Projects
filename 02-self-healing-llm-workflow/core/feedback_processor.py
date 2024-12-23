"""
FeedbackProcessor: Processes feedback and generates learning signals
"""

import logging
import time
from typing import Dict, List, Any, Optional

class FeedbackProcessor:
    """
    Processes feedback from workflow executions for continuous learning
    
    Features:
    - Performance analysis
    - Quality scoring
    - Learning signal generation
    - Feedback aggregation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feedback_history: List[Dict[str, Any]] = []
        
    async def generate_feedback(self, execution) -> List[float]:
        """Generate feedback scores for workflow execution"""
        try:
            feedback_scores = []
            
            # Performance feedback
            performance_score = self._calculate_performance_score(execution)
            feedback_scores.append(performance_score)
            
            # Quality feedback
            quality_score = self._calculate_quality_score(execution)
            feedback_scores.append(quality_score)
            
            # Efficiency feedback
            efficiency_score = self._calculate_efficiency_score(execution)
            feedback_scores.append(efficiency_score)
            
            # Reliability feedback
            reliability_score = self._calculate_reliability_score(execution)
            feedback_scores.append(reliability_score)
            
            # Store feedback
            feedback_record = {
                'execution_id': execution.id,
                'scores': feedback_scores,
                'timestamp': time.time(),
                'metrics': {
                    'performance': performance_score,
                    'quality': quality_score,
                    'efficiency': efficiency_score,
                    'reliability': reliability_score
                }
            }
            
            self.feedback_history.append(feedback_record)
            
            self.logger.debug(f"Generated feedback for execution {execution.id}: {feedback_scores}")
            return feedback_scores
            
        except Exception as e:
            self.logger.error(f"Failed to generate feedback: {str(e)}")
            return [0.5, 0.5, 0.5, 0.5]  # Default neutral scores

    def _calculate_performance_score(self, execution) -> float:
        """Calculate performance score based on success rate and execution time"""
        success_weight = 0.7
        time_weight = 0.3
        
        # Success rate component
        success_score = execution.success_rate
        
        # Time efficiency component (normalized)
        expected_time = len(execution.tasks) * 10.0  # Assume 10 seconds per task baseline
        time_score = min(1.0, expected_time / max(execution.total_execution_time, 0.1))
        
        return success_score * success_weight + time_score * time_weight

    def _calculate_quality_score(self, execution) -> float:
        """Calculate quality score based on task outputs and error rates"""
        if not execution.tasks:
            return 0.5
        
        # Quality indicators
        completed_tasks = sum(1 for task in execution.tasks 
                            if task.status.value in ['completed', 'recovered'])
        error_rate = len(execution.healing_events) / len(execution.tasks)
        
        # Quality score calculation
        completion_rate = completed_tasks / len(execution.tasks)
        error_penalty = max(0, 1.0 - error_rate)
        
        return (completion_rate + error_penalty) / 2.0

    def _calculate_efficiency_score(self, execution) -> float:
        """Calculate efficiency score based on resource utilization"""
        if not execution.tasks:
            return 0.5
        
        # Simple efficiency calculation
        avg_task_time = execution.total_execution_time / len(execution.tasks)
        expected_task_time = 5.0  # Baseline expectation
        
        efficiency = min(1.0, expected_task_time / max(avg_task_time, 0.1))
        
        # Penalize excessive healing events
        healing_penalty = max(0, 1.0 - len(execution.healing_events) * 0.1)
        
        return efficiency * healing_penalty

    def _calculate_reliability_score(self, execution) -> float:
        """Calculate reliability score based on consistency and error recovery"""
        if not execution.tasks:
            return 0.5
        
        # Reliability factors
        task_success_rate = execution.success_rate
        
        # Recovery effectiveness
        if execution.healing_events:
            recovered_tasks = sum(1 for task in execution.tasks 
                                if task.status.value == 'recovered')
            recovery_rate = recovered_tasks / len(execution.healing_events)
        else:
            recovery_rate = 1.0  # No failures to recover from
        
        return (task_success_rate + recovery_rate) / 2.0

    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get feedback system statistics"""
        if not self.feedback_history:
            return {
                'total_feedback_records': 0,
                'average_scores': {'performance': 0.5, 'quality': 0.5, 'efficiency': 0.5, 'reliability': 0.5}
            }
        
        # Calculate averages
        avg_performance = sum(f['metrics']['performance'] for f in self.feedback_history) / len(self.feedback_history)
        avg_quality = sum(f['metrics']['quality'] for f in self.feedback_history) / len(self.feedback_history)
        avg_efficiency = sum(f['metrics']['efficiency'] for f in self.feedback_history) / len(self.feedback_history)
        avg_reliability = sum(f['metrics']['reliability'] for f in self.feedback_history) / len(self.feedback_history)
        
        return {
            'total_feedback_records': len(self.feedback_history),
            'average_scores': {
                'performance': avg_performance,
                'quality': avg_quality,
                'efficiency': avg_efficiency,
                'reliability': avg_reliability
            }
        } 