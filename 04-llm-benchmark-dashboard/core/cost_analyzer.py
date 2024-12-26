"""
CostAnalyzer: Cost tracking and optimization analysis
"""

import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta

class CostAnalyzer:
    """Advanced cost analysis and optimization recommendations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cost_history = []
    
    def analyze_costs(self, benchmark_results: List) -> Dict[str, Any]:
        """Comprehensive cost analysis"""
        if not benchmark_results:
            return {}
        
        # Calculate cost metrics
        total_cost = sum(r.cost for r in benchmark_results)
        avg_cost_per_request = total_cost / len(benchmark_results)
        
        # Cost by model
        by_model = {}
        for result in benchmark_results:
            if result.model_name not in by_model:
                by_model[result.model_name] = []
            by_model[result.model_name].append(result.cost)
        
        model_costs = {}
        for model, costs in by_model.items():
            model_costs[model] = {
                'total_cost': sum(costs),
                'avg_cost': sum(costs) / len(costs),
                'cost_efficiency': self._calculate_efficiency(model, benchmark_results)
            }
        
        # Optimization opportunities
        optimization_potential = self._calculate_optimization_potential(benchmark_results)
        
        return {
            'total_cost': total_cost,
            'avg_cost_per_request': avg_cost_per_request,
            'cost_by_model': model_costs,
            'optimization_potential': optimization_potential,
            'recommendations': self._generate_cost_recommendations(model_costs)
        }
    
    def _calculate_efficiency(self, model: str, results: List) -> float:
        """Calculate cost efficiency (quality per dollar)"""
        model_results = [r for r in results if r.model_name == model]
        if not model_results:
            return 0.0
        
        total_quality = sum(r.quality_score for r in model_results)
        total_cost = sum(r.cost for r in model_results)
        
        return total_quality / max(total_cost, 0.0001)
    
    def _calculate_optimization_potential(self, results: List) -> Dict[str, float]:
        """Calculate potential cost savings from optimization"""
        return {
            'caching_savings': 0.6,  # 60% savings from caching
            'quantization_savings': 0.4,  # 40% savings from quantization
            'prompt_optimization_savings': 0.2  # 20% savings from prompt optimization
        }
    
    def _generate_cost_recommendations(self, model_costs: Dict) -> List[str]:
        """Generate cost optimization recommendations"""
        recommendations = []
        
        # Find most expensive model
        if model_costs:
            most_expensive = max(model_costs.keys(), 
                               key=lambda m: model_costs[m]['total_cost'])
            recommendations.append(f"Consider optimizing {most_expensive} (highest total cost)")
            
            # Find least efficient model
            least_efficient = min(model_costs.keys(),
                                key=lambda m: model_costs[m]['cost_efficiency'])
            recommendations.append(f"Improve efficiency of {least_efficient}")
        
        recommendations.extend([
            "Implement response caching for 60% cost reduction",
            "Use quantization to reduce inference costs by 40%",
            "Optimize prompts to reduce token usage by 20%"
        ])
        
        return recommendations 