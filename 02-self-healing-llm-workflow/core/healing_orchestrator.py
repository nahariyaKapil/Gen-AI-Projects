"""
HealingOrchestrator: Manages self-healing and recovery strategies
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass

class RecoveryStrategy(Enum):
    RETRY = "retry"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    AGENT_SWITCH = "agent_switch"
    TASK_DECOMPOSITION = "task_decomposition"
    CONTEXT_ENHANCEMENT = "context_enhancement"

@dataclass
class RecoveryPlan:
    strategy: RecoveryStrategy
    parameters: Dict[str, Any]
    confidence: float
    estimated_success_rate: float

class HealingOrchestrator:
    """
    Orchestrates self-healing and recovery for failed workflow tasks
    
    Features:
    - Failure pattern analysis
    - Recovery strategy selection
    - Adaptive healing based on historical data
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.recovery_history: List[Dict[str, Any]] = []
        self.strategy_success_rates = {
            strategy: 0.5 for strategy in RecoveryStrategy
        }
        
    async def analyze_failure(self, task, healing_events: List[Dict[str, Any]]) -> RecoveryPlan:
        """Analyze task failure and determine recovery strategy"""
        try:
            error_info = task.error_info
            error_type = error_info.get('error_type', 'Unknown')
            retry_count = task.retry_count
            
            # Analyze error patterns
            strategy = await self._select_recovery_strategy(error_type, retry_count, healing_events)
            
            # Generate recovery parameters
            parameters = await self._generate_recovery_parameters(task, strategy)
            
            # Estimate success probability
            confidence = self._calculate_confidence(strategy, task)
            success_rate = self.strategy_success_rates.get(strategy, 0.5)
            
            recovery_plan = RecoveryPlan(
                strategy=strategy,
                parameters=parameters,
                confidence=confidence,
                estimated_success_rate=success_rate
            )
            
            self.logger.info(f"Generated recovery plan for task {task.name}: {strategy.value}")
            return recovery_plan
            
        except Exception as e:
            self.logger.error(f"Failed to analyze failure: {str(e)}")
            # Default recovery plan
            return RecoveryPlan(
                strategy=RecoveryStrategy.RETRY,
                parameters={'delay': 1.0},
                confidence=0.3,
                estimated_success_rate=0.3
            )

    async def apply_recovery(self, task, recovery_plan: RecoveryPlan):
        """Apply recovery strategy to failed task"""
        try:
            strategy = recovery_plan.strategy
            parameters = recovery_plan.parameters
            
            self.logger.info(f"Applying recovery strategy {strategy.value} to task {task.name}")
            
            if strategy == RecoveryStrategy.RETRY:
                await self._apply_retry_recovery(task, parameters)
            
            elif strategy == RecoveryStrategy.PARAMETER_ADJUSTMENT:
                await self._apply_parameter_adjustment(task, parameters)
            
            elif strategy == RecoveryStrategy.AGENT_SWITCH:
                await self._apply_agent_switch(task, parameters)
            
            elif strategy == RecoveryStrategy.TASK_DECOMPOSITION:
                await self._apply_task_decomposition(task, parameters)
            
            elif strategy == RecoveryStrategy.CONTEXT_ENHANCEMENT:
                await self._apply_context_enhancement(task, parameters)
            
            # Record recovery attempt
            self.recovery_history.append({
                'task_id': task.id,
                'strategy': strategy.value,
                'timestamp': time.time(),
                'parameters': parameters
            })
            
            return task
            
        except Exception as e:
            self.logger.error(f"Failed to apply recovery: {str(e)}")
            raise

    async def _select_recovery_strategy(self, error_type: str, retry_count: int, healing_events: List) -> RecoveryStrategy:
        """Select appropriate recovery strategy based on error analysis"""
        
        # Simple strategy selection based on error type and retry count
        if retry_count == 1:
            if 'timeout' in error_type.lower():
                return RecoveryStrategy.PARAMETER_ADJUSTMENT
            elif 'auth' in error_type.lower() or 'permission' in error_type.lower():
                return RecoveryStrategy.AGENT_SWITCH
            else:
                return RecoveryStrategy.RETRY
        
        elif retry_count == 2:
            if 'complex' in error_type.lower() or 'large' in error_type.lower():
                return RecoveryStrategy.TASK_DECOMPOSITION
            else:
                return RecoveryStrategy.CONTEXT_ENHANCEMENT
        
        else:
            # Last resort - try agent switch
            return RecoveryStrategy.AGENT_SWITCH

    async def _generate_recovery_parameters(self, task, strategy: RecoveryStrategy) -> Dict[str, Any]:
        """Generate parameters for recovery strategy"""
        
        if strategy == RecoveryStrategy.RETRY:
            return {
                'delay': min(2.0 ** task.retry_count, 30.0),  # Exponential backoff
                'timeout_multiplier': 1.5
            }
        
        elif strategy == RecoveryStrategy.PARAMETER_ADJUSTMENT:
            return {
                'timeout_increase': 2.0,
                'batch_size_reduction': 0.5,
                'temperature_adjustment': -0.1
            }
        
        elif strategy == RecoveryStrategy.AGENT_SWITCH:
            return {
                'exclude_agents': [task.agent_type],
                'prefer_reliable': True
            }
        
        elif strategy == RecoveryStrategy.TASK_DECOMPOSITION:
            return {
                'max_subtasks': 3,
                'overlap_threshold': 0.2
            }
        
        elif strategy == RecoveryStrategy.CONTEXT_ENHANCEMENT:
            return {
                'add_examples': True,
                'simplify_language': True,
                'add_constraints': True
            }
        
        return {}

    def _calculate_confidence(self, strategy: RecoveryStrategy, task) -> float:
        """Calculate confidence in recovery strategy"""
        base_confidence = self.strategy_success_rates.get(strategy, 0.5)
        
        # Adjust based on task characteristics
        if task.retry_count == 1:
            base_confidence *= 0.9
        elif task.retry_count == 2:
            base_confidence *= 0.7
        else:
            base_confidence *= 0.5
        
        return min(max(base_confidence, 0.1), 0.95)

    async def _apply_retry_recovery(self, task, parameters: Dict[str, Any]):
        """Apply retry recovery with delay"""
        delay = parameters.get('delay', 1.0)
        await asyncio.sleep(delay)
        
        # Clear previous error info
        task.error_info = None
        task.status = None  # Reset for retry

    async def _apply_parameter_adjustment(self, task, parameters: Dict[str, Any]):
        """Apply parameter adjustments to task"""
        # Adjust input parameters
        if 'timeout_increase' in parameters:
            if 'timeout' in task.input_data:
                task.input_data['timeout'] *= parameters['timeout_increase']
        
        if 'temperature_adjustment' in parameters:
            if 'temperature' in task.input_data:
                task.input_data['temperature'] += parameters['temperature_adjustment']
                task.input_data['temperature'] = max(0.0, min(1.0, task.input_data['temperature']))

    async def _apply_agent_switch(self, task, parameters: Dict[str, Any]):
        """Switch to different agent type"""
        excluded = parameters.get('exclude_agents', [])
        
        # Simple agent switching logic
        agent_alternatives = {
            'llm_reasoning': 'code_generator',
            'code_generator': 'data_processor',
            'data_processor': 'llm_reasoning'
        }
        
        if task.agent_type in agent_alternatives:
            new_agent_type = agent_alternatives[task.agent_type]
            if new_agent_type not in excluded:
                task.agent_type = new_agent_type
                self.logger.info(f"Switched task {task.name} to agent type: {new_agent_type}")

    async def _apply_task_decomposition(self, task, parameters: Dict[str, Any]):
        """Decompose complex task into simpler subtasks"""
        # For demo purposes, simplify the task input
        if 'prompt' in task.input_data:
            original_prompt = task.input_data['prompt']
            simplified_prompt = f"Please provide a step-by-step solution for: {original_prompt[:200]}..."
            task.input_data['prompt'] = simplified_prompt
            task.input_data['decomposed'] = True

    async def _apply_context_enhancement(self, task, parameters: Dict[str, Any]):
        """Enhance task context for better success"""
        if parameters.get('add_examples'):
            task.input_data['include_examples'] = True
        
        if parameters.get('simplify_language'):
            task.input_data['use_simple_language'] = True
        
        if parameters.get('add_constraints'):
            task.input_data['add_constraints'] = True

    def update_strategy_performance(self, strategy: RecoveryStrategy, success: bool):
        """Update success rate for recovery strategy"""
        current_rate = self.strategy_success_rates[strategy]
        
        # Simple exponential moving average
        if success:
            self.strategy_success_rates[strategy] = current_rate * 0.9 + 0.1
        else:
            self.strategy_success_rates[strategy] = current_rate * 0.9

    def get_healing_statistics(self) -> Dict[str, Any]:
        """Get healing system statistics"""
        return {
            'total_recoveries': len(self.recovery_history),
            'strategy_success_rates': {s.value: rate for s, rate in self.strategy_success_rates.items()},
            'recent_recoveries': len([r for r in self.recovery_history 
                                    if time.time() - r['timestamp'] < 3600])  # Last hour
        } 