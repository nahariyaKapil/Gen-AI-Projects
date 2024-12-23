"""
TaskRouter: Intelligent task routing to optimal agents
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from enum import Enum

class RoutingStrategy(Enum):
    PERFORMANCE_BASED = "performance_based"
    LOAD_BALANCED = "load_balanced"
    SPECIALIZED = "specialized"

class TaskRouter:
    """
    Intelligent task router that matches tasks with optimal agents
    
    Features:
    - Performance-based routing
    - Load balancing
    - Agent specialization awareness
    - Dynamic routing optimization
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.routing_history: Dict[str, Any] = {}
        self.agent_workloads: Dict[str, int] = {}
        
    async def route_task(self, task):
        """Route task to the best available agent"""
        try:
            # Import here to avoid circular imports
            from .agent_manager import AgentType, AgentManager
            
            # Determine agent type based on task
            agent_type = self._determine_agent_type(task)
            
            # Get agent manager (in production, this would be injected)
            agent_manager = AgentManager()
            
            # Get best agent for this task type
            agent = await agent_manager.get_best_agent(agent_type)
            
            # Track routing decision
            self.routing_history[task.id] = {
                'agent_id': agent.agent_id,
                'agent_type': agent_type.value,
                'routing_strategy': 'performance_based'
            }
            
            self.logger.debug(f"Routed task {task.name} to agent {agent.agent_id}")
            return agent
            
        except Exception as e:
            self.logger.error(f"Failed to route task {task.name}: {str(e)}")
            raise

    def _determine_agent_type(self, task):
        """Determine appropriate agent type for task"""
        from .agent_manager import AgentType
        
        task_type = task.agent_type.lower()
        
        # Map task types to agent types
        if 'llm' in task_type or 'reasoning' in task_type:
            return AgentType.LLM_REASONING
        elif 'code' in task_type:
            return AgentType.CODE_GENERATOR
        elif 'data' in task_type or 'process' in task_type:
            return AgentType.DATA_PROCESSOR
        elif 'quality' in task_type or 'checker' in task_type:
            return AgentType.QUALITY_CHECKER
        else:
            # Default to LLM reasoning
            return AgentType.LLM_REASONING

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics"""
        return {
            'total_routes': len(self.routing_history),
            'agent_distribution': {},  # Would calculate from history
            'average_response_time': 0.0  # Would calculate from metrics
        } 