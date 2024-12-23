"""
WorkflowEngine: Core orchestrator for self-healing LLM workflows

This engine manages the execution of complex LLM workflows with built-in
error recovery, adaptive learning, and multi-agent coordination.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from uuid import uuid4
import traceback
import os

from .agent_manager import AgentManager
from .memory_system import MemorySystem
from .healing_orchestrator import HealingOrchestrator
from .task_router import TaskRouter
from .feedback_processor import FeedbackProcessor

# Production-level configuration  
class ProductionConfig:
    def __init__(self):
        self._config = self._load_config()
    
    def _load_config(self):
        """Load configuration from multiple sources for production"""
        config = {
            'openai_api_key': os.getenv('OPENAI_API_KEY', ''),
            'max_workers': int(os.getenv('MAX_WORKERS', '8')),
            'timeout': int(os.getenv('TIMEOUT', '60')),
            'retry_attempts': int(os.getenv('RETRY_ATTEMPTS', '5')),
            'max_retries': int(os.getenv('MAX_RETRIES', '5')),
            'backoff_factor': float(os.getenv('BACKOFF_FACTOR', '2.0')),
            'batch_size': int(os.getenv('BATCH_SIZE', '100')),
            'enable_advanced_memory': os.getenv('ENABLE_ADVANCED_MEMORY', 'true').lower() == 'true',
            'enable_self_healing': os.getenv('ENABLE_SELF_HEALING', 'true').lower() == 'true',
            'enable_performance_monitoring': os.getenv('ENABLE_PERFORMANCE_MONITORING', 'true').lower() == 'true',
            'enable_auto_scaling': os.getenv('ENABLE_AUTO_SCALING', 'true').lower() == 'true'
        }
        
        # Try to load from Streamlit secrets if available
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                config.update({
                    'openai_api_key': st.secrets.get('OPENAI_API_KEY', config['openai_api_key']),
                    'max_workers': st.secrets.get('MAX_WORKERS', config['max_workers']),
                    'timeout': st.secrets.get('TIMEOUT', config['timeout']),
                    'retry_attempts': st.secrets.get('RETRY_ATTEMPTS', config['retry_attempts']),
                    'batch_size': st.secrets.get('BATCH_SIZE', config['batch_size']),
                    'enable_advanced_memory': st.secrets.get('ENABLE_ADVANCED_MEMORY', config['enable_advanced_memory']),
                    'enable_self_healing': st.secrets.get('ENABLE_SELF_HEALING', config['enable_self_healing']),
                    'enable_performance_monitoring': st.secrets.get('ENABLE_PERFORMANCE_MONITORING', config['enable_performance_monitoring']),
                    'enable_auto_scaling': st.secrets.get('ENABLE_AUTO_SCALING', config['enable_auto_scaling'])
                })
        except:
            # Streamlit not available or secrets not configured
            pass
        
        return config
    
    def get(self, key, default=None):
        return self._config.get(key, default)
    
    def is_production_mode(self):
        """Check if running in production mode"""
        api_key = self.get('openai_api_key')
        return api_key and api_key != 'demo_mode' and api_key.startswith('sk-')
    
    def get_feature_flags(self):
        """Get feature flags for production features"""
        return {
            'advanced_memory': self.get('enable_advanced_memory', True),
            'self_healing': self.get('enable_self_healing', True),
            'performance_monitoring': self.get('enable_performance_monitoring', True),
            'auto_scaling': self.get('enable_auto_scaling', True)
        }

class ExpertMetricsCollector:
    def __init__(self):
        self.metrics = {}
        self.counters = {}
        self.request_times = {}
        
    def track_request(self, service, operation):
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    self.record_metric(f"{service}_{operation}_success", 1, {"service": service, "operation": operation})
                    self.record_metric(f"{service}_{operation}_duration", execution_time, {"service": service, "operation": operation})
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    self.record_metric(f"{service}_{operation}_error", 1, {"service": service, "operation": operation})
                    self.record_metric(f"{service}_{operation}_duration", execution_time, {"service": service, "operation": operation})
                    raise
            return wrapper
        return decorator
    
    def record_metric(self, name, value, tags=None):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({
            'value': value,
            'timestamp': time.time(),
            'tags': tags or {}
        })
        
    def increment_counter(self, name, value=1, tags=None):
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += value
        
    def get_metrics(self):
        return {
            'metrics': self.metrics,
            'counters': self.counters,
            'request_times': self.request_times
        }
    
    def record_gauge(self, name, value, tags=None):
        """Record gauge metric"""
        self.record_metric(name, value, tags)
    
    def time_operation(self, operation_name):
        """Context manager for timing operations"""
        import time
        class Timer:
            def __init__(self, collector, name):
                self.collector = collector
                self.name = name
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.collector.record_metric(f"{self.name}_duration", duration)
        
        return Timer(self, operation_name)

config_instance = ProductionConfig()
metrics_collector = ExpertMetricsCollector()

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    HEALING = "healing"
    RECOVERED = "recovered"

@dataclass
class WorkflowTask:
    """Represents a task in the workflow"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    agent_type: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.PENDING
    error_info: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    max_retries: int = 3
    execution_time: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

@dataclass 
class WorkflowExecution:
    """Tracks the execution of an entire workflow"""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    tasks: List[WorkflowTask] = field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: float = 0.0
    end_time: float = 0.0
    total_execution_time: float = 0.0
    success_rate: float = 0.0
    healing_events: List[Dict[str, Any]] = field(default_factory=list)
    feedback_scores: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class WorkflowEngine:
    """
    Advanced workflow engine with self-healing capabilities
    
    Features:
    - Multi-agent task execution
    - Automatic error recovery
    - Adaptive learning from failures
    - Real-time monitoring and metrics
    - Intelligent task routing
    - Memory-based optimization
    """
    
    def __init__(self):
        self.config = config_instance
        self.logger = logging.getLogger(__name__)
        self.metrics = metrics_collector
        
        # Initialize core components
        self.agent_manager = AgentManager()
        self.memory_system = MemorySystem()
        self.healing_orchestrator = HealingOrchestrator()
        self.task_router = TaskRouter()
        self.feedback_processor = FeedbackProcessor()
        
        # Active executions
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.execution_history: List[WorkflowExecution] = []
        
        # Performance tracking
        self.total_executions = 0
        self.successful_executions = 0
        self.healing_success_rate = 0.0
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            'task_started': [],
            'task_completed': [],
            'task_failed': [],
            'healing_triggered': [],
            'execution_completed': []
        }
        
        self.logger.info("WorkflowEngine initialized successfully")

    async def create_workflow(self, 
                            name: str, 
                            tasks: List[Dict[str, Any]], 
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new workflow execution
        
        Args:
            name: Workflow name
            tasks: List of task definitions
            metadata: Additional workflow metadata
            
        Returns:
            workflow_id: Unique identifier for the workflow execution
        """
        try:
            workflow_execution = WorkflowExecution(
                name=name,
                metadata=metadata or {}
            )
            
            # Create workflow tasks (first pass - create tasks without resolving dependencies)
            task_name_to_id = {}
            for task_def in tasks:
                task = WorkflowTask(
                    name=task_def['name'],
                    agent_type=task_def['agent_type'],
                    input_data=task_def.get('input_data', {}),
                    dependencies=[],  # Will be resolved in second pass
                    max_retries=task_def.get('max_retries', 3)
                )
                workflow_execution.tasks.append(task)
                task_name_to_id[task.name] = task.id
            
            # Second pass - resolve dependencies from task names to task IDs
            for i, task_def in enumerate(tasks):
                task = workflow_execution.tasks[i]
                dep_names = task_def.get('dependencies', [])
                
                # Convert dependency names to task IDs
                resolved_deps = []
                for dep_name in dep_names:
                    if dep_name in task_name_to_id:
                        resolved_deps.append(task_name_to_id[dep_name])
                    else:
                        raise ValueError(f"Dependency '{dep_name}' not found for task '{task.name}'")
                
                task.dependencies = resolved_deps
            
            # Validate no circular dependencies
            self._validate_dependencies(workflow_execution.tasks)
            
            self.active_executions[workflow_execution.id] = workflow_execution
            
            self.logger.info(f"Created workflow: {name} (ID: {workflow_execution.id})")
            self.metrics.increment_counter('workflows_created')
            
            return workflow_execution.id
            
        except Exception as e:
            self.logger.error(f"Failed to create workflow: {str(e)}")
            raise
    
    def _validate_dependencies(self, tasks: List[WorkflowTask]):
        """Validate that there are no circular dependencies"""
        task_map = {task.id: task for task in tasks}
        visited = set()
        rec_stack = set()
        
        def has_cycle(task_id):
            visited.add(task_id)
            rec_stack.add(task_id)
            
            task = task_map[task_id]
            for dep_id in task.dependencies:
                if dep_id not in visited:
                    if has_cycle(dep_id):
                        return True
                elif dep_id in rec_stack:
                    return True
            
            rec_stack.remove(task_id)
            return False
        
        for task in tasks:
            if task.id not in visited:
                if has_cycle(task.id):
                    raise ValueError(f"Circular dependency detected involving task '{task.name}'")

    async def execute_workflow(self, workflow_id: str) -> WorkflowExecution:
        """
        Execute a workflow with self-healing capabilities
        
        Args:
            workflow_id: ID of the workflow to execute
            
        Returns:
            WorkflowExecution: Completed execution details
        """
        if workflow_id not in self.active_executions:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        execution = self.active_executions[workflow_id]
        execution.status = WorkflowStatus.RUNNING
        execution.start_time = time.time()
        
        try:
            self.logger.info(f"Starting workflow execution: {execution.name}")
            self._trigger_event('execution_started', execution)
            
            # Execute tasks with dependency resolution
            await self._execute_tasks_with_dependencies(execution)
            
            # Calculate final metrics
            execution.end_time = time.time()
            execution.total_execution_time = execution.end_time - execution.start_time
            execution.success_rate = self._calculate_success_rate(execution)
            
            # Determine final status
            if all(task.status == WorkflowStatus.COMPLETED for task in execution.tasks):
                execution.status = WorkflowStatus.COMPLETED
                self.successful_executions += 1
            elif any(task.status == WorkflowStatus.RECOVERED for task in execution.tasks):
                execution.status = WorkflowStatus.RECOVERED
                self.successful_executions += 1
            else:
                execution.status = WorkflowStatus.FAILED
            
            # Process feedback and learn
            await self._process_execution_feedback(execution)
            
            # Move to history
            self.execution_history.append(execution)
            del self.active_executions[workflow_id]
            
            self.total_executions += 1
            self._trigger_event('execution_completed', execution)
            
            self.logger.info(f"Workflow completed: {execution.name} - Status: {execution.status}")
            self.metrics.increment_counter('workflows_completed')
            self.metrics.record_gauge('workflow_success_rate', 
                                    self.successful_executions / self.total_executions)
            
            return execution
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.end_time = time.time()
            self.logger.error(f"Workflow execution failed: {str(e)}")
            self.metrics.increment_counter('workflow_failures')
            raise

    async def _execute_tasks_with_dependencies(self, execution: WorkflowExecution):
        """Execute tasks respecting dependency order"""
        completed_tasks = set()
        remaining_tasks = {task.id: task for task in execution.tasks}
        
        while remaining_tasks:
            # Find tasks ready to execute (dependencies satisfied)
            ready_tasks = []
            for task in remaining_tasks.values():
                if all(dep_id in completed_tasks for dep_id in task.dependencies):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                raise RuntimeError("Circular dependency detected in workflow")
            
            # Execute ready tasks in parallel
            tasks_to_execute = []
            for task in ready_tasks:
                tasks_to_execute.append(self._execute_single_task(task, execution))
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks_to_execute, return_exceptions=True)
            
            # Process results and update completion status
            for task, result in zip(ready_tasks, results):
                if isinstance(result, Exception):
                    self.logger.error(f"Task {task.name} failed: {str(result)}")
                
                completed_tasks.add(task.id)
                del remaining_tasks[task.id]

    async def _execute_single_task(self, task: WorkflowTask, execution: WorkflowExecution):
        """Execute a single task with error handling and recovery"""
        task.status = WorkflowStatus.RUNNING
        task.updated_at = time.time()
        
        start_time = time.time()
        self._trigger_event('task_started', task)
        
        try:
            # Merge dependency outputs into task input data
            enhanced_input_data = self._merge_dependency_outputs(task, execution)
            
            # Route task to appropriate agent
            agent = await self.task_router.route_task(task)
            
            # Execute task with monitoring
            with self.metrics.time_operation(f'task_execution_{task.agent_type}'):
                result = await agent.execute_task(enhanced_input_data)
            
            task.output_data = result
            task.status = WorkflowStatus.COMPLETED
            task.execution_time = time.time() - start_time
            
            self.logger.info(f"Task completed successfully: {task.name}")
            self._trigger_event('task_completed', task)
            
        except Exception as e:
            task.error_info = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': time.time()
            }
            
            self.logger.error(f"Task failed: {task.name} - {str(e)}")
            self._trigger_event('task_failed', task)
            
            # Attempt healing
            if task.retry_count < task.max_retries:
                await self._attempt_task_healing(task, execution)
            else:
                task.status = WorkflowStatus.FAILED
                task.execution_time = time.time() - start_time

    def _merge_dependency_outputs(self, task: WorkflowTask, execution: WorkflowExecution) -> Dict[str, Any]:
        """Merge dependency task outputs into task input data"""
        enhanced_input_data = task.input_data.copy()
        
        # Find dependency tasks and their outputs
        dependency_outputs = {}
        task_map = {t.id: t for t in execution.tasks}
        
        for dep_id in task.dependencies:
            if dep_id in task_map:
                dep_task = task_map[dep_id]
                if dep_task.output_data:
                    dependency_outputs[f"{dep_task.name}_output"] = dep_task.output_data
        
        # For Quality Checker tasks, specifically merge content from dependency tasks
        if task.agent_type == "quality_checker":
            # Look for content in dependency outputs
            for dep_name, dep_output in dependency_outputs.items():
                if "Content Generation" in dep_name:
                    # Extract content from LLM reasoning output
                    if isinstance(dep_output, dict) and 'result' in dep_output:
                        result = dep_output['result']
                        if isinstance(result, dict) and 'content' in result:
                            enhanced_input_data['content'] = result['content']
                        elif isinstance(result, str):
                            enhanced_input_data['content'] = result
                    # Also check for generated code
                elif "Code Generation" in dep_name:
                    if isinstance(dep_output, dict) and 'code' in dep_output:
                        enhanced_input_data['content'] = dep_output['code']
        
        # Add dependency outputs to input data for other use cases
        if dependency_outputs:
            enhanced_input_data['dependency_outputs'] = dependency_outputs
        
        return enhanced_input_data

    async def _attempt_task_healing(self, task: WorkflowTask, execution: WorkflowExecution):
        """Attempt to heal a failed task"""
        task.status = WorkflowStatus.HEALING
        task.retry_count += 1
        
        healing_event = {
            'task_id': task.id,
            'retry_attempt': task.retry_count,
            'error_info': task.error_info,
            'timestamp': time.time()
        }
        
        execution.healing_events.append(healing_event)
        self._trigger_event('healing_triggered', healing_event)
        
        try:
            # Use healing orchestrator to determine recovery strategy
            recovery_strategy = await self.healing_orchestrator.analyze_failure(
                task, execution.healing_events
            )
            
            # Apply recovery strategy
            healed_task = await self.healing_orchestrator.apply_recovery(
                task, recovery_strategy
            )
            
            # Re-execute the healed task
            await self._execute_single_task(healed_task, execution)
            
            if healed_task.status == WorkflowStatus.COMPLETED:
                healed_task.status = WorkflowStatus.RECOVERED
                self.logger.info(f"Task healed successfully: {task.name}")
                
                # Update healing success rate
                healing_successes = sum(1 for event in execution.healing_events 
                                      if event.get('success', False))
                self.healing_success_rate = healing_successes / len(execution.healing_events)
                
        except Exception as healing_error:
            self.logger.error(f"Healing failed for task {task.name}: {str(healing_error)}")
            task.status = WorkflowStatus.FAILED

    async def _process_execution_feedback(self, execution: WorkflowExecution):
        """Process feedback and update learning systems"""
        try:
            # Generate feedback scores
            feedback_scores = await self.feedback_processor.generate_feedback(execution)
            execution.feedback_scores = feedback_scores
            
            # Update memory system with learnings
            await self.memory_system.store_execution_knowledge(execution)
            
            # Update agent capabilities based on performance
            await self.agent_manager.update_agent_performance(execution)
            
        except Exception as e:
            self.logger.error(f"Failed to process feedback: {str(e)}")

    def _calculate_success_rate(self, execution: WorkflowExecution) -> float:
        """Calculate success rate for the execution"""
        if not execution.tasks:
            return 0.0
        
        successful_tasks = sum(1 for task in execution.tasks 
                             if task.status in [WorkflowStatus.COMPLETED, WorkflowStatus.RECOVERED])
        return successful_tasks / len(execution.tasks)

    def _trigger_event(self, event_name: str, data: Any):
        """Trigger event handlers"""
        if event_name in self.event_handlers:
            for handler in self.event_handlers[event_name]:
                try:
                    handler(data)
                except Exception as e:
                    self.logger.error(f"Event handler failed: {str(e)}")

    def add_event_handler(self, event_name: str, handler: Callable):
        """Add an event handler"""
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
        self.event_handlers[event_name].append(handler)

    def get_execution_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current status of a workflow execution"""
        if workflow_id in self.active_executions:
            execution = self.active_executions[workflow_id]
        else:
            execution = next((e for e in self.execution_history if e.id == workflow_id), None)
            if not execution:
                raise ValueError(f"Workflow {workflow_id} not found")
        
        return {
            'id': execution.id,
            'name': execution.name,
            'status': execution.status.value,
            'progress': len([t for t in execution.tasks if t.status == WorkflowStatus.COMPLETED]) / len(execution.tasks),
            'success_rate': execution.success_rate,
            'healing_events': len(execution.healing_events),
            'execution_time': execution.total_execution_time,
            'task_count': len(execution.tasks)
        }

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system performance metrics"""
        return {
            'total_executions': self.total_executions,
            'successful_executions': self.successful_executions,
            'overall_success_rate': self.successful_executions / max(1, self.total_executions),
            'healing_success_rate': self.healing_success_rate,
            'active_workflows': len(self.active_executions),
            'agent_count': self.agent_manager.get_agent_count(),
            'memory_size': self.memory_system.get_memory_size(),
            'avg_execution_time': sum(e.total_execution_time for e in self.execution_history) / max(1, len(self.execution_history))
        } 