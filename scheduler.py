import time
import threading
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import deque

class PowerState(Enum):
    FULL_POWER = 1
    BALANCED = 2
    POWER_SAVE = 3
    IDLE = 4

@dataclass
class Task:
    id: int
    priority: int
    deadline: float
    cpu_burst: float
    memory_footprint: int
    is_realtime: bool
    power_profile: float  # Expected power consumption

class Core:
    def __init__(self, core_id: int, max_freq: float):
        self.core_id = core_id
        self.current_task: Optional[Task] = None
        self.current_freq = max_freq
        self.max_freq = max_freq
        self.power_state = PowerState.IDLE
        self.utilization_history = deque(maxlen=100)

    def set_frequency(self, freq: float) -> None:
        self.current_freq = min(freq, self.max_freq)

    def update_utilization(self, utilization: float) -> None:
        self.utilization_history.append(utilization)

class EnergyEfficientScheduler:
    def __init__(self, num_cores: int = 4):
        self.cores = [Core(i, max_freq=2.5) for i in range(num_cores)]
        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.running = False
        self.lock = threading.Lock()
        
        # Energy efficiency parameters
        self.power_threshold = 0.8  # 80% threshold for DVFS
        self.idle_timeout = 0.1  # seconds before core goes to idle
        self.learning_rate = 0.01
        
        # Performance metrics
        self.energy_consumption = 0.0
        self.context_switches = 0
        self.missed_deadlines = 0

    def add_task(self, task: Task) -> None:
        with self.lock:
            self.task_queue.append(task)
            self._sort_task_queue()

    def _sort_task_queue(self) -> None:
        # Sort based on priority, deadline, and energy profile
        self.task_queue.sort(
            key=lambda t: (
                -t.priority,  # Higher priority first
                t.deadline,   # Earlier deadline first
                t.power_profile  # Lower power consumption first
            )
        )

    def _select_optimal_core(self, task: Task) -> Optional[Core]:
        best_core = None
        min_power_impact = float('inf')
        
        for core in self.cores:
            if core.current_task is None:
                power_impact = self._calculate_power_impact(core, task)
                if power_impact < min_power_impact:
                    min_power_impact = power_impact
                    best_core = core
        return best_core

    def _calculate_power_impact(self, core: Core, task: Task) -> float:
        # Consider frequency scaling and power state transition costs
        base_power = core.current_freq * task.power_profile
        
        # Add power state transition cost
        if core.power_state == PowerState.IDLE:
            base_power += 0.2  # Wake-up energy penalty
        
        # Add thermal impact based on recent utilization
        history = list(core.utilization_history)
        if history:
            recent_history = history[-5:] if len(history) >= 5 else history
            recent_util = sum(recent_history) / len(recent_history)
            thermal_impact = recent_util * 0.1
            base_power += thermal_impact
        
        return base_power

    def _adjust_core_frequency(self, core: Core, task: Task) -> None:
        # Implement DVFS based on task requirements and current system state
        if task.is_realtime:
            core.set_frequency(core.max_freq)
        else:
            # Calculate average utilization from history
            if len(core.utilization_history) > 0:
                avg_utilization = sum(core.utilization_history) / len(core.utilization_history)
            else:
                avg_utilization = 0.5  # Default to 50% if no history
                
            # Update utilization with current task
            current_utilization = task.cpu_burst / task.deadline if task.deadline > 0 else 0.5
            core.update_utilization(current_utilization)
            
            if avg_utilization < self.power_threshold:
                new_freq = core.current_freq * 0.8  # Reduce frequency by 20%
                core.set_frequency(max(1.0, new_freq))  # Don't go below 1.0 GHz
            else:
                # Increase frequency if utilization is high
                new_freq = min(core.max_freq, core.current_freq * 1.2)
                core.set_frequency(new_freq)

    def _execute_task(self, core: Core, task: Task) -> None:
        # Quick simulation of task execution
        if core.current_task is not None:
            self.context_switches += 1
        
        core.current_task = task
        core.power_state = PowerState.FULL_POWER
        self._adjust_core_frequency(core, task)
        
        # Simplified energy calculation
        energy_consumed = task.cpu_burst * core.current_freq * task.power_profile
        self.energy_consumption += energy_consumed
        
        if task.deadline < task.cpu_burst:
            self.missed_deadlines += 1
        
        core.current_task = None

    def start(self) -> None:
        self.running = True
        self._scheduler_loop()

    def stop(self) -> None:
        self.running = False

    def _scheduler_loop(self) -> None:
        max_tasks = 10  # Limit number of tasks to process
        processed_tasks = 0
        
        while self.running and processed_tasks < max_tasks:
            with self.lock:
                if self.task_queue:
                    task = self.task_queue.pop(0)
                    optimal_core = self._select_optimal_core(task)
                    
                    if optimal_core:
                        self._execute_task(optimal_core, task)
                        self.completed_tasks.append(task)
                        processed_tasks += 1
                    else:
                        self.task_queue.append(task)
            time.sleep(0.001)
        self.running = False

    def get_metrics(self) -> Dict:
        return {
            "energy_consumption": self.energy_consumption,
            "context_switches": self.context_switches,
            "missed_deadlines": self.missed_deadlines,
            "completed_tasks": len(self.completed_tasks)
        }
