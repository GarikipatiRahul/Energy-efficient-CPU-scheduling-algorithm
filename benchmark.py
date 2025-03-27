import time
import random
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
from scheduler import EnergyEfficientScheduler, Task
from energy_manager import EnergyManager
from ml_predictor import WorkloadPredictor

class TraditionalScheduler:
    def __init__(self, algorithm: str):
        self.algorithm = algorithm
        self.energy_consumption = 0.0
        
    def schedule(self, tasks: List[Task]) -> Dict:
        if self.algorithm == "round_robin":
            return self._round_robin(tasks)
        elif self.algorithm == "sjf":
            return self._shortest_job_first(tasks)
        elif self.algorithm == "edf":
            return self._earliest_deadline_first(tasks)
        
    def _round_robin(self, tasks: List[Task]) -> Dict:
        time_quantum = 0.1
        total_time = sum(task.cpu_burst for task in tasks)
        context_switches = len(tasks) * (total_time / time_quantum)
        energy = total_time * 1.0  # Assuming constant power draw
        
        return {
            "energy": energy,
            "context_switches": context_switches,
            "completion_time": total_time
        }
        
    def _shortest_job_first(self, tasks: List[Task]) -> Dict:
        sorted_tasks = sorted(tasks, key=lambda t: t.cpu_burst)
        total_time = sum(task.cpu_burst for task in sorted_tasks)
        energy = total_time * 0.9  # Slightly better than RR
        
        return {
            "energy": energy,
            "context_switches": len(tasks) - 1,
            "completion_time": total_time
        }
        
    def _earliest_deadline_first(self, tasks: List[Task]) -> Dict:
        sorted_tasks = sorted(tasks, key=lambda t: t.deadline)
        total_time = sum(task.cpu_burst for task in sorted_tasks)
        energy = total_time * 0.95  # Better deadline meeting, slightly higher power
        
        return {
            "energy": energy,
            "context_switches": len(tasks) - 1,
            "completion_time": total_time
        }

class Benchmark:
    def __init__(self):
        self.energy_efficient_scheduler = EnergyEfficientScheduler()
        self.energy_manager = EnergyManager()
        self.workload_predictor = WorkloadPredictor()
        
        self.traditional_schedulers = {
            "round_robin": TraditionalScheduler("round_robin"),
            "sjf": TraditionalScheduler("sjf"),
            "edf": TraditionalScheduler("edf")
        }
        
    def generate_synthetic_workload(self, num_tasks: int) -> List[Task]:
        tasks = []
        for i in range(num_tasks):
            is_realtime = random.random() < 0.3
            task = Task(
                id=i,
                priority=random.randint(1, 3),
                deadline=random.uniform(0.1, 0.5) if is_realtime else 1.0,
                cpu_burst=random.uniform(0.1, 0.3),
                memory_footprint=random.randint(1, 50),
                is_realtime=is_realtime,
                power_profile=random.uniform(0.5, 1.0)
            )
            tasks.append(task)
        return tasks

    def run_benchmark(self, num_tasks: int = 10) -> Dict:
        print("Generating tasks...")
        tasks = self.generate_synthetic_workload(num_tasks)
        results = {}
        
        # Test energy-efficient scheduler
        print("Testing energy-efficient scheduler...")
        for task in tasks:
            self.energy_efficient_scheduler.add_task(task)
        
        self.energy_efficient_scheduler.start()
        
        results["energy_efficient"] = {
            "metrics": self.energy_efficient_scheduler.get_metrics(),
            "power_savings": self.energy_manager.get_power_savings_estimate()
        }
        
        # Test traditional schedulers
        print("Testing traditional schedulers...")
        for name, scheduler in self.traditional_schedulers.items():
            print(f"- Running {name}...")
            results[name] = scheduler.schedule(tasks)
            
        return results

    def plot_results(self, results: Dict) -> None:
        metrics = ['energy_consumption', 'context_switches', 'completion_time', 'missed_deadlines']
        schedulers = list(results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Scheduler Performance Comparison')
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = []
            for scheduler in schedulers:
                if metric in results[scheduler]:
                    values.append(results[scheduler][metric])
                elif metric == 'completion_time' and scheduler == 'energy_efficient':
                    # Estimate completion time for energy efficient scheduler
                    values.append(sum(task.cpu_burst for task in self.energy_efficient_scheduler.completed_tasks))
                else:
                    values.append(0)
            
            ax = axes[i]
            bars = ax.bar(schedulers, values)
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylabel('Value')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
        
def main():
    print("Quick Benchmark of Energy-Efficient CPU Scheduler")
    print("-" * 50)
    
    benchmark = Benchmark()
    results = benchmark.run_benchmark(num_tasks=10)
    
    print("\nResults Summary:")
    print("-" * 50)
    
    for scheduler, metrics in results.items():
        print(f"\n{scheduler.upper()}:")
        for metric, value in metrics.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    print(f"  {k}: {v:.2f}")
            else:
                print(f"  {metric}: {value:.2f}")
    
    print("\nGenerating visualization...")
    benchmark.plot_results(results)
    print("Done! Results saved to 'benchmark_results.png'")

if __name__ == "__main__":
    main()
