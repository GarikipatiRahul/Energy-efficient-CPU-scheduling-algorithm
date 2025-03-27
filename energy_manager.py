import psutil
from typing import Dict, List
import numpy as np
from dataclasses import dataclass

@dataclass
class PowerProfile:
    voltage: float
    frequency: float
    power_draw: float

class EnergyManager:
    def __init__(self):
        self.voltage_levels = [0.8, 0.9, 1.0, 1.1, 1.2]  # Volts
        self.frequency_levels = [1.0, 1.5, 2.0, 2.5, 3.0]  # GHz
        self.power_profiles = self._generate_power_profiles()
        self.current_profile = None
        
        # Temperature thresholds
        self.TEMP_THRESHOLD_HIGH = 80  # Celsius
        self.TEMP_THRESHOLD_LOW = 60
        
        # Power consumption history
        self.power_history = []
        self.max_history_size = 1000

    def _generate_power_profiles(self) -> List[PowerProfile]:
        profiles = []
        for v in self.voltage_levels:
            for f in self.frequency_levels:
                # P = C * V^2 * f (simplified power equation)
                power = 0.5 * (v ** 2) * f  # Constant C assumed as 0.5
                profiles.append(PowerProfile(v, f, power))
        return profiles

    def get_optimal_profile(self, utilization: float, temperature: float) -> PowerProfile:
        """
        Select optimal voltage-frequency pair based on current conditions
        """
        if temperature >= self.TEMP_THRESHOLD_HIGH:
            # Thermal throttling needed
            profiles = [p for p in self.power_profiles if p.frequency <= 2.0]
        elif temperature <= self.TEMP_THRESHOLD_LOW:
            # Can use full performance
            profiles = self.power_profiles
        else:
            # Moderate profiles
            profiles = [p for p in self.power_profiles if p.frequency <= 2.5]

        # Find optimal profile based on utilization
        target_freq = max(1.0, min(3.0, utilization * 3.0))
        optimal = min(profiles, 
                     key=lambda p: abs(p.frequency - target_freq))
        
        self.current_profile = optimal
        return optimal

    def monitor_power_consumption(self) -> Dict:
        """
        Monitor system-wide power consumption
        """
        try:
            # Get CPU frequency
            freq = psutil.cpu_freq().current / 1000  # Convert to GHz
            
            # Get CPU utilization
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get temperature if available
            temp = 0
            try:
                temp = psutil.sensors_temperatures()['coretemp'][0].current
            except:
                temp = 70  # Default assumption if sensors unavailable
            
            # Estimate power consumption
            if self.current_profile:
                power = self.current_profile.power_draw * (cpu_percent / 100)
            else:
                power = 0.5 * (1.1 ** 2) * freq * (cpu_percent / 100)
            
            # Store in history
            self.power_history.append(power)
            if len(self.power_history) > self.max_history_size:
                self.power_history.pop(0)
            
            return {
                "current_power": power,
                "temperature": temp,
                "frequency": freq,
                "utilization": cpu_percent
            }
        except:
            return {
                "current_power": 0,
                "temperature": 70,
                "frequency": 1.0,
                "utilization": 0
            }

    def get_power_savings_estimate(self) -> float:
        """
        Calculate estimated power savings compared to baseline
        """
        if not self.power_history:
            return 0.0
            
        # Baseline assumes maximum voltage and frequency
        baseline_power = 0.5 * (self.voltage_levels[-1] ** 2) * self.frequency_levels[-1]
        actual_power = np.mean(self.power_history)
        
        savings = (baseline_power - actual_power) / baseline_power * 100
        return max(0, savings)

    def get_dvfs_recommendation(self, workload_prediction: float) -> PowerProfile:
        """
        Provide DVFS recommendations based on predicted workload
        """
        if workload_prediction < 0.3:
            # Light workload: use power-saving profile
            return min(self.power_profiles, key=lambda p: p.power_draw)
        elif workload_prediction > 0.7:
            # Heavy workload: use performance profile
            return max(self.power_profiles, key=lambda p: p.frequency)
        else:
            # Moderate workload: use balanced profile
            mid_idx = len(self.power_profiles) // 2
            return self.power_profiles[mid_idx]
