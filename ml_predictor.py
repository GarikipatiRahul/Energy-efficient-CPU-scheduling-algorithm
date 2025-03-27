import numpy as np
from sklearn.ensemble import RandomForestRegressor
from collections import deque
from typing import List, Tuple

class WorkloadPredictor:
    def __init__(self, history_size: int = 100):
        self.model = RandomForestRegressor(n_estimators=100)
        self.history_size = history_size
        self.workload_history = deque(maxlen=history_size)
        self.feature_history = deque(maxlen=history_size)
        self.trained = False
        
        # Feature importance tracking
        self.feature_names = [
            'avg_cpu_usage',
            'task_arrival_rate',
            'avg_task_size',
            'time_of_day',
            'day_of_week'
        ]
        
    def _prepare_features(self, current_state: dict) -> np.ndarray:
        """
        Extract and normalize features from current system state
        """
        features = [
            current_state.get('cpu_usage', 0),
            current_state.get('task_arrival_rate', 0),
            current_state.get('avg_task_size', 0),
            current_state.get('time_of_day', 0) / 24.0,  # Normalize to [0,1]
            current_state.get('day_of_week', 0) / 7.0    # Normalize to [0,1]
        ]
        return np.array(features).reshape(1, -1)

    def update(self, current_state: dict, actual_workload: float) -> None:
        """
        Update the model with new data
        """
        features = self._prepare_features(current_state)
        self.feature_history.append(features)
        self.workload_history.append(actual_workload)
        
        # Train model when enough data is collected
        if len(self.workload_history) >= self.history_size // 2:
            self._train_model()

    def _train_model(self) -> None:
        """
        Train the prediction model on historical data
        """
        if len(self.workload_history) < 10:  # Minimum required samples
            return
            
        X = np.vstack(self.feature_history)
        y = np.array(self.workload_history)
        
        try:
            self.model.fit(X, y)
            self.trained = True
            
            # Update feature importance
            importances = self.model.feature_importances_
            self.feature_importance = dict(zip(self.feature_names, importances))
            
        except Exception as e:
            print(f"Training failed: {e}")
            self.trained = False

    def predict_workload(self, current_state: dict) -> Tuple[float, float]:
        """
        Predict future workload and return confidence
        """
        if not self.trained:
            # Return conservative estimate if model isn't trained
            return 0.5, 0.0
            
        features = self._prepare_features(current_state)
        
        try:
            # Get predictions from all trees
            predictions = [tree.predict(features) for tree in self.model.estimators_]
            predictions = np.array(predictions).flatten()
            
            # Calculate mean and confidence
            mean_prediction = np.mean(predictions)
            confidence = 1.0 - (np.std(predictions) / max(mean_prediction, 1e-5))
            
            return float(mean_prediction), float(confidence)
            
        except Exception as e:
            print(f"Prediction failed: {e}")
            return 0.5, 0.0

    def get_feature_importance(self) -> dict:
        """
        Return the importance of each feature in prediction
        """
        if not self.trained:
            return {name: 0.0 for name in self.feature_names}
        return self.feature_importance

    def detect_anomaly(self, current_state: dict, threshold: float = 2.0) -> bool:
        """
        Detect if current workload is anomalous
        """
        if not self.trained or len(self.workload_history) < 10:
            return False
            
        prediction, _ = self.predict_workload(current_state)
        actual = current_state.get('cpu_usage', 0)
        
        # Calculate z-score
        recent_workloads = list(self.workload_history)[-10:]
        mean_workload = np.mean(recent_workloads)
        std_workload = np.std(recent_workloads) + 1e-5  # Avoid division by zero
        
        z_score = abs(actual - mean_workload) / std_workload
        
        return z_score > threshold
