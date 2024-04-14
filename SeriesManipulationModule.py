import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.stattools import adfuller
import logging
import datetime

from IPython.display import display


class Stat:
    def __init__(self, threshold, direction="unknown", init_stat=0.0):
        self.threshold = threshold
        self.direction = direction
        self._stat = init_stat

    def update(self, value):
        raise NotImplementedError("Must be implemented by subclass.")

class AdjustedCusum(Stat):
    def __init__(self, mean_diff, threshold, direction="unknown", init_stat=0.0):
        super().__init__(threshold, direction, init_stat)
        self.mean_diff = mean_diff
        
    def update(self, value):
        zeta_k = value - self.mean_diff / 2
        self._stat = max(0, self._stat + zeta_k)
        if self._stat > self.threshold:
            return True
        return False

class MeanExp:
    def __init__(self, new_value_weight):
        self._new_value_weight = new_value_weight
        self._values_sum = 0.0
        self._weights_sum = 0.0

    def update(self, new_value):
        self._values_sum = (1 - self._new_value_weight) * self._values_sum + new_value
        self._weights_sum = (1 - self._new_value_weight) * self._weights_sum + 1.0

    def value(self):
        if self._weights_sum <= 1:
            raise Exception('Not enough data')
        return self._values_sum / self._weights_sum

class AnomalyCatcher:
    def __init__(self, new_value_weight_mean=0.01, new_value_weight_var=0.01, mean_diff=1.0, cusum_threshold=30, window_size=30):
        self.mean_estimator = MeanExp(new_value_weight_mean)
        self.variance_estimator = MeanExp(new_value_weight_var)
        self.cusum = AdjustedCusum(mean_diff, cusum_threshold)
        self.window_size = window_size  # Добавление размера окна как параметра класса

    def process(self, data):
        mean_values, var_values, cusum_values = [], [], []
        for x in data[-self.window_size:]:  # Ограничение анализа последними window_size точками
            try:
                mean_estimate = self.mean_estimator.value()
            except Exception:
                mean_estimate = 0.
            try:
                var_estimate = self.variance_estimator.value()
            except Exception:
                var_estimate = 1.

            adjusted_value = (x - mean_estimate) / np.sqrt(var_estimate)
            cusum_trigger = self.cusum.update(adjusted_value)
            
            self.mean_estimator.update(x)
            diff_value = (x - mean_estimate) ** 2
            self.variance_estimator.update(diff_value)
            
            mean_values.append(mean_estimate)
            var_values.append(np.sqrt(var_estimate))
            cusum_values.append(self.cusum._stat)

        return mean_values, var_values, cusum_values, cusum_trigger

    def visualize(self, data, mean_estimates, var_estimates):
        plt.figure(figsize=(12, 6))
        times = np.arange(len(data))[-self.window_size:]
        data = data[-self.window_size:]
        mean_estimates = mean_estimates[-self.window_size:]
        var_estimates = var_estimates[-self.window_size:]
        
        plt.plot(times, data, label="Data", color='blue', alpha=0.5)
        plt.plot(times, mean_estimates, label="Estimated Mean", color='black')
        
        upper_bound = np.array(mean_estimates) + np.array(var_estimates)
        lower_bound = np.array(mean_estimates) - np.array(var_estimates)
        plt.fill_between(times, lower_bound, upper_bound, color='gray', alpha=0.3, label='Estimated Variance')
        
        plt.title("Data with Estimated Mean and Variance")
        plt.xlabel("Time Index")
        plt.ylabel("Values")
        plt.legend()
        plt.grid(True)
        plt.show()


class RetrainingManager:
    def __init__(self, model, anomaly_catcher, data, period = 60, strategy='none', logger=None):
        self.model = model
        self.anomaly_catcher = anomaly_catcher
        self.data_x, self.data_y = data[0], data[1] 
        self.strategy = strategy
        self.logger = logger or logging.getLogger(__name__)
        self.last_retraining_info = {}  # Словарь для хранения информации о последнем дообучении
        self.period = period
        
    def update(self, new_data):
        # print(f"{new_data[0]}")
        # print(f"{new_data[1]}")
        self.data_x.loc[self.data_x.index[-1] + datetime.timedelta(days=1)] = new_data[0]
        self.data_y.loc[self.data_y.index[-1] + datetime.timedelta(days=1)] = new_data[1]


    def initiate_detection(self) -> bool:
        anomalies = self.anomaly_catcher.detect_anomalies(self.data_y[-self.anomaly_catcher.window_size * 2:])
        return anomalies.size == 0

    def predict(self):
        if hasattr(self.model, 'predict'):
            predictions = self.model.predict(self.data)
            self.logger.info("Generated predictions.")
            return predictions
        else:
            self.logger.error("Model does not support prediction.")
            raise NotImplementedError("Model does not have a predict method.")
        
    def retrain_model(self):
        self.logger.info(f"Retraining the model using data from the last {self.period} days...")
        X_train, y_train = self.data_x[-self.period:], self.data_y[-self.period:]
        # display(X_train)
        # display(y_train)
        self.model.fit(X_train, y_train) # Предположим, здесь происходит дообучение
        self.last_retraining_info = {
            'period': self.period,
            'timestamp': datetime.datetime.now(),
            'data_points_used': self.period
        }
        self.logger.info("Model retrained successfully.")
        return "Model retrained successfully for period: " + str(self.period) + " days"
    
    def generate_retraining_report(self):
        if self.last_retraining_info:
            report = f"Last Retraining Report:\n"
            report += f"Timestamp: {self.last_retraining_info['timestamp']}\n"
            report += f"Data points used: {self.last_retraining_info['data_points_used']}\n"
            report += f"Training period: {self.last_retraining_info['period']} days\n"
            self.logger.info(report)
            return report
        else:
            return "No retraining has been performed yet."
            
    def _apply_strategy(self, anomalies):
        # По дефолту, стратегия отключена и не будет работать        
        if self.strategy == 'basic':
            # Базовая стратегия: количество дней обратно пропорционально количеству аномалий
            self.period = max(7, self.period // (1 + len(anomalies)))
        elif self.strategy == 'statistical':
            # Статистическая стратегия: использование меры изменчивости данных
            std_dev_factor = max(1, np.std(self.data[-self.period:]) / np.std(self.data))
            self.period = max(7, int(self.period * std_dev_factor))
        elif self.strategy == 'adaptive':
            recent_change = np.ptp(self.data[-self.period:])
            total_change = np.ptp(self.data)
            adaptive_period = int(self.period * (1 - recent_change / total_change))
        else:
            self.period = self.period