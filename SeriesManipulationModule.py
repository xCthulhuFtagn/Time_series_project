import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.stattools import adfuller
import logging
import datetime
from scipy.stats import norm
import seaborn as sns

from IPython.display import display


def normal_likelihood(value, mean_0, mean_8, std):
    return np.log(norm.pdf(value, mean_0, std) / 
                norm.pdf(value, mean_8, std))
        
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
        self.history = []
        self.dates = []  # Добавляем список для хранения дат

    def update(self, value, date=None):
        print(f"gotten: {value}")
        zeta_k = normal_likelihood(value, self.mean_diff, 0., 1.)
        self._stat = max(0, self._stat + zeta_k)
        self.history.append(self._stat)
        if date is not None:
            self.dates.append(date)
        return self._stat > self.threshold

    def visualize_history(self):
        plt.figure(figsize=(12, 6))
        x_axis = self.dates if self.dates else list(range(len(self.history)))
        plt.plot(x_axis, self.history, label="Cusum Statistic History", color='blue')
        plt.axhline(y=self.threshold, color='red', linestyle='--', label='Threshold')
        plt.title("History of Cusum Statistic Over Time")
        plt.xlabel("Date" if self.dates else "Time Step")
        plt.ylabel("Cusum Statistic Value")
        plt.legend()
        plt.grid(True)
        plt.gcf().autofmt_xdate()  # Автоматически форматирует даты для лучшего отображения
        plt.show()

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
    def __init__(self, new_value_weight_mean=0.01, new_value_weight_var=0.01, mean_diff=1.0, cusum_threshold=30., window_size=30):
        self.mean_estimator = MeanExp(new_value_weight_mean)
        self.variance_estimator = MeanExp(new_value_weight_var)
        self.cusum = AdjustedCusum(mean_diff, cusum_threshold)
        self.window_size = window_size  # Добавление размера окна как параметра класса
        self.stat_trajectory, self.mean_values, self.var_values = [], [], []
        self.values = []

    def detect_anomalies(self, y):
        try:
            mean_estimate = self.mean_estimator.value()
        except Exception:
            mean_estimate = 0.
        try:
            var_estimate = self.variance_estimator.value()
        except Exception:
            var_estimate = 1.

        adjusted_value = (y - mean_estimate) / np.sqrt(var_estimate)
        print(f"adj: {adjusted_value}")
        cusum_trigger = self.cusum.update(adjusted_value)
        
        self.mean_estimator.update(y)
        diff_value = (y - mean_estimate) ** 2
        self.variance_estimator.update(diff_value)
        
        self.values.append(y)
        self.stat_trajectory.append(self.cusum._stat)
        self.mean_values.append(mean_estimate)
        self.var_values.append(np.sqrt(var_estimate))

        return cusum_trigger

    def visualize_mean_variance(self,ax=None):
        sns.lineplot(self.values,ax=None)
        sns.lineplot(np.array(self.mean_values),ax=ax)
        sns.lineplot(np.array(self.mean_values) + np.sqrt(self.var_values),ax=ax)
        sns.lineplot(np.array(self.mean_values) - np.sqrt(self.var_values),ax=ax)

    def visualize_stat_trajectory(self,ax=None):
        sns.lineplot(self.stat_trajectory, ax=ax)

class RetrainingManager:
    def __init__(self, model, anomaly_catcher, data, period = 60, strategy='none', logger=None):
        self.model = model
        self.anomaly_catcher = anomaly_catcher
        self.data_x, self.data_y = data[0], data[1] 
        self.strategy = strategy
        self.logger = logger or logging.getLogger(__name__)
        self.last_retraining_info = {}  # Словарь для хранения информации о последнем дообучении
        self.period = period

    def detect(self, y) -> bool:
        return self.anomaly_catcher.detect_anomalies(y)

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