import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.stattools import adfuller
import logging
import datetime

from IPython.display import display


class AnomalyCatcher:
    def __init__(self, methods=None, auto_debug=True, window_size=60, adfuller_p_value=0.05, moving_std_multiplier=2, cusum_threshold=1, isolation_forest_contamination=0.01):
        if methods is None:
            methods = ['adfuller', 'moving_std', 'cusum', 'isolation_forest']
        self.methods = methods
        self.auto_debug = auto_debug
        self.window_size = window_size
        self.adfuller_p_value = adfuller_p_value
        self.moving_std_multiplier = moving_std_multiplier
        self.cusum_threshold = cusum_threshold
        self.isolation_forest_contamination = isolation_forest_contamination

    def detect_anomalies(self, data):
        anomalies = np.array([], dtype=int)
        display(data)
        for method in self.methods:
            if method == 'adfuller':
                anomalies = np.union1d(anomalies, self._adfuller_test(data))
            elif method == 'moving_std':
                anomalies = np.union1d(anomalies, self._moving_std_test(data))
            elif method == 'cusum':
                anomalies = np.union1d(anomalies, self._cusum_test(data))
            elif method == 'isolation_forest':
                anomalies = np.union1d(anomalies, self._isolation_forest_test(data))
        
        if len(anomalies) > 0:
            self._visualize_anomalies(data, anomalies)
            print("Anomalies detected!")
            if self.auto_debug:
                print("Automatic debugging initiated.")
            else:
                print("Manual debugging required.")
        else:
            print("No anomalies detected.")
        return anomalies

    def _adfuller_test(self, data):
        result = adfuller(data[-self.window_size:])
        p_value = result[1]
        if p_value > self.adfuller_p_value:
            return np.array([len(data)-1])
        return np.array([])

    def _moving_std_test(self, data):
        if len(data) < self.window_size:
            return np.array([])
        moving_avg = np.mean(data[-self.window_size:])
        moving_std = np.std(data[-self.window_size:])
        anomalies = np.where((data[-self.window_size:] - moving_avg) > self.moving_std_multiplier * moving_std)[0]
        return anomalies + len(data) - self.window_size

    def _cusum_test(self, data):
        if len(data) < self.window_size:
            return np.array([])
        data_segment = data[-self.window_size:]
        mean = np.mean(data_segment)
        cumsum_values = np.cumsum(data_segment - mean)
        anomalies = np.where((cumsum_values > self.cusum_threshold) | (cumsum_values < -self.cusum_threshold))[0]
        return anomalies + len(data) - self.window_size

    def _isolation_forest_test(self, data):
        if len(data) < self.window_size:
            return np.array([])
        data_segment = data[-self.window_size:].reshape(-1, 1)
        clf = IsolationForest(contamination=self.isolation_forest_contamination)
        clf.fit(data_segment)
        predictions = clf.predict(data_segment)
        return np.where(predictions == -1)[0] + len(data) - self.window_size

    def _visualize_anomalies(self, data, anomalies):
        plt.figure(figsize=(10, 4))
        plt.plot(data, label='Data')
        valid_anomalies = anomalies[anomalies < len(data)]
        plt.scatter(valid_anomalies, data[valid_anomalies], color='red', label='Anomalies')
        plt.axvspan(len(data) - self.window_size, len(data), color='yellow', alpha=0.3, label='Analysis Window')
        plt.legend()
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
        return anomalies == []

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