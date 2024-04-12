import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.stattools import adfuller


class AnomalyCatcher:
    def __init__(self, methods=None, auto_debug=True):
        if methods is None:
            methods = ['adfuller', 'moving_std', 'cusum', 'isolation_forest']
        self.methods = methods
        self.auto_debug = auto_debug

    def detect_anomalies(self, data):
        anomalies = np.array([], dtype=int)
        if 'adfuller' in self.methods:
            anomalies = np.union1d(anomalies, self._adfuller_test(data))
        if 'moving_std' in self.methods:
            anomalies = np.union1d(anomalies, self._moving_std_test(data))
        if 'cusum' in self.methods:
            anomalies = np.union1d(anomalies, self._cusum_test(data))
        if 'isolation_forest' in self.methods:
            anomalies = np.union1d(anomalies, self._isolation_forest_test(data))

        if len(anomalies) > 0:
            self._visualize_anomalies(data, anomalies)
            print("Anomalies detected!")
            if self.auto_debug:
                print("Automatic debugging initiated.")
            else:
                print("Manual debugging required.")

        return anomalies

    def _adfuller_test(self, data, significance=0.05):
        result = adfuller(data)
        p_value = result[1]
        if p_value > significance:
            return np.array([len(data)-1])
        return np.array([])

    def _moving_std_test(self, data, window=30, multiplier=2):
        moving_avg = np.mean(data[-window:])
        moving_std = np.std(data[-window:])
        return np.where((data[-window:] - moving_avg) > multiplier * moving_std)[0] + len(data) - window

    def _cusum_test(self, data, threshold=5):
        mean = np.mean(data)
        cumsum_values = cumsum(data - mean)
        significant_changes = np.where((cumsum_values > threshold) | (cumsum_values < -threshold))[0]
        return significant_changes

    def _isolation_forest_test(self, data):
        clf = IsolationForest(contamination=0.01)
        clf.fit(data.reshape(-1, 1))
        predictions = clf.predict(data.reshape(-1, 1))
        return np.where(predictions == -1)[0]

    def _visualize_anomalies(self, data, anomalies):
        # Позже можно доработать
        plt.figure(figsize=(10, 4))
        plt.plot(data, label='Data')
        plt.scatter(anomalies, data[anomalies], color='red', label='Anomalies')
        plt.legend()
        plt.show()

class RetrainingManager:
    def __init__(self, model, anomaly_catcher, data, strategy='basic'):
        self.model = model
        self.anomaly_catcher = anomaly_catcher
        self.data = data
        self.strategy = strategy
    
    def determine_retraining_period(self):
        anomalies = self.anomaly_catcher.detect_anomalies(self.data)
        if len(anomalies) > 0:
            return self._apply_strategy(anomalies)
        return 60  # Если аномалий нет, реже дообучаем
    
    def _apply_strategy(self, anomalies):
        if self.strategy == 'basic':
            # Базовая стратегия: количество дней обратно пропорционально количеству аномалий
            return max(7, 30 // len(anomalies))
        elif self.strategy == 'statistical':
            # Статистическая стратегия: использование меры изменчивости данных
            std_dev = np.std(self.data)
            if std_dev > 1.0:
                return 7  # Частое дообучение при высокой изменчивости
            else:
                return 30  # Реже дообучаем при низкой изменчивости
        elif self.strategy == 'adaptive':
            # Адаптивная стратегия: анализируем диапазон данных за последние 30 дней
            recent_range = np.ptp(self.data[-30:])
            return max(7, int(30 * (1 - recent_range / np.ptp(self.data))))
        else:
            return 30  # Фоллбэк стратегия
    
    def retrain_model(self, period):
        print(f"Retraining the model using data from the last {period} days...")
        training_data = self.data[-period:]
        self.model.fit(training_data) # Предположим, здесь происходит дообучение
        return "Model retrained successfully for period: " + str(period) + " days"