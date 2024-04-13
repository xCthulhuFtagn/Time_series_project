import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.stattools import adfuller
import logging
import datetime

class AnomalyCatcher:
    def __init__(self, methods=None, auto_debug=True):
        if methods is None:
            methods = ['adfuller', 'moving_std', 'cusum', 'isolation_forest']
        self.methods = methods
        self.auto_debug = auto_debug

    def detect_anomalies(self, data):
        anomalies = np.array([], dtype=int)  # Указываем явно тип данных
        if 'adfuller' in self.methods:
            anomalies = np.union1d(anomalies, self._adfuller_test(data).astype(int))
        if 'moving_std' in self.methods:
            anomalies = np.union1d(anomalies, self._moving_std_test(data).astype(int))
        if 'cusum' in self.methods:
            anomalies = np.union1d(anomalies, self._cusum_test(data).astype(int))
        if 'isolation_forest' in self.methods:
            anomalies = np.union1d(anomalies, self._isolation_forest_test(data).astype(int))

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


    def _adfuller_test(self, data, significance=0.05):
        result = adfuller(data)
        p_value = result[1]
        if p_value > significance:
            return np.array([len(data)-1])
        return np.array([])

    def _moving_std_test(self, data, window=30, multiplier=2):
        moving_avg = np.mean(data[-window:])
        moving_std = np.std(data[-window:])
        relative_anomalies = np.where((data[-window:] - moving_avg) > multiplier * moving_std)[0]
        absolute_anomalies = relative_anomalies + (len(data) - window)
        return absolute_anomalies.astype(int)  # Приведение типов к integer


    def _cusum_test(self, data, threshold=5):
        mean = np.mean(data)
        cumsum_values = np.cumsum(data - mean)
        significant_changes = np.where((cumsum_values > threshold) | (cumsum_values < -threshold))[0]
        return significant_changes

    def _isolation_forest_test(self, data):
        clf = IsolationForest(contamination=0.01)
        clf.fit(data.reshape(-1, 1))
        predictions = clf.predict(data.reshape(-1, 1))
        return np.where(predictions == -1)[0]

    def _visualize_anomalies(self, data, anomalies):
        plt.figure(figsize=(10, 4))
        plt.plot(data, label='Data')
        # Проверка и фильтрация
        valid_anomalies = anomalies[(anomalies < len(data)) & (anomalies >= 0)]
        plt.scatter(valid_anomalies, data[valid_anomalies], color='red', label='Anomalies')
        plt.legend()
        plt.show()





class RetrainingManager:
    def __init__(self, model, anomaly_catcher, data, strategy='basic', logger=None):
        self.model = model
        self.anomaly_catcher = anomaly_catcher
        self.data = data
        self.strategy = strategy
        self.logger = logger or logging.getLogger(__name__)
        self.last_retraining_info = {}  # Словарь для хранения информации о последнем дообучении

    def determine_retraining_period(self):
        self.logger.info("Determining retraining period based on detected anomalies.")
        anomalies = self.anomaly_catcher.detect_anomalies(self.data)
        if len(anomalies) > 0:
            period = self._apply_strategy(anomalies)
            self.logger.info(f"Retraining period set to {period} days due to anomalies.")
            return period
        return 60
    
    def retrain_model(self, period):
        self.logger.info(f"Retraining the model using data from the last {period} days...")
        training_data = self.data[-period:]
        self.model.fit(training_data) # Предположим, здесь происходит дообучение
        self.last_retraining_info = {
            'period': period,
            'timestamp': datetime.datetime.now(),
            'data_points_used': len(training_data)
        }
        self.logger.info("Model retrained successfully.")
        return "Model retrained successfully for period: " + str(period) + " days"
    
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


if __name__ == "__main__":
    # Пример использования
    print("Демонстрационный режим модуля")
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    model = IsolationForest()
    anomaly_catcher = AnomalyCatcher(auto_debug=True)
    data = np.random.normal(size=100) + np.linspace(-1, 1, 100)
    data[40:45] += 7
    manager = RetrainingManager(model, anomaly_catcher, data, strategy='statistical', logger=logger)
    retraining_period = manager.determine_retraining_period()
    retraining_result = manager.retrain_model(retraining_period)
    report = manager.generate_retraining_report()
    print(report)
