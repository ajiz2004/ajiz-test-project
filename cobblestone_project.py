import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# Function to simulate a data stream with regular patterns, seasonal elements, and random noise
def simulate_data_stream(n=100, anomaly_chance=0.05):
    try:
        if n <= 0:
            raise ValueError("Data stream length 'n' must be greater than 0.")
        data_stream = []
        for i in range(n):
            # Seasonal pattern (sin wave) + random noise
            value = 1000 * np.sin(i / 10) + random.uniform(-50, 50)
            
            # Simulate anomalies randomly
            if random.random() < anomaly_chance:
                value = random.uniform(5000, 10000)  # Anomalous high values
            data_stream.append(value)
        return data_stream
    except Exception as e:
        print(f"Error during data stream simulation: {e}")
        return []

# Exponentially weighted moving average for anomaly detection
class AnomalyDetector:
    def __init__(self, alpha=0.1, threshold_factor=3):
        if not (0 < alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1.")
        if threshold_factor <= 0:
            raise ValueError("Threshold factor must be greater than 0.")
        self.alpha = alpha  # Weight for moving average
        self.threshold_factor = threshold_factor  # Factor for anomaly threshold
        self.ema = None  # Exponentially weighted moving average
        self.ema_std = None  # Exponentially weighted standard deviation

    def detect_anomalies(self, data_stream):
        if not data_stream:
            raise ValueError("Data stream cannot be empty.")
        anomalies = []
        for i, value in enumerate(data_stream):
            if self.ema is None:  # Initialize EMA and EMA std
                self.ema = value
                self.ema_std = 0
            else:
                # Calculate new EMA and EMA of the standard deviation
                prev_ema = self.ema
                self.ema = self.alpha * value + (1 - self.alpha) * self.ema
                self.ema_std = np.sqrt(self.alpha * (value - prev_ema) ** 2 + (1 - self.alpha) * self.ema_std ** 2)
            
            # Calculate threshold for anomaly detection
            threshold = self.ema + self.threshold_factor * self.ema_std
            
            # Detect anomaly if value exceeds the threshold
            if value > threshold:
                anomalies.append(i)
        return anomalies

# Function to visualize the real-time data stream and anomalies
def visualize_real_time(data_stream, anomalies):
    try:
        if not data_stream or not isinstance(data_stream, list):
            raise ValueError("Data stream must be a non-empty list.")
        if not all(isinstance(x, (int, float)) for x in data_stream):
            raise ValueError("Data stream must contain only numeric values.")
        
        fig, ax = plt.subplots()
        ax.set_xlim(0, len(data_stream))
        ax.set_ylim(min(data_stream) - 10, max(data_stream) + 10)
        ax.set_title('Real-Time Data Stream Anomaly Detection')
        ax.set_xlabel('Time')
        ax.set_ylabel('Transaction Amount')

        line, = ax.plot([], [], lw=2, label="Data Stream", color="blue")
        anomaly_points, = ax.plot([], [], 'ro', label="Anomalies")
        ax.legend()

        # Initialization function for the animation
        def init():
            line.set_data([], [])
            anomaly_points.set_data([], [])
            return line, anomaly_points

        # Update function for the animation
        def update(frame):
            line.set_data(range(frame + 1), data_stream[:frame + 1])
            detected_anomalies = [data_stream[i] for i in anomalies if i <= frame]
            anomaly_times = [i for i in anomalies if i <= frame]
            anomaly_points.set_data(anomaly_times, detected_anomalies)
            return line, anomaly_points

        ani = FuncAnimation(fig, update, frames=len(data_stream), init_func=init, blit=True, interval=100)
        plt.show()

    except Exception as e:
        print(f"Error during visualization: {e}")

# Function to run the anomaly detection and visualization
def run_anomaly_detection(n=100, anomaly_chance=0.05, alpha=0.1, threshold_factor=3):
    try:
        # Validate input parameters
        if n <= 0:
            raise ValueError("The number of data points 'n' must be greater than zero.")
        if not (0 < anomaly_chance <= 1):
            raise ValueError("Anomaly chance must be between 0 and 1.")
        if not (0 < alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1.")
        if threshold_factor <= 0:
            raise ValueError("Threshold factor must be greater than zero.")

        # Simulate data stream
        data_stream = simulate_data_stream(n, anomaly_chance)

        if not data_stream:
            raise ValueError("Data stream generation failed.")

        # Initialize anomaly detector
        detector = AnomalyDetector(alpha=alpha, threshold_factor=threshold_factor)

        # Detect anomalies in the data stream
        anomalies = detector.detect_anomalies(data_stream)

        # Visualize the data stream and anomalies
        visualize_real_time(data_stream, anomalies)

    except ValueError as ve:
        print(f"Input validation error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run the anomaly detection with appropriate parameters
run_anomaly_detection(n=100, anomaly_chance=0.05, alpha=0.1, threshold_factor=3)
