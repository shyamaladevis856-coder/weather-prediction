"""Creates a sample weather dataset (dataset/weather.csv).
If you have a real dataset, just put it in dataset/weather.csv with the
columns: Temperature, Humidity, WindSpeed, Weather  (and skip this file)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1500  # rows per class

def make(label, temp, hum, wind):
    return pd.DataFrame({
        "Temperature": rng.normal(*temp, N).round(1),
        "Humidity": np.clip(rng.normal(*hum, N), 10, 100).round(0),
        "WindSpeed": np.clip(rng.normal(*wind, N), 0, 80).round(1),
        "Weather": label,
    })

df = pd.concat([
    make("Sunny",  (34, 4), (48, 12), (10, 5)),
    make("Cloudy", (28, 3), (70, 8),  (14, 5)),
    make("Rainy",  (25, 3), (86, 6),  (20, 7)),
]).sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("dataset/weather.csv", index=False)
print("Saved dataset/weather.csv", df.shape)
