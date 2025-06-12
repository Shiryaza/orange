import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, shapiro

# Load the CSV
df = pd.read_csv('colour (1).csv')

# Define target columns
columns = ['H', 'S', 'V']

for col in columns:
    print(f"\n--- Processing Column: {col} ---")

    if col not in df.columns:
        print(f"❌ Column '{col}' not found in the CSV.")
        continue

    data = df[col].dropna()

    if data.empty:
        print(f"❌ Column '{col}' has no valid (non-NaN) data.")
        continue

    if not np.issubdtype(data.dtype, np.number):
        print(f"❌ Column '{col}' contains non-numeric data. Type: {data.dtype}")
        continue

    # Fit normal distribution
    mean, std = norm.fit(data)

    # Plot
    plt.figure(figsize=(7, 4))
    sns.histplot(data, bins=30, kde=True, stat="density", label='Data', color='skyblue')

    x = np.linspace(min(data), max(data), 100)
    p = norm.pdf(x, mean, std)
    plt.plot(x, p, 'r--', label=f'Normal Fit\nμ={mean:.2f}, σ={std:.2f}')

    # Shapiro-Wilk normality test
    stat, p_val = shapiro(data)
    is_normal = 'Yes' if p_val > 0.05 else 'No'

    print(f"✔️ Column: {col}, Mean: {mean:.2f}, Std: {std:.2f}, p-value: {p_val:.4f} → Normal? {is_normal}")

    plt.title(f'{col} Distribution\nNormal? {is_normal} (p={p_val:.3f})')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()
