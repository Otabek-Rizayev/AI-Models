import numpy as np
import pandas as pd

df = pd.read_csv('data-sets/houses.csv', index_col=0)
df.head()

"""# Ustunlar ta'rifi
- `location` - sotilayotgan uy manzili
- `district` - uy joylashgan tuman
- `rooms` - xonalar soni
- `size` - uy maydoni (kv.m)
- `level` - uy
"""

df['district'].value_counts()

housing = df[df.district=='Чиланзарский']
housing.head()

X = housing['size']
y = housing['price']

theta1 = np.sum((X - np.mean(X)) * (y - np.mean(y))) / np.sum((X - np.mean(X)) ** 2)
theta0 = np.mean(y) - theta1 * np.mean(X)
print(f"Regression Model: y={theta0:.2f}, X={theta1:.2f}")

X_test = housing.sample(10, random_state=42)['size'].to_numpy()
y_test = housing.sample(10, random_state=42)['price'].to_numpy()
print(f"Asl narxi(qiymati): y={y_test.astype(int)}")
print(f"x={X_test.astype(int)}")
y_pred = theta0 + theta1 * X_test
print(f"Prediction(Bashorat): y={y_pred.astype(int)}")
