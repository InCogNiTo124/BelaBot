#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sys
#import plotly
#import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

plt.rcParams['backend'] = 'qt5agg'


y = np.array(list(map(float, sys.stdin)))
x = np.arange(len(y), dtype=np.float32)

linreg = LinearRegression()
linreg.fit(x.reshape(-1, 1), y.reshape(-1, 1))
y_pred = linreg.predict(x.reshape(-1, 1))
print(linreg.score(x.reshape(-1, 1), y.reshape(-1, 1)))
print(linreg.coef_, linreg.intercept_)
plt.scatter(x, y, s=5)
plt.plot(x.ravel(), y_pred.ravel(), 'g-')
plt.legend()
df = pd.DataFrame(y)
df['ewm'] = df[0].ewm(20).mean()
plt.plot(df['ewm'].values, 'r-')
plt.show()
