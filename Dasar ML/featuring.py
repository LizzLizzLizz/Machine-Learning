import pandas as pd

data = pd.DataFrame()

data['bin'] = pd.cut(data['value'], bins=[0, 30, 70, 100], labels=["Low", "Mid", "High"])
