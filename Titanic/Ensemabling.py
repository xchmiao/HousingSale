import pandas as pd
from scipy.stats.mstats import mode
import os
import numpy as np

path = r"E:\Kaggle\Titanic\Data\Result"
os.chdir(path)
filelist = [fp for fp in os.listdir(path)]
df = pd.DataFrame()

for fp in filelist:
    df1 = pd.read_csv(fp)
    if filelist.index(fp) == 0:
        df = df1
    else:
        df = pd.merge(df, df1, on = "PassengerId", how = "inner")
    print df.head()

df["Sum"] = 0
df["Sum"] = df.iloc[:, 1:5].mean(1)
f = lambda x: 1 if x > 0.5 else 0
df["Result"] = df["Sum"].apply(f)

df = df[["PassengerId", "Result"]]
df = df.rename(columns = {"Result": "Survived"})

df.to_csv("Submission.csv", index = False)
