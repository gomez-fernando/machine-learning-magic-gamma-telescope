import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("hola")

cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic04.data", names=cols)
print(df.head(5))

print(df['class'].unique())

df["class"] = (df["class"] == "g").astype(int)

print(df['class'])
print(df['class'].unique())