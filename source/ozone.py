import pandas as pd
from scipy import stats

df=pd.read_csv("d:/data/ozone/ozone.csv")
print(df.head())

df2=df.dropna(axis=0)
print(df2.head())

x2=df2["Temp"].values
y2=df2["Ozone"].values

result=stats.linregress(x2,y2)
print(result)

slope, intercept, r_value, p_value, stderr=stats.linregress(x2,y2)

print(80*slope+intercept)