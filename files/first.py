import numpy as np
import pandas as pd

df = pd.read_csv('../csv/orders')

df['OrderDate'] = pd.to_datetime(df['OrderDate'])

df['TotalAmount'] = df['Quantity'] * df['price']

print(df['TotalAmount'].sum())

df['TotalAmount'].mean()

filtered = df[df['price'] > 500]
print(filtered)

df = df.sort_values(by='OrderDate', ascending=False)

june = df[ (df['OrderDate'] >= '2023-06-05') & (df['OrderDate'] <= '2023-06-10') ]
print(june)

top3 = df.groupby('Client')["TotalAmount"].sum() \
         .sort_values(ascending=False) \
         .head(3)

print(top3)

cat = df.groupby('Category')['Quantity'].sum() 
sales = df.groupby("Category")["TotalAmount"].sum()
print(cat)
print(sales)

