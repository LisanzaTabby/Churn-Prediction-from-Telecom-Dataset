import pandas as pd

# Task 1: Loading the Dataset
df = pd.read_csv('high_service_calls.csv')

print(df.head(10))

higher_service_calls = df.query('`Frequency of use` > 200')
print(higher_service_calls)