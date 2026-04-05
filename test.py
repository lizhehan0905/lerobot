import pandas as pd
df = pd.read_parquet("/home/hpc/VLA/data_12_17_train/meta/episodes/chunk-000/file-000.parquet")
print(df.dtypes)
print(df.columns)
print(df.head(2))
