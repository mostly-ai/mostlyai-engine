# %%
import pandas as pd
df = pd.read_csv("https://github.com/mostly-ai/public-demo-data/raw/dev/census/census.csv.gz")
df = df[['income']].head(1000)
df['copy'] = df['income']
df
# %%
from mostlyai.engine import TabularARGN
argn = TabularARGN(enable_flexible_generation=True)
argn.fit(df)
# %%
argn.predict_proba(df, target="income")
# %%
df
# %%
