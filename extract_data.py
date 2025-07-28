import nbformat
import pandas as pd
import ast

# Load the notebook
with open("political_sentiments.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Loop through notebook cells to find DataFrame definition
df_found = False
for cell in nb.cells:
    if cell.cell_type == 'code' and 'DataFrame' in cell.source:
        try:
            # Very basic: look for literal dict-based DataFrame
            exec(cell.source, globals())
            if 'df' in globals():
                df = globals()['df']
                df.to_csv("data.csv", index=False)
                print("✅ DataFrame saved as data.csv")
                df_found = True
                break
        except Exception as e:
            continue

if not df_found:
    print("❌ Couldn't extract any DataFrame.")
