from utils import *
import pandas as pd

# Set display options to show more columns and wider width
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Read file
df = read_file_to_dataframe('../logs/20250723.txt')
print(df)
df.to_excel('logs1.xlsx', index=False)

# Assuming your DataFrame is called df
# Replace 'method_name', 'case_number', and 'accuracy' with actual column names if different
# If column names are not known, inspect df.columns

# Group by method and case
grouped = df.groupby(['method_name', 'case_number'])

# Calculate required statistics
analysis = grouped['accuracy'].agg(['max', 'mean', 'min', 'var']).reset_index()

# Rename columns for clarity
analysis.columns = ['method_name', 'case_number', 'best_accuracy', 'average_accuracy', 'worst_accuracy', 'accuracy_variance']

# Sort by case_number for comparison
analysis_sorted = analysis.sort_values(by='case_number')
print(analysis_sorted)

