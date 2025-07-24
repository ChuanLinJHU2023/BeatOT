from utils import *
from visualizations import *
df = read_file_to_dataframe('../../logs/20250723.txt')

# Group by 'method_name' and 'case_number'
grouped = df.groupby(['method_name', 'case_number'])

for (method, case), group in grouped:
    # Pick one save_file, e.g., the first one
    save_file = group['save_file'].iloc[0]

    # Generate the image
    visualize_a_case(
        case_number=case,
        checkpoint_file_path=f"./checkpoints/{save_file}",
        visualization_file_path=f"./images/experiment_{method}_{case}.png"
    )