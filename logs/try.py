import pandas as pd
import re


def parse_line(line):
    """
    Parse a single line into a dictionary.
    Handles list values and expands them into multiple fields.
    """
    data = {}
    # Split the line into key-value pairs
    pairs = re.findall(r"(\w+)\s*=\s*([^;]+);", line)

    for key, value in pairs:
        value = value.strip()
        # Check if the value looks like a list
        if value.startswith('[') and value.endswith(']'):
            # Remove brackets
            list_str = value[1:-1].strip()
            # Find individual list items
            items = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?", list_str)
            # Convert items to float or int as appropriate
            numeric_items = [float(item) if '.' in item or 'e' in item else int(item) for item in items]
            # Expand into multiple fields
            for idx, item in enumerate(numeric_items, start=1):
                data[f"{key}{idx}"] = item
        else:
            # Convert to float or int if possible
            if re.match(r"^-?\d+\.?\d*(e[-+]?\d+)?$", value):
                if '.' in value or 'e' in value:
                    data[key] = float(value)
                else:
                    data[key] = int(value)
            else:
                # Remove quotes if present
                value = value.strip("'\"")
                data[key] = value
    return data


def read_file_to_dataframe(file_path):
    """
    Reads the text file with multiple lines, parses each line, and returns a pandas DataFrame.
    """
    records = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                record = parse_line(line)
                records.append(record)
    df = pd.DataFrame(records)
    return df


# Example usage:
# df = read_file_to_dataframe('your_file.txt')
# print(df)


df = read_file_to_dataframe('20250723.txt')
print(df)
