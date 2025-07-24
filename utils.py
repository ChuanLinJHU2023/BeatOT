def print_format_string(text, total_length):
    text_length = len(text)
    if text_length > total_length:
        print(text[:total_length])
        return

    # Calculate the number of "#" symbols on each side
    side_length = (total_length - text_length) // 2

    # Handle odd total length when the difference isn't even
    left_hashes = "#" * side_length
    right_hashes = "#" * (total_length - text_length - side_length)

    # Construct and print the final string
    result = left_hashes + text + right_hashes
    print(result)


def get_variable_names_and_values(*variables, separator='; '):
    import inspect
    frame = inspect.currentframe().f_back
    result_parts = []

    for var in variables:
        for name, val in frame.f_locals.items():
            if val is var:
                result_parts.append(f"{name} = {repr(val)}")
                break

    return separator.join(result_parts)


def get_timestamp_filename(just_day=False):
    import datetime
    if just_day:
        return datetime.datetime.now().strftime("%Y%m%d")
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


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
