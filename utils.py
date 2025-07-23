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

