import pandas as pd

def parse_attribute(value, rules: dict):
    """
    Parse a single attribute according to the provided rules.

    Supported rules:
        - 'prefix': str
        - 'suffix': str
        - 'join': str delimiter if value is a list
        - 'upper': bool
        - 'lower': bool
        - 'bool_format': tuple (true_str, false_str)
        - 'skip_none': bool
        - 'none_value': str or fallback value for None
    """
    # Handle None values early
    if value is None:
        if rules.get("skip_none", True):
            return rules.get("none_value", "")
        return None

    # Handle booleans
    if (isinstance(value, bool) and "bool_format" in rules ) or "bool_format" in rules:
        true_str, false_str = rules["bool_format"]
        value = true_str if value else false_str

    # Handle array joining
    if isinstance(value, (list, tuple)) and "join" in rules:
        value = rules.get("join", ",").join(map(str, value))

    # Handle numeric values
    if isinstance(value, (int, float)) and not (value is None or pd.isna(value)):
        if rules.get('int', True):
            value = int(value)
        else:
            value = float(value)

    # Convert to string for further operations
    value = str(value)

    # Casing
    if rules.get("upper"):
        value = value.upper()
    if rules.get("lower"):
        value = value.lower()

    # Add prefix and suffix
    prefix = rules.get("prefix", "")
    suffix = rules.get("suffix", "")
    value = f"{prefix}{value}{suffix}"

    return value