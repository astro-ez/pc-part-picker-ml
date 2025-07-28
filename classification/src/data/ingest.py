import pandas as pd
import json
from .parts_attrs_dict import parts_attrs_dict
from .parse_utils import parse_attribute

def ingest_data():

    # Start creating the raw dataset
    df = pd.DataFrame(columns=['part_name', 'part_type'])

    for key in parts_attrs_dict.keys():
        read_file = f"../shared/data/raw/pc-parts-json/{key}.json"
        try:
            # Try first with lines=True for JSONL format
            pdf_read = pd.read_json(read_file, orient='records', lines=True)
        except ValueError:
            # If that fails, try standard JSON format
            pdf_read = pd.read_json(read_file, orient='records')
        
        new_attributes = parts_attrs_dict[key].keys()
        attrs_values = {}

        # pdf_read = pdf_read.loc[:1, :]

        for attr in new_attributes:
            attrs_values[attr] = pd.Series(dtype=str)

            if attr not in df.columns:
                raise ValueError(f"Attribute '{attr}' not found in the DataFrame for key '{key}'")
            
            # Handle both dictionary values with 'columns' and direct string values
            if isinstance(parts_attrs_dict[key][attr], dict):
                columns = parts_attrs_dict[key][attr].get("columns", {})
            else:
                parsed_values = pd.Series([parts_attrs_dict[key][attr]] * len(pdf_read), dtype=str, name=attr)
                attrs_values[attr] = parsed_values.to_list()
                continue

            for col, rules in columns.items():
                if col not in pdf_read.columns:
                    raise ValueError(f"Column '{col}' not found in the DataFrame for key '{key}'")
                    
                # Parse each attribute according to its rules
                parsed_values = pdf_read[col].apply(lambda x: parse_attribute(x, rules))
                # concat the parsed values to the attrs_value of the current attribute
                if attrs_values[attr].empty:
                    attrs_values[attr] = parsed_values
                else:
                    # Handle NaN values properly when concatenating
                    if parsed_values.isna().any():
                        # Only concatenate where parsed_values is not NaN
                        mask = ~parsed_values.isna()
                        # For rows where parsed_values is not NaN, concatenate
                        attrs_values[attr].loc[mask] = attrs_values[attr].loc[mask].str.cat(parsed_values.loc[mask], sep=' ')
                    else:
                        # Simple concatenation when no NaN values are present
                        attrs_values[attr] = attrs_values[attr].str.cat(parsed_values, sep=' ')
                        
        attr_df = pd.DataFrame(attrs_values) 
        df = pd.concat([df, attr_df], ignore_index=True)

    # Save the final DataFrame to a CSV File
    df.to_csv("../shared/data/interim/pc-parts-merged.csv", index=False)


def main():
    ingest_data()
    print("Data ingestion completed successfully.")

if __name__ == "__main__":
    main()