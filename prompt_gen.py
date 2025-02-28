
import re
import json
import random

def kb_gen_prompt(attr_name, dataset_name, idx_list, dirty_csv, attr_analy_content):
    prompt = f'You are a top data scientist, especially in data cleaning. Please generate a comprehensive and expert guide for identifying and analyzing common errors in the \'{attr_name}\' attribute of the \'{dataset_name}\' table:'
    if len(attr_analy_content) > 0:
        prompt += f"\n\nHere are the data distribution analysis results for the attribute \'{attr_name}\':"
        prompt += attr_analy_content
    prompt += f'\n\nHere are some examples for \'{attr_name}\' along with strong correlated attribute values:\n'
    
    max_example_num = 20
    idx_list = idx_list[:max_example_num] if len(idx_list) > max_example_num else idx_list
    random.shuffle(idx_list)
    example_vals = []
    for idx in idx_list:
        example_vals.append(str(dirty_csv.loc[int(idx), :].to_dict()))
    example_vals_str = '\n'.join(example_vals)
    prompt += example_vals_str
    
    prompt += f'\n\nPlease first explain the meaning of attribute \'{attr_name}\'.\n'
    prompt += f'\n\nThen, for each error type below, considering the data distribution analysis results, provide specific causes, examples, and detection methods for \'{attr_name}\':\n'
    prompt += '1. Pattern Violations: Expected formats and non-conforming value identification.\n'
    prompt += '2. Missing Values: Explicit null indicators and implicit missing data patterns.\n'
    prompt += f'3. Constraint Violations: Relationships of \'{attr_name}\' with other attributes and how to verify them.\n'
    prompt += '4. Out-of-domain Values: Valid value range/set and possible outliers.\n'
    prompt += '5. Typos: Common misspellings or data entry errors and detection strategies.\n'
    prompt += '6. Common Knowledge Violations: Expected rules/facts and methods to identify contradictions.\n\n'
    prompt += f'Please also generate some possible errors for the \'{attr_name}\' attribute data based on the above error types. '
    prompt += '\n\nIMPORTANT NOTE: When analyzing potential errors, if you are not completely certain that a value is wrong, please respect the mainstream data distribution patterns. Some values that appear unusual may actually be valid according to local requirements or domain-specific conventions. Only flag values as errors when you have high confidence they violate clear patterns or rules.'
    return prompt


def error_check_prompt(col_values, col_name):
    lines = col_values.strip().split('\n')
    try:
        col_list = re.findall(r'"([^"]+)"\s*:', lines[0])
    except json.JSONDecodeError as e:
       print(f"JSON Decode Error: {e}")
       print(f"Problematic JSON string: {lines[0]}")

    template_dict_1 = {key: f'{key}_example_val_1' for key in col_list}
    template_dict_2 = {key: f'{key}_example_val_2' for key in col_list}
    
    prompt = ""
    prompt += f"As a data quality expert, please first analyze attribute relations and analyze the '{col_name}' attribute values for potential errors. Ignore case sensitivity\n"
    prompt += f"Provide your analysis on `{col_name}` values in JSON format as follows, **do not care problems in other attributes**:\n\n"
    prompt += '''
```json
{'''
    prompt += f'''"column_name": "{col_name}",'''
    prompt += '''
  "entries": [
    {'''
    prompt += f'''\n"value_row": "{template_dict_1}",'''
    prompt += f'''\n"error_analysis": "[Brief explanation of the error analysis, if applicable]",'''
    prompt += f'''\n"has_error_in_{col_name}_value": true/false,'''
    prompt += '''
    },
    {'''
    prompt += f'''\n"value_row": "{template_dict_2}",'''
    prompt += f'''\n"error_analysis": "[Brief explanation of the error analysis, if applicable]",'''
    prompt += f'''\n"has_error_in_{col_name}_value": true/false,'''
    prompt += '''
    }
  ]
}
```
\n\n'''
    prompt += "If unsure, do not indicate an error.\n"
    prompt += "- Please ignore the case sensitivity issues.\n\n"
    prompt += "-----------------------------------------------\n\n"
    prompt += "Here are the given inputs:\n"
    prompt += f"Values of column '{col_name}' along with related attribute values:\n"
    prompt += f"'{col_values}'\n"
    prompt += f"Provide your analysis on `{col_name}` values in the required JSON format, **do not care problems in other attributes**:\n"
    return prompt


def create_err_gen_inst_prompt(clean_vals, dirty_vals, target_attribute, num_errors=20):
    if len(clean_vals) > 0:
        temp_vals = clean_vals[0]
    elif len(dirty_vals) > 0:
        temp_vals = dirty_vals[0]
    else:
        print(f"No vals in clean_vals and dirty_vals of attr {target_attribute}")
        temp_vals = f"{target_attribute}: none"
    attrs = re.findall(r"'(\w+)':", str(temp_vals))
    template_dict_1 = {key: f'{key}_val_1' for key in attrs}
    template_dict_1[target_attribute] = 'error_value_1'
    template_dict_2 = {key: f'{key}_val_2' for key in attrs}
    template_dict_2[target_attribute] = 'error_value_2'
    
    prompt = f"""
You are a data quality analyst with extensive experience in identifying and generating realistic data errors. Your task is to analyze a given dataset and generate plausible errors for a specific attribute, simulating real-world data quality issues.

I will provide you with a sample of **possible** clean and dirty values in a tabular format for various attributes. Your objectives are to:

1. Analyze the data to identify patterns, relationships, and constraints between attributes.
2. Focus on the attribute named `{target_attribute}` and generate realistic errors that could occur in real-world scenarios.
3. Ensure the errors you generate are diverse and cover multiple error types.

Your task is to analyze the data and identify inner relationships. Based on this analysis, generate errors specifically for the attribute `attribute_name` as they might occur in real-world scenarios. 
The types of errors include the following ones
1. Pattern Violations: Values that don't match the expected format
2. Explicit/Implicit Missing Values: Null values or placeholders for missing data
3. Constraints Violations: Values that conflict with other columns or violate business rules
4. Out-of-domain values: Values outside the expected range or set
5. Typos: Spelling or data entry errors
6. Violate common knowledge: Values that contradict widely known facts
"""
    prompt += f"For the attribute `{target_attribute}`, here are the given **possible** clean tuples:\n"
    prompt += '\n'.join([str(i) for i in clean_vals]) + '\n'
    prompt += f"There are also some **possible** wrong tuples for reference:\n"
    prompt += '\n'.join([str(i) for i in dirty_vals]) + '\n\n'
    prompt += f"Please analyze the error pattern and generate {num_errors} realistic errors specifically for the attribute `{target_attribute}`:\n"
    prompt += f"""
The output should be in the following strict format:
['{target_attribute}', error_value_1, Reason: 'Error type1: Specific reason', {str(template_dict_1)}]
['{target_attribute}', error_value_2, Reason: 'Error type2: Specific reason', {str(template_dict_2)}]
...
Please ensure that the reasons for each error are clearly specified.
Do not be the same as the reference values.
--------------------------------------------------------------------------
"""
    return prompt


def pre_func_prompt(attr_name, data_example):
    prompt = (
        f"You are a Data Cleaning Specialist tasked with distinguishing between clean and dirty cells in the `{attr_name}`.\n\n"
        
        f"Here are examples for the '{attr_name}' column:\n"
        f"{data_example}\n\n"

        "Your task:\n"
        f"1. Analyze the `{attr_name}` column values.\n"
        "2. Create precise judge functions in Python that:\n"
        f"- Receive the row content of the `{attr_name}` column\n"
        "- Return True for clean values, False for dirty values\n"
        "- Use the naming convention 'is_clean_[judgment]'\n"
        "- Cover different perspectives of cleanliness as possible\n"
        "- Do not contain blank lines inner functions\n\n"

"Example function code snippet:\n"
"```python "
f"def is_clean_[judgment](row, attr):\n"
f"    # Value of `{attr_name}` is row[attr]\n"
"    # Your logic here\n"
"    return True  # or False\n"
"```\n"
"Provide your functions below:\n"
    )
    return prompt


def err_clean_func_prompt(attr_name, clean_info, errs_info):
    prompt = (
        f"You are a Data Cleaning Specialist tasked with identifying and distinguishing between clean and dirty cells in the `{attr_name}` column.\n\n"
        f"Clean examples for the '{attr_name}' column:\n"
        f"{clean_info}\n\n"
        f"Error examples for the '{attr_name}' column:\n"
        f"{errs_info}\n\n"

        "Your task:\n"
        f"1. Analyze the `{attr_name}` column values.\n"
        "2. Compare the differences between clean and dirty values.\n"
        "3. Create precise judge functions in Python that:\n"
        f"- Receive the row content of the `{attr_name}` column\n"
        "- Return True for clean values, False for dirty values\n"
        "- Use the naming convention 'is_clean_[judgment]'\n"
        "- Cover different perspectives of cleanliness as possible\n"
        "- Do not contain blank lines inner functions\n\n"

"Example function code snippet:\n"
"```python "
f"def is_clean_[judgment](row, attr):\n"
f"    # Value of `{attr_name}` is row[attr]\n"
"    # Your logic here\n"
"    return True  # or False\n"
"```\n"
"Provide your functions below:\n"
    )
    return prompt