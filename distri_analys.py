import pandas as pd
import numpy as np
import re
import ast
import collections
from utility import get_ans_from_llm
from typing import List, Dict
from datetime import datetime
from collections import Counter


class LLMDataDistrAnalyzer:
    def __init__(self, dirty_csv):
        self.dirty_csv = dirty_csv
        
    def get_column_examples(self, attr_name: str, n_samples: int = 10) -> str:
        """Get sample values from the specified column."""
        if attr_name not in self.dirty_csv.columns:
            return f"Error: Column {attr_name} not found"
        
        samples = self.dirty_csv.sample(n=min(n_samples, len(self.dirty_csv)))
        result_str = '\n'.join([', '.join([f"{attr}: {value}" for attr, value in row.items()]) for row in samples.to_dict('records')])
        return f"Examples of values in column '{attr_name}': \n{result_str}"

    def generate_llm_prompt(self, attr_name: str) -> str:
        """Generate prompt for LLM to create analysis functions."""
        examples = self.get_column_examples(attr_name)
        
        prompt = f"""
Based on the column '{attr_name}' with examples: {examples}

Please generate Python functions to analyze the data distribution from various perspectives, so that we can verify whether an error is reasonable or not. 
Each function should:
1. Take parameters (dirty_csv: dataframe, attr_name: str), regard all values in dirty_csv are **strings**
2. Return a string containing the **detailed** analysis results
3. Do not enumerate/count all values, showing representative ones
4. **Also import necessary libraries**
        
Example function code snippet:\n
```python 
def distr_analysis_[perspective](dirty_csv, attr_name):
    # Your logic here
    return 'Detailed description of the analysis results'
```\n
Provide your functions below:\n
        """
        return prompt, examples

    def validate_and_clean_function(self, function_code: str) -> str:
        try:
            ast.parse(function_code)
            return function_code
        except SyntaxError:
            return None

    def execute_function(self, function_code: str, attr_name: str) -> str:
        """Execute a single extracted function safely."""
        try:
            # Create namespace for function execution
            namespace = {
                'dirty_csv': self.dirty_csv,
                'attr_name': attr_name,
                'pd': pd,
                'np': np,
                'datetime': datetime,
                'Counter': Counter,
                're': re,
                'collections': collections,
            }
            
            setup_code = """
import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, List
from collections import Counter
            """
            exec(setup_code + "\n" + function_code, globals(), namespace)
            function_name = function_code.split("def ")[1].split("(")[0].strip()
            func = namespace[function_name]
            result = func(self.dirty_csv, attr_name)
            return result
            
        except Exception as e:
            return f"Error executing function: {str(e)}"

    def analyze_data(self, attr_name: str, llm_response: str, output_file: str) -> Dict:
        functions = extract_func(llm_response)
        self.functions = functions
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Analysis Results for {attr_name}\n")
            f.write("=" * 50 + "\n")
        results = ""
        for i, func_code in enumerate(functions, 1):
            clean_code = self.validate_and_clean_function(func_code)
            if clean_code:
                result = self.execute_function(clean_code, attr_name)
                if result is not None:
                    results += f"\n\n=== Data Distribution Analysis {i} ===\n"
                    if isinstance(result, str) and len(result.split('\n')) > 30:
                        result = '\n'.join(result.split('\n')[:30]) + "\n... Too long, only sample some examples."
                    results = results + '\n' + result
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n\n=== Data Distribution Analysis Function {i} ===\n")
                        f.write(clean_code)
                        f.write("\n**Running Results:**\n")
                        f.write(str(result))
                    
        return results
    
    
def extract_func(text_content):
    try:
        code_blocks = re.findall(r'```(.*?)```', text_content, re.DOTALL)
    except re.error as e:
        print(f"Regex error: {e}")
        return [], []
    func_list = []
    for code_block in code_blocks:
        functions = re.findall(r'def \w+\(.*?\):\n(?:[ \t]*\n)*(?: .*\n)+', code_block)
        for function in functions:
            try:
                function_name = re.findall(r'def (\w+)', function)[0]
            except IndexError:
                print("Function name not found in the function definition.")
                continue
            func_list.append(function)
    return func_list

