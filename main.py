import pandas as pd
import ollama
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import sys
from contextlib import redirect_stdout

def analyze_csv(file_path):
    print(f"Reading CSV file: {file_path}")
    df = pd.read_csv(file_path)
    print(f"CSV file read successfully. Shape: {df.shape}")
    return df

def generate_ollama_prompt(df):
    print("\nGenerating prompt for Ollama...")
    prompt = f"""Analyze the following DataFrame and generate Python code to provide summary statistics and visualizations.

DataFrame Information:
{df.info()}

Sample Data:
{df.head().to_string()}

Please generate Python code that does the following:
1. Calculate and print basic summary statistics for all numerical columns.
2. For categorical columns, show value counts and percentages.
3. Create at least 3 relevant visualizations (e.g., histograms, scatter plots, box plots, bar charts) based on the data.
4. Provide a brief textual analysis of each visualization.

Use pandas, matplotlib, and seaborn libraries. Assume the DataFrame is already loaded as 'df'.
The code should be self-contained and ready to run.
Load the data from {file_path}

Return only the Python code without any additional explanation or markdown formatting.
"""
    print("Prompt generated successfully.")
    return prompt

def execute_llm_code(code, df):
    print("\nExecuting LLM-generated code...")
    
    # Remove any leading/trailing whitespace and backticks
    code = code.strip().strip('`')
    
    # If the code starts with "python", remove it
    if code.startswith('python'):
        code = code[6:].lstrip()
    
    # Create a string buffer to capture print output
    buffer = io.StringIO()
    
    # Redirect stdout to the buffer
    with redirect_stdout(buffer):
        try:
            # Execute the code in a restricted environment
            exec(code, {'df': df, 'pd': pd, 'plt': plt, 'sns': sns, 'print': print})
        except Exception as e:
            print(f"Error executing LLM-generated code: {str(e)}")
    
    # Get the captured output
    output = buffer.getvalue()
    
    print("Code execution completed.")
    return output

def main(file_path):
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    df = analyze_csv(file_path)
    prompt = generate_ollama_prompt(df)

    print("\nSending prompt to Ollama...")
    response = ollama.generate(model="llama3.1", prompt=prompt)
    print("Received response from Ollama.")
    
    print("\nOllama's response (Python code):")
    print(response['response'])

    # Execute the LLM-generated code
    output = execute_llm_code(response['response'], df)

    print("\nOutput from LLM-generated code:")
    print(output)

    print("\nAnalysis complete. Check the current directory for any generated visualization files.")

if __name__ == "__main__":
    file_path = r"C:\Users\Defualt\Downloads\insurance_data.csv"
    main(file_path)