# app.py
import pandas as pd
import numpy as np
import ollama
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.set_loglevel('critical')
import seaborn as sns
from flask import Flask, render_template, request, jsonify
import os
import tempfile
import traceback
import logging
import time
import re
import warnings
import subprocess

app = Flask(__name__)
# logging.basicConfig(level=logging.DEBUG)
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

def save_uploaded_file(file):
    try:
        _, temp_path = tempfile.mkstemp(suffix='.csv')
        file.save(temp_path)
        return temp_path
    except Exception as e:
        logging.error(f"Error saving uploaded file: {str(e)}")
        raise

def analyze_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"CSV file read successfully. Shape: {df.shape}")
        column_info = [(col, str(df[col].dtype)) for col in df.columns]
        return df, column_info
    except Exception as e:
        logging.error(f"Error analyzing CSV: {str(e)}")
        raise

def preprocess_code(code):
    # Remove IPython imports
    code = re.sub(r'from IPython\.display import .*', '', code)
    code = re.sub(r'import IPython.*', '', code)
    
    # Replace display() with print()
    code = re.sub(r'display\((.*?)\)', r'print(\1)', code)
    
    # Replace HTML() with a pass statement
    code = re.sub(r'HTML\((.*?)\)', 'pass', code)
    
    return code

def format_output(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_html(classes='dataframe', index=True)
    elif isinstance(obj, pd.Series):
        return obj.to_frame().to_html(classes='dataframe', index=True)
    elif isinstance(obj, str):
        # Check if the string is a pandas Series representation
        if 'Name: ' in obj and 'dtype: ' in obj:
            # Convert the string back to a Series and then to HTML
            try:
                series = pd.read_csv(io.StringIO(obj), sep='\s+', index_col=0, squeeze=True)
                return series.to_frame().to_html(classes='dataframe', index=True)
            except:
                pass
        return f"<pre>{obj}</pre>"
    else:
        return str(obj)

def execute_llm_code(code, df):
    code = code.strip().strip('`')
    if code.startswith('python'):
        code = code[6:].lstrip()
    
    # Preprocess the code to remove IPython dependencies
    code = preprocess_code(code)
    
    output = io.StringIO()
    figures = []
    
    def custom_print(*args, **kwargs):
        for arg in args:
            formatted_output = format_output(arg)
            print(formatted_output, file=output)
        if 'end' in kwargs:
            print(kwargs['end'], file=output)

    def custom_savefig(*args, **kwargs):
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        figures.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
        plt.close()
        logging.debug(f"Figure captured. Total figures: {len(figures)}")
    
    # Replace plt.show() with custom_savefig
    code = code.replace('plt.show()', 'custom_savefig()')
    
    logging.info(f"Executing LLM-generated code. DataFrame shape: {df.shape}")
    
    try:
        exec(code, {
            'pd': pd, 
            'np': np,
            'plt': plt, 
            'sns': sns, 
            'print': custom_print,
            'custom_savefig': custom_savefig,
            'df': df  # Pass the full dataframe to the execution environment
        })
    except Exception as e:
        logging.error(f"Error executing LLM-generated code: {str(e)}")
        logging.error(f"Preprocessed LLM-generated code:\n{code}")
        print(f"Error executing LLM-generated code: {str(e)}", file=output)
        print(f"Traceback:\n{traceback.format_exc()}", file=output)
    
    # If no figures were generated, create default plots
    if not figures:
        logging.warning("No figures were generated. Creating default plots.")
        generate_default_plots(df)
    
    return output.getvalue(), figures

def generate_default_plots(df):
    # Generate a histogram for the first numeric column
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_columns) > 0:
        plt.figure(figsize=(10, 6))
        df[numeric_columns[0]].hist()
        plt.title(f"Histogram of {numeric_columns[0]}")
        plt.xlabel(numeric_columns[0])
        plt.ylabel("Frequency")
        custom_savefig()

    # Generate a scatter plot for the first two numeric columns
    if len(numeric_columns) > 1:
        plt.figure(figsize=(10, 6))
        plt.scatter(df[numeric_columns[0]], df[numeric_columns[1]])
        plt.title(f"Scatter Plot: {numeric_columns[0]} vs {numeric_columns[1]}")
        plt.xlabel(numeric_columns[0])
        plt.ylabel(numeric_columns[1])
        custom_savefig()

    # Generate a box plot for a numeric column grouped by a categorical column
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(numeric_columns) > 0 and len(categorical_columns) > 0:
        plt.figure(figsize=(10, 6))
        df.boxplot(column=numeric_columns[0], by=categorical_columns[0])
        plt.title(f"Box Plot: {numeric_columns[0]} by {categorical_columns[0]}")
        plt.suptitle("")  # Remove automatic suptitle
        custom_savefig()

def generate_ollama_prompt(df, column_info, file_path):
    try:
        file_path = rf"{file_path}"
        prompt = f"""Analyze the CSV file and generate Python code to provide summary statistics and visualizations.

DataFrame Information:
{df.info()}

Column Types:
{column_info}

Sample Data (first 5 rows):
{df.head().to_string()}

Full Dataset Shape: {df.shape}

Please generate Python code that does the following:
1. The full DataFrame is already loaded as 'df'. Do not load the CSV file again.
2. If any column has values Yes / No then replace Yes with 1 and No with 0 before doing anything else.
3. Calculate basic summary statistics for all numerical columns using df.describe() and print the result.
4. For categorical columns, show value counts and percentages using value_counts(normalize=True) and print the results.
   Use the following format for each categorical column:
   print("Value Counts and Percentages for [Column Name]:")
   print(df['column_name'].value_counts(normalize=True))
   print(df['column_name'].value_counts(normalize=True) * 100)
5. Create at least 8 relevant visualizations (e.g., histograms, scatter plots, box plots, bar charts) based on the full dataset.
6. Provide a brief textual analysis of each visualization using print statements.

Use pandas (pd), numpy (np), matplotlib.pyplot (plt), and seaborn (sns) libraries. 
The code should be self-contained and ready to run.
Do not use IPython or any IPython-specific functions like display() or HTML().
Use print() for all output.
For each visualization:
    1. Call plt.figure(figsize=(10, 6)) to create a new figure
    2. Create the plot (e.g., plt.hist(), sns.scatterplot(), etc.) using the full dataset
    3. Add appropriate labels and title
    4. Call plt.show() to display the figure

Do not use subplots or plt.subplots(). Create each plot separately.
Do not save any files to disk.

Return only the Python code without any additional explanation or markdown formatting.
"""
        logging.info(f"Generated prompt for Ollama. DataFrame shape: {df.shape}")
        return prompt
    except Exception as e:
        logging.error(f"Error generating Ollama prompt: {str(e)}")
        raise

@app.route('/analyze', methods=['POST'])
def analyze():
    file_path = None
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Get the selected model and custom prompt
        model = request.form.get('modelSelect', 'llama3.1')
        custom_prompt = request.form.get('prompt', '')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            file_path = temp_file.name
            file.save(file_path)
        
        df, column_info = analyze_csv(file_path)
        
        # Use custom prompt if provided, otherwise generate the default prompt
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = generate_ollama_prompt(df, column_info, file_path)
        
        logging.debug(f"Model used in Ollama:\n{model}")
        logging.debug(f"Sending prompt to Ollama:\n{prompt}")
        response = ollama.generate(model=model, prompt=prompt)
        logging.debug(f"Received response from Ollama:\n{response['response']}")
        
        output, figures = execute_llm_code(response['response'], df)
        
        logging.debug(f"Number of figures generated: {len(figures)}")
        
        return jsonify({
            'output': output,
            'figures': figures
        })
    except Exception as e:
        logging.error(f"Error in analyze route: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logging.error(f"Error removing temporary file: {str(e)}")

@app.route('/get_models', methods=['GET'])
def get_models():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        models = [line.split()[0] for line in result.stdout.strip().split('\n')[1:]]
        return jsonify(models)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)