#!/usr/bin/env python3
"""
Generate LaTeX table files for all benchmarks using create_leaderboard function
This avoids Unicode issues by programmatically creating clean LaTeX tables
"""

import os
import sys
import pandas as pd

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db import TracePreprocessor
from utils.viz import create_leaderboard

def sanitize_latex(text):
    """Clean text for LaTeX compatibility"""
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    
    # Replace problematic characters
    replacements = {
        '✓': 'Yes',
        '×': 'No', 
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '^': r'\^{}',
        '~': r'\textasciitilde{}',
        '\\': r'\textbackslash{}',
        '{': r'\{',
        '}': r'\}',
        '€': r'\euro{}',
        '£': r'\pounds{}',
        '°': r'$^\circ$',
        'τ': r'$\tau$',
        '→': r'$\rightarrow$',
        '←': r'$\leftarrow$',
        '↑': r'$\uparrow$',
        '↓': r'$\downarrow$',
        '≤': r'$\leq$',
        '≥': r'$\geq$',
        '≠': r'$\neq$',
        '±': r'$\pm$',
    }
    
    # Remove zero-width characters
    text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    return text.strip()

def format_percentage(value):
    """Format percentage values for LaTeX"""
    if pd.isna(value) or value is None:
        return ""
    try:
        return f"{float(value):.1f}\\%"
    except (ValueError, TypeError):
        return sanitize_latex(str(value))

def format_cost(value):
    """Format cost values for LaTeX"""
    if pd.isna(value) or value is None:
        return ""
    try:
        return f"\\${float(value):.2f}"
    except (ValueError, TypeError):
        return sanitize_latex(str(value))

def format_pareto(value):
    """Format Pareto column for LaTeX"""
    if pd.isna(value) or value is None:
        return ""
    if str(value).lower() in ['true', '1', 'yes']:
        return 'Yes'
    return ""

def create_latex_table(leaderboard_df, benchmark_name):
    """Create LaTeX table content from leaderboard DataFrame"""
    
    # Define column mappings and formatters - only the specified columns
    column_configs = {
        'Agent Name': {'header': 'Scaffold', 'formatter': sanitize_latex},
        'Models': {'header': 'Model', 'formatter': sanitize_latex},
        'Accuracy': {'header': 'Accuracy', 'formatter': format_percentage},
        'Total Cost': {'header': 'Cost (USD)', 'formatter': format_cost},
        'Is Pareto': {'header': 'Pareto Optimal', 'formatter': format_pareto},
    }
    
    # Select columns in the desired order (only these columns)
    desired_columns = ['Agent Name', 'Models', 'Accuracy', 'Total Cost', 'Is Pareto']
    available_columns = [col for col in desired_columns if col in leaderboard_df.columns]
    
    # Use all available data - no artificial limit
    df_subset = leaderboard_df.copy()
    
    # Build table header
    headers = [column_configs[col]['header'] for col in available_columns]
    header_line = " & ".join(headers) + " \\\\"
    
    # Build table rows
    rows = []
    for _, row in df_subset.iterrows():
        row_data = []
        for col in available_columns:
            formatter = column_configs[col]['formatter']
            formatted_value = formatter(row[col])
            row_data.append(formatted_value)
        
        row_line = " & ".join(row_data) + " \\\\"
        rows.append(row_line)
    
    # Generate column specification - optimized for the 5 columns
    colspec = "lcccc"  # Left-align Scaffold, center the rest (Model, Accuracy, Cost, Pareto)
    
    # Combine into full table
    table_content = f"""\\begin{{tabular}}{{{colspec}}}
\\toprule
{header_line}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}"""
    
    return table_content

def generate_all_tables():
    """Generate table files for all benchmarks"""
    
    # Create tables directory
    tables_dir = "/workspaces/hal-frontend/latex_sections/tables"
    os.makedirs(tables_dir, exist_ok=True)
    
    preprocessor = TracePreprocessor()
    
    # Benchmark definitions
    benchmarks = [
        ('assistantbench', 'AssistantBench'),
        ('gaia', 'GAIA'),
        ('corebench_hard', 'CORE-Bench'),
        ('swebench_verified_mini', 'SWE-bench Verified Mini'),
        ('online_mind2web', 'Online Mind2Web'), 
        ('scicode', 'SciCode'),
        ('scienceagentbench', 'ScienceAgentBench'),
        ('taubench_airline', 'TAU-bench Airline'),
        ('usaco', 'USACO')
    ]
    
    for benchmark_key, benchmark_display in benchmarks:
        print(f"Generating table for {benchmark_display}...")
        
        try:
            # Get data and create leaderboard
            results_df = preprocessor.get_parsed_results_with_costs(benchmark_key)
            if results_df.empty:
                print(f"  ⚠️ No data available for {benchmark_key}")
                continue
                
            leaderboard_df = create_leaderboard(results_df, benchmark_name=benchmark_key)
            
            # Generate LaTeX table
            table_content = create_latex_table(leaderboard_df, benchmark_display)
            
            # Write to file
            table_filename = f"{benchmark_key}_full_results.tex"
            table_path = os.path.join(tables_dir, table_filename)
            
            with open(table_path, 'w', encoding='utf-8') as f:
                f.write(table_content)
            
            print(f"  ✓ Created {table_filename}")
            
        except Exception as e:
            print(f"  ✗ Error generating {benchmark_key}: {e}")
    
    print(f"\n✅ Table generation complete! All files saved to {tables_dir}")
    print(f"📁 Generated {len([f for f in os.listdir(tables_dir) if f.endswith('.tex')])} table files")

if __name__ == "__main__":
    generate_all_tables()