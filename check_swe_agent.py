import os
import json
from utils.db import TracePreprocessor

def check_swe_agent_files():
    """Check both database and JSON files for SWE-Agent variants"""
    print("=== CHECKING DATABASE ===")
    
    preprocessor = TracePreprocessor()
    benchmark_name = 'swebench_verified_mini'
    
    try:
        with preprocessor.get_conn(benchmark_name) as conn:
            # Check for SWE-Agent entries in database
            query = '''
                SELECT agent_name, model_name, accuracy, total_cost, run_id
                FROM parsed_results 
                WHERE benchmark_name = ? 
                AND (agent_name LIKE '%SWE%' OR agent_name LIKE '%swe%' OR agent_name LIKE '%My Agent%')
                ORDER BY agent_name, model_name
            '''
            
            import pandas as pd
            df = pd.read_sql_query(query, conn, params=(benchmark_name,))
            
            if df.empty:
                print("No SWE-Agent entries found in database")
            else:
                print("SWE-Agent entries in database:")
                print(df.to_string(index=False))
                
    except Exception as e:
        print(f"Error checking database: {e}")
    
    print("\n=== CHECKING JSON FILES ===")
    
    # Check JSON files in evals_live directory
    evals_dir = "/workspaces/hal-frontend/evals_live"
    swe_files = []
    
    if os.path.exists(evals_dir):
        for filename in os.listdir(evals_dir):
            if filename.startswith("swebench") and filename.endswith(".json"):
                filepath = os.path.join(evals_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        # Read just enough to get the config
                        content = f.read(2000)  # Read first 2KB
                        if '"config"' in content:
                            # Parse the beginning of the file
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if '"agent_name"' in line:
                                    agent_name = line.split(':')[1].strip().strip('",')
                                    break
                            
                            # Check if this is a SWE-Agent related file
                            if any(keyword in agent_name.lower() for keyword in ['swe', 'my agent']):
                                swe_files.append((filename, agent_name))
                except Exception as e:
                    continue
    
    if swe_files:
        print("SWE-Agent related JSON files found:")
        for filename, agent_name in swe_files:
            print(f"  {filename} -> {agent_name}")
            
            # Check if it's o4-mini
            if 'o4mini' in filename or 'o4-mini' in filename:
                print(f"    *** Contains o4-mini! ***")
    else:
        print("No SWE-Agent related JSON files found")
    
    print("\n=== SUMMARY ===")
    print("You have these SWE-Agent variants with o4-mini:")
    
    o4_mini_files = [f for f in swe_files if 'o4mini' in f[0]]
    for filename, agent_name in o4_mini_files:
        if '1745279519' in filename:
            print(f"  ✓ My Agent(o4-mini-2025-04-16) LOW reasoning - {filename}")
        elif '1745345781' in filename:
            print(f"  ✓ My Agent(o4-mini-2025-04-16) HIGH reasoning - {filename}")
        else:
            print(f"  ✓ {agent_name} - {filename}")

if __name__ == "__main__":
    check_swe_agent_files()
