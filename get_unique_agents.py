import sqlite3
from pathlib import Path
import pandas as pd

def get_unique_agents():
    """Get all unique agent names from all database files"""
    db_dir = Path('preprocessed_traces')
    
    if not db_dir.exists():
        print(f"Directory {db_dir} does not exist!")
        return
    
    all_agents = set()
    benchmark_agent_counts = {}
    
    # Get all .db files
    db_files = list(db_dir.glob('*.db'))
    
    if not db_files:
        print("No database files found!")
        return
    
    print(f"Found {len(db_files)} database files")
    
    for db_file in db_files:
        benchmark_name = db_file.stem
        
        try:
            # Connect to database
            conn = sqlite3.connect(db_file)
            
            # Get unique agent names from parsed_results table
            query = "SELECT DISTINCT agent_name FROM parsed_results"
            df = pd.read_sql_query(query, conn)
            
            if not df.empty:
                agents_in_benchmark = set(df['agent_name'].tolist())
                all_agents.update(agents_in_benchmark)
                benchmark_agent_counts[benchmark_name] = len(agents_in_benchmark)
                
                print(f"{benchmark_name}: {len(agents_in_benchmark)} unique agents")
            else:
                print(f"{benchmark_name}: No agents found")
            
            conn.close()
            
        except Exception as e:
            print(f"Error processing {db_file}: {e}")
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Total unique agent names across all benchmarks: {len(all_agents)}")
    print(f"Total benchmarks: {len(benchmark_agent_counts)}")
    
    # Print all unique agent names sorted
    print(f"\n=== ALL UNIQUE AGENT NAMES ===")
    sorted_agents = sorted(all_agents)
    for i, agent in enumerate(sorted_agents, 1):
        print(f"{i:3d}. {agent}")
    
    # Save to file
    output_file = Path('unique_agents.txt')
    with open(output_file, 'w') as f:
        f.write("All unique agent names from databases:\n")
        f.write("=" * 50 + "\n\n")
        for agent in sorted_agents:
            f.write(f"{agent}\n")
    
    print(f"\nAgent names saved to: {output_file}")
    
    return sorted_agents

if __name__ == "__main__":
    get_unique_agents()
