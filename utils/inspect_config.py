import json
from pathlib import Path
import sys

def inspect_config(file_stem):
    try:
        # Look for the file in evals_live folder
        file_path = Path('evals_live') / f"{file_stem}.json"
        
        if not file_path.exists():
            print(f"Error: File {file_path} not found")
            return
            
        # Load and parse JSON
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Extract and display config
        if 'config' in data:
            config = data['config']
            print("\nConfig contents:")
            print("---------------")
            for key, value in config.items():
                print(f"{key}: {value}")
        else:
            print("No config section found in the file")
            
    except json.JSONDecodeError:
        print(f"Error: {file_stem}.json is not a valid JSON file")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python inspect_config.py <file_stem>")
        sys.exit(1)
        
    inspect_config(sys.argv[1])