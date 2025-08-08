import json
from pathlib import Path
import os

def reduce_json_file(file_path):
    """Reduce a JSON file to keep only essential keys"""
    try:
        # Read the original file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check if already processed
        if 'metadata' in data and data.get('metadata', {}).get('processed'):
            print(f"File {file_path} already reduced, skipping...")
            return
        
        original_size = os.path.getsize(file_path)
        
        # Create reduced version with only essential keys
        reduced_data = {
            'config': data.get('config', {}),
            'results': data.get('results', {}),
            'total_usage': data.get('total_usage', {}),
            'metadata': {
                'processed': True,
                'original_file_size_bytes': original_size,
                'processed_date': data.get('config', {}).get('date', 'unknown')
            }
        }
        
        # Write the reduced version back to the same file
        with open(file_path, 'w') as f:
            json.dump(reduced_data, f, indent=2)
        
        new_size = os.path.getsize(file_path)
        reduction_percent = ((original_size - new_size) / original_size) * 100
        
        print(f"Reduced {file_path.name}: {original_size:,} → {new_size:,} bytes ({reduction_percent:.1f}% reduction)")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    """Process all JSON files in the evals_live directory"""
    processed_dir = Path("evals_live")
    
    if not processed_dir.exists():
        print(f"Directory {processed_dir} does not exist!")
        return
    
    # Read the list of processed files to only reduce those
    processed_files_log = Path("preprocessed_traces/processed_files.txt")
    processed_files = set()
    
    if processed_files_log.exists():
        with open(processed_files_log, 'r') as f:
            processed_files = set(line.strip() for line in f)
    else:
        print("No processed_files.txt found, will process all JSON files")
    
    json_files = list(processed_dir.glob('*.json'))
    total_files = len(json_files)
    
    if not json_files:
        print("No JSON files found in evals_live directory")
        return
    
    print(f"Found {total_files} JSON files to potentially reduce...")
    
    total_original_size = 0
    total_new_size = 0
    files_processed = 0
    
    for i, file_path in enumerate(json_files, 1):
        # Only process files that are in the processed_files list (if it exists)
        if processed_files and str(file_path) not in processed_files:
            continue
            
        print(f"[{i}/{total_files}] Processing {file_path.name}...")
        
        original_size = os.path.getsize(file_path)
        reduce_json_file(file_path)
        new_size = os.path.getsize(file_path)
        
        total_original_size += original_size
        total_new_size += new_size
        files_processed += 1
    
    if files_processed > 0:
        total_reduction_percent = ((total_original_size - total_new_size) / total_original_size) * 100
        print(f"\n=== SUMMARY ===")
        print(f"Files processed: {files_processed}")
        print(f"Total size reduction: {total_original_size:,} → {total_new_size:,} bytes")
        print(f"Overall reduction: {total_reduction_percent:.1f}%")
        print(f"Space saved: {total_original_size - total_new_size:,} bytes ({(total_original_size - total_new_size) / (1024*1024):.1f} MB)")
    else:
        print("No files were processed")

if __name__ == "__main__":
    main()
