import os
import zipfile
import json
from json_encryption import JsonEncryption
from tqdm import tqdm
from huggingface_hub import HfApi

# Create an instance with your password
encryptor = JsonEncryption("hal1234")

# Create encrypted_files directory if it doesn't exist
encrypted_dir = "encrypted_files"
if not os.path.exists(encrypted_dir):
    os.makedirs(encrypted_dir)

# Get all JSON files from evals_live directory
json_files = [f for f in os.listdir("evals_live") if f.endswith(".json")]

if not json_files:
    print("No files found to encrypt and upload")
    exit()

# Initialize Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Process each file
for json_file in tqdm(json_files, desc="Encrypting files", unit="file"):
    input_path = os.path.join("evals_live", json_file)
    
    # Read the JSON to get the run_id
    with open(input_path) as f:
        data = json.load(f)
        run_id = data["config"]["run_id"]
    
    encrypted_path = os.path.join(encrypted_dir, json_file)
    zip_path = os.path.join(encrypted_dir, f"{run_id}.zip")
    
    try:
        # Only encrypt and zip if the zip doesn't exist yet
        if not os.path.exists(zip_path):
            # Encrypt the JSON file
            encryptor.encrypt_json_file(input_path, encrypted_path)
            
            # Create zip archive containing the encrypted file
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(encrypted_path, os.path.basename(encrypted_path))
            
            # Remove the unzipped encrypted file
            os.remove(encrypted_path)

        # Upload to Hugging Face Hub (attempt for all files)
        api.upload_file(
            path_or_fileobj=zip_path,
            path_in_repo=os.path.basename(zip_path),
            repo_id="agent-evals/agent_traces",
            repo_type="dataset",
            commit_message=f"Add encrypted trace for {json_file}"
        )
        print(f"Successfully uploaded {zip_path}")

    except Exception as e:
        print(f"Error processing {json_file}: {str(e)}")

print(f"Processed {len(json_files)} files")