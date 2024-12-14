import os
import json
import asyncio
import aiofiles
from agent_monitor.monitor import analyze_agent_steps
from agent_monitor.failure_report import analyze_agent_performance, AsyncOpenAIClient
import traceback
from tqdm import tqdm

async def check_and_process_uploads():
    upload_dir =  "evals_upload"
    processed_dir = "evals_processed"
    live_dir = "evals_live"
    
    new_uploads = [f for f in os.listdir(upload_dir) if f.endswith('.json')]
    
    if not new_uploads:
        print("No new uploads found.")
        return
    
    # check for all new uploads whether they are already in live or processed directory
    # Also check whether the files are actually identical
    unprocessed_uploads = []
    for upload in new_uploads:
        upload_path = os.path.join(upload_dir, upload)
        processed_path = os.path.join(processed_dir, upload)
        live_path = os.path.join(live_dir, upload)
        
        if not os.path.exists(live_path) and not os.path.exists(processed_path):
            unprocessed_uploads.append(upload)
        elif os.path.exists(processed_path):

            print(f"Upload {upload} is already in processed directory.")
            
        elif os.path.exists(live_path):
            print(f"Upload {upload} is already in live directory.")
        else:
            unprocessed_uploads.append(upload)

    print(f"Processing {len(unprocessed_uploads)} new uploads.")
    tasks = []
    for upload in tqdm(unprocessed_uploads):
        upload_path = os.path.join(upload_dir, upload)
        processed_path = os.path.join(processed_dir, upload)
        # tasks.append(process_single_upload(upload_path, processed_path)) # for async processing
        await process_single_upload(upload_path, processed_path)


    # await asyncio.gather(*tasks) # for async processing


async def process_single_upload(upload_path, processed_path):
    # Check the structure of the upload
    check_result = await check_upload_structure(upload_path)
    
    if check_result['is_valid']:
        # Process the file
        await process_upload(upload_path, processed_path)
        
        # Move the file to processed directory
        # await asyncio.to_thread(shutil.move, upload_path, processed_path)
        
    else:
        print(f"Upload check failed for {upload_path}: {check_result['message']}")

        
        
async def check_upload_structure(file_path):
    try:
        async with aiofiles.open(file_path, 'r') as f:
            data = json.loads(await f.read())
        
        # Check for required keys
        required_keys = ['config', 'results', 'raw_eval_results', 'raw_logging_results']
        missing_keys = [key for key in required_keys if key not in data]
        
        if missing_keys:
            return {'is_valid': False, 'message': f"Missing required keys: {', '.join(missing_keys)}"}
        
        # Check for specific structure in raw_logging_results
        if not isinstance(data['raw_logging_results'], list) and not "inspect" in data['config']['benchmark_name']:
            return {'is_valid': False, 'message': "raw_logging_results should be a list"}
        
        if "inspect" not in data['config']['benchmark_name']:
            for item in data['raw_logging_results']:
                if not all(key in item for key in ['weave_task_id', 'inputs', 'outputs']):
                    return {'is_valid': False, 'message': "Each item in raw_logging_results should have weave_task_id, inputs, and outputs"}
        
        return {'is_valid': True, 'message': "File structure is valid"}

    except json.JSONDecodeError:
        return {'is_valid': False, 'message': "Invalid JSON format"}
    except Exception as e:
        return {'is_valid': False, 'message': f"Unexpected error: {str(e)}"}
    

async def process_upload(input_path, output_path):
    print(f"Processing {input_path}...")
    # load the file
    with open(input_path, 'r') as f:
        data = json.loads(f.read())

    assert 'raw_logging_results' in data, "raw_logging_results key not found in the file"

    try:
        if isinstance(data['raw_logging_results'], list):
            openai_client = AsyncOpenAIClient(model="gpt-4o-mini")
            processed_calls = await analyze_agent_steps(data['raw_logging_results'], openai_client, llm_eval=False)
        else:
            processed_calls = data['raw_logging_results']
            
            
        # # experimental
        # failure_report = await analyze_agent_performance(data['raw_logging_results'], data['results']['failed_tasks'], openai_client)

        data['raw_logging_results'] = processed_calls
        data['failure_report'] = None
    except Exception as e:
        traceback.print_exc()
        print(f"Error in processing: {str(e)}")
        return

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Processing of {input_path} successful. Results saved to {output_path}")




if __name__ == "__main__":
    # process single upload for testing
    asyncio.run(process_single_upload("evals_upload/inspect_evalsswe_bench_1729538131_UPLOAD.json", "evals_processed/inspect_evalsswe_bench_1729538131_UPLOAD.json"))