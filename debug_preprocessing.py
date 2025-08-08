import json
import os
from utils.db import TracePreprocessor

def debug_preprocessing():
    """Debug preprocessing for SWE-Agent o4-mini files"""
    
    # Check the specific files we found
    o4_mini_files = [
        "swebench_verified_mini_my_agento4mini20250416_1745279519_UPLOAD.json",  # Low
        "swebench_verified_mini_my_agento4mini20250416_1745345781_UPLOAD.json"   # High
    ]
    
    preprocessor = TracePreprocessor()
    
    for filename in o4_mini_files:
        filepath = f"/workspaces/hal-frontend/evals_live/{filename}"
        print(f"\n=== DEBUGGING {filename} ===")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            config = data.get('config', {})
            agent_args = config.get('agent_args', {})
            
            print(f"Agent name: {config.get('agent_name')}")
            print(f"Benchmark: {config.get('benchmark_name')}")
            print(f"Run ID: {config.get('run_id')}")
            
            # Check model name extraction logic
            model_name_short = config.get('model_name_short')
            print(f"model_name_short from config: {model_name_short}")
            
            if model_name_short is None:
                # Try different possible keys
                model_name = agent_args.get('model_name') or agent_args.get('agent.model.name')
                reasoning_effort = agent_args.get('reasoning_effort') or agent_args.get('agent.model.reasoning_effort')
                print(f"model_name from agent_args: {model_name}")
                print(f"reasoning_effort from agent_args: {reasoning_effort}")
                print(f"Available agent_args keys: {list(agent_args.keys())}")
                
                if reasoning_effort:
                    primary_model_name = f"{model_name} {reasoning_effort}"
                else:
                    primary_model_name = model_name
            else:
                primary_model_name = model_name_short
                
            print(f"Final primary_model_name: {primary_model_name}")
            
            # Check model mapping
            show_primary_model_name = preprocessor.get_model_show_name(primary_model_name)
            print(f"show_primary_model_name after mapping: {show_primary_model_name}")
            
            # Check if it would be skipped
            from utils.db import MODELS_TO_SKIP
            if show_primary_model_name in MODELS_TO_SKIP:
                print(f"❌ WOULD BE SKIPPED! Model in MODELS_TO_SKIP")
            else:
                print(f"✅ Would be processed")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    debug_preprocessing()
