import json
import os
from utils.db import TracePreprocessor, MODELS_TO_SKIP

def check_all_files_processing():
    """Check if all files in evals_live would be processed"""
    
    evals_dir = "/workspaces/hal-frontend/evals_live"
    preprocessor = TracePreprocessor()
    
    total_files = 0
    would_process = 0
    would_skip = 0
    errors = 0
    
    skip_reasons = {}
    error_files = []
    
    print("=== CHECKING ALL FILES IN EVALS_LIVE ===\n")
    
    for filename in sorted(os.listdir(evals_dir)):
        if not filename.endswith('.json'):
            continue
            
        total_files += 1
        filepath = os.path.join(evals_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                # Read just the first part to get config
                content = f.read(5000)  # Read first 5KB
                if '"config"' not in content:
                    continue
                    
                # Try to parse as JSON
                f.seek(0)
                data = json.load(f)
            
            config = data.get('config', {})
            agent_args = config.get('agent_args', {})
            agent_name = config.get('agent_name', 'Unknown')
            benchmark_name = config.get('benchmark_name', 'Unknown')
            
            # Simulate the preprocessing logic
            
            # 1. Extract primary model name
            if benchmark_name in ['corebench_hard', 'swebench_verified_mini', 'taubench_airline']:
                primary_model_name = config.get('model_name_short')
                if primary_model_name is None:
                    # Try both old and new key formats
                    primary_model_name = (agent_args.get('model_name') or 
                                        agent_args.get('agent.model.name'))
                    # get reasoning effort if any - try both old and new key formats
                    reasoning_effort = (agent_args.get('reasoning_effort') or
                                      agent_args.get('agent.model.reasoning_effort'))
                    if reasoning_effort:
                        primary_model_name = f"{primary_model_name} {reasoning_effort}"
            else:
                # Find primary model from agent_name knowing Agent name is in the format "AgentName (ModelName)"
                primary_model_name = agent_name.split('(')[-1].strip(' )') if '(' in agent_name else None
            
            # 2. Get show name
            show_primary_model_name = preprocessor.get_model_show_name(primary_model_name) if primary_model_name else None
            
            # 3. Check if would be skipped
            if show_primary_model_name in MODELS_TO_SKIP:
                would_skip += 1
                reason = f"Model '{show_primary_model_name}' in MODELS_TO_SKIP"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                print(f"❌ SKIP: {filename}")
                print(f"   Agent: {agent_name}")
                print(f"   Model: {primary_model_name} → {show_primary_model_name}")
                print(f"   Reason: {reason}\n")
            else:
                would_process += 1
                print(f"✅ PROCESS: {filename}")
                print(f"   Agent: {agent_name}")
                print(f"   Model: {primary_model_name} → {show_primary_model_name}\n")
                
        except Exception as e:
            errors += 1
            error_files.append((filename, str(e)))
            print(f"⚠️  ERROR: {filename} - {e}\n")
    
    print("=" * 60)
    print("SUMMARY:")
    print(f"Total JSON files: {total_files}")
    print(f"Would be processed: {would_process}")
    print(f"Would be skipped: {would_skip}")
    print(f"Errors: {errors}")
    
    if skip_reasons:
        print(f"\nSkip reasons:")
        for reason, count in skip_reasons.items():
            print(f"  {reason}: {count} files")
    
    if error_files:
        print(f"\nError files:")
        for filename, error in error_files:
            print(f"  {filename}: {error}")
    
    print(f"\nProcessing rate: {would_process/total_files*100:.1f}%")

if __name__ == "__main__":
    check_all_files_processing()
