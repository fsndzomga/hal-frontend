# I copied this from https://github.com/vminvsky/hal-analysis and slightly edited.

import pandas as pd
import json
from glob import glob
import numpy as np
import ast
import os

MODEL_NAME_MAP = {
    "claude-3-7-sonnet-20250219": "claude-3-7-sonnet-2025-02-19",
    "claude-3-7-sonnet-20250219 high": "claude-3-7-sonnet-2025-02-19 high",
    "together_ai/deepseek-ai/DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    "openai/gpt-4.1-2025-04-14": "gpt-4.1-2025-04-14",
    "together_ai/deepseek-ai/DeepSeek-V3": "deepseek-ai/DeepSeek-V3",
    "gemini/gemini-2.0-flash": "gemini-2.0-flash",
    "o4-mini-2025-04-16 medium": "o4-mini-2025-04-16",
    "gpt-4.1":"gpt-4.1-2025-04-14",
    "claude-3-7-sonnet-20250219 low": "claude-3-7-sonnet-2025-02-19 low",
    "anthropic/claude-3-7-sonnet-20250219": "claude-3-7-sonnet-2025-02-19",
    "openai/o3-2025-04-16 low": "o3-2025-04-16 low",
    "openai/o4-mini-2025-04-16": "o4-mini-2025-04-16",
    "openai/o4-mini-2025-04-16 low": "o4-mini-2025-04-16 low",
    "openai/o4-mini-2025-04-16 high": "o4-mini-2025-04-16 high",
    "openai/o3-2025-04-16": "o3-2025-04-16",
    "o3-2025-04-16 medium": "o3-2025-04-16",
    "openai/o3-mini-2025-01-31 low": "o3-mini-2025-01-31 low",
    "anthropic/claude-3-7-sonnet-20250219 low": "claude-3-7-sonnet-2025-02-19 low",
    "anthropic/claude-3-7-sonnet-20250219 high": "claude-3-7-sonnet-2025-02-19 high",
    "o3-2025-04-03": "o3-2025-04-16"
}

AGENT_NAME_MAP = {
    "taubench_fewshot_o320250403": "TAU-bench FewShot (o3-2025-04-03)",
    "taubench_fewshot_o3mini20250131_high": "TAU-bench FewShot (o3-mini-2025-01-31)",
    "TAU-bench FewShot (claude-3-7-sonnet-20250219)": "TAU-bench FewShot (claude-3-7-sonnet-2025-02-19)",
    "HAL Generalist Agent (claude-3-7-sonnet-20250219)": "HAL Generalist Agent (claude-3-7-sonnet-2025-02-19)",
    "usaco_episodic__semantic_o3mini20250131_high": "USACO Episodic + Semantic (o3-mini-2025-01-31 high)",
    "usaco_episodic__semantic_deepseekaideepseekr1": "USACO Episodic + Semantic (deepseek-ai/DeepSeek-R1)",
    "usaco_episodic__semantic_gpt4120250414": "USACO Episodic + Semantic (gpt-4.1-2025-04-14)",
    "usaco_episodic__semantic_o3mini20250131_low": "USACO Episodic + Semantic (o3-mini-2025-01-31 low)",
    "usaco_episodic__semantic_deepseekaideepseekv3": "USACO Episodic + Semantic (deepseek-ai/DeepSeek-V3)",
    "hal_generalist_agent_o3mini20250131_high": "HAL Generalist Agent (o3-mini-2025-01-31 high)",
    "My Agent(o3-mini)": "SWE-Agent (o3-mini)",
    "My Agent(gemini/gemini-2.0-flash)": "SWE-Agent (gemini-2.0-flash)",
    "hal_generalist_agent_gemini20flash": "HAL Generalist Agent (gemini-2.0-flash)",
    "My Agent(claude-3-7-sonnet-20250219)": "SWE-Agent (claude-3-7-sonnet-2025-02-19)",
    "My Agent(o3-2025-04-16)": "SWE-Agent (o3-2025-04-16)",
    "My Agent(gpt-4o)": "SWE-Agent (gpt-4o)",
    "hal_generalist_agent_deepseekaideepseekv3": "HAL Generalist Agent (deepseek-ai/DeepSeek-V3)",
    "hal_generalist_agent_deepseekaideepseekr1": "HAL Generalist Agent (deepseek-ai/DeepSeek-R1)",
    "My Agent(o4-mini-2025-04-16)": "SWE-Agent (o4-mini-2025-04-16)",
    "my_agenttogether_aideepseekaideepseekv3": "SWE-Agent (deepseek-ai/DeepSeek-V3)",
    "My Agent(o1)": "SWE-Agent (o1)",
    "hal_generalist_agent_o3mini20250131_low": "HAL Generalist Agent (o3-mini-2025-01-31 low)",
    "My Agent(together_ai/deepseek-ai/DeepSeek-R1)": "SWE-Agent (deepseek-ai/DeepSeek-R1)",
    "My Agent(gpt-4.1)": "SWE-Agent (gpt-4.1)",
    "colbench_text_o4mini": "Col-bench Text (o4-mini-2025-04-16 low)",
    "colbench_text_gemmini2flash": "Col-bench Text (gemini-2.0-flash)",
    "colbench_text_gpt41": "Col-bench Text (gpt-4.1-2025-04-14)",
    "colbench_text_sonnet37": "Col-bench Text (claude-3-7-sonnet-2025-02-19 low)",
    "colbench_text_deepseekr1": "Col-bench Text (deepseek-ai/DeepSeek-R1)",
    "colbench_text_deepseekv3": "Col-bench Text (deepseek-ai/DeepSeek-V3)",
    "colbench_text_gemini": "Col-bench Text (gemini-2.0-flash)",
    "colbench_text_deepseekv3": "Col-bench Text (deepseek-ai/DeepSeek-V3)",
    "colbench_text_sonnet": "Col-bench Text (claude-3-7-sonnet-2025-02-19)",
    "colbench_text_o4minilow": "Col-bench Text (o4-mini-2025-04-16 low)",
    "colbench_text_deepseekr1": "Col-bench Text (deepseek-ai/DeepSeek-R1)",
    "colbench_text_o4minimedium": "Col-bench Text (o4-mini-2025-04-16 medium)",
    "colbench_text_o4minihigh": "Col-bench Text (o4-mini-2025-04-16 high)",
    "SAB Self-Debug o3 medium": "SAB Example Agent (o3-2025-04-16)",
    "SAB Self-Debug Claude-3-7": "SAB Example Agent (claude-3-7-sonnet-2025-02-19)",
    "SAB Self-Debug o4-mini low": "SAB Example Agent (o4-mini-2025-04-16 low)",
    "SAB Self-Debug gemini-2-0-flash": "SAB Example Agent (gemini-2.0-flash)",
    "SAB Self-Debug Claude-3-7 high": "SAB Example Agent (claude-3-7-sonnet-2025-02-19 high)",
    "sab_selfdebug_gpt_41": "SAB Example Agent (gpt-4.1-2025-04-14)",
    "SAB Self-Debug DS-V3": "SAB Example Agent (deepseek-ai/DeepSeek-V3)",
    "SAB Self-Debug Claude-3-7 low": "SAB Example Agent (claude-3-7-sonnet-2025-02-19 low)",
    "SAB Self-Debug o4-mini high": "SAB Example Agent (o4-mini-2025-04-16 high)",
    "sab_selfdebug_dsr1": "SAB Example Agent (deepseek-ai/DeepSeek-R1)",
    "SAB Self-Debug gemini-2-5-pro": "SAB Example Agent (gemini-2.5-pro-preview)",
    "Hal Generalist Agent (o3-2025-04-16)": "HAL Generalist Agent (o3-2025-04-16)",
    "Hal Generalist Agent (gpt-4.1-2025-04-14)": "HAL Generalist Agent (gpt-4.1-2025-04-14)",
    "Hal Generalist Agent (o4-mini-2025-04-16 low)": "HAL Generalist Agent (o4-mini-2025-04-16 low)",
    "Hal Generalist Agent (o4-mini-2025-04-16)": "HAL Generalist Agent (o4-mini-2025-04-16)",
    "Hal Generalist Agent (o4-mini-2025-04-16 high)": "HAL Generalist Agent (o4-mini-2025-04-16 high)",
    "coreagent": "CORE-Agent",
    'Browser-Use_test(DeepSeek-R1)': 'Browser-Use(DeepSeek-R1)',
    "Browser-Use_test(claude-3-7-sonnet-20250219)": "Browser-Use(claude-3-7-sonnet-20250219)",
    "Browser-Use_test(DeepSeek-V3)": "Browser-Use(DeepSeek-V3)",
    "Hal Generalist Agent (GPT4.1)": "HAL Generalist Agent (GPT4.1)",
    "Hal Generalist Agent (DeepSeek-R1)": "HAL Generalist Agent (DeepSeek-R1)",
    "Hal Generalist Agent (O3-low)": "HAL Generalist Agent (O3-low)",
    "Hal Generalist Agent (DeepSeek-V3)": "HAL Generalist Agent (DeepSeek-V3)",
    "Hal Generalist Agent (Sonnet3.7)": "HAL Generalist Agent (Sonnet3.7)",
    "Hal Generalist Agent (O4-mini-high)": "HAL Generalist Agent (O4-mini-high)",
    "Hal Generalist Agent (Sonnet 3.7)": "HAL Generalist Agent (Sonnet 3.7)",
    "Hal Generalist Agent (o4-mini-high)": "HAL Generalist Agent (o4-mini-high)",
    "Hal Generalist Agent (o3-low)": "HAL Generalist Agent (o3-2025-04-16 low)",
    "Hal Generalist Agent (o4-mini-low)": "HAL Generalist Agent (o4-mini-low)"
}

AGENT_NAME_MAP = {
    "taubench_fewshot_o320250403": "TAU-bench FewShot (o3-2025-04-03)",
    "taubench_fewshot_o3mini20250131_high": "TAU-bench FewShot (o3-mini-2025-01-31)",
    "TAU-bench FewShot (claude-3-7-sonnet-20250219)": "TAU-bench FewShot (claude-3-7-sonnet-2025-02-19)",
    "HAL Generalist Agent (claude-3-7-sonnet-20250219)": "HAL Generalist Agent (claude-3-7-sonnet-2025-02-19)",
    "usaco_episodic__semantic_o3mini20250131_high": "USACO Episodic + Semantic (o3-mini-2025-01-31 high)",
    "usaco_episodic__semantic_deepseekaideepseekr1": "USACO Episodic + Semantic (deepseek-ai/DeepSeek-R1)",
    "usaco_episodic__semantic_gpt4120250414": "USACO Episodic + Semantic (gpt-4.1-2025-04-14)",
    "usaco_episodic__semantic_o3mini20250131_low": "USACO Episodic + Semantic (o3-mini-2025-01-31 low)",
    "usaco_episodic__semantic_deepseekaideepseekv3": "USACO Episodic + Semantic (deepseek-ai/DeepSeek-V3)",
    "hal_generalist_agent_o3mini20250131_high": "HAL Generalist Agent (o3-mini-2025-01-31 high)",
    "My Agent(o3-mini)": "SWE-Agent (o3-mini)",
    "My Agent(gemini/gemini-2.0-flash)": "SWE-Agent (gemini-2.0-flash)",
    "hal_generalist_agent_gemini20flash": "HAL Generalist Agent (gemini-2.0-flash)",
    "My Agent(claude-3-7-sonnet-20250219)": "SWE-Agent (claude-3-7-sonnet-2025-02-19)",
    "My Agent(o3-2025-04-16)": "SWE-Agent (o3-2025-04-16)",
    "My Agent(gpt-4o)": "SWE-Agent (gpt-4o)",
    "hal_generalist_agent_deepseekaideepseekv3": "HAL Generalist Agent (deepseek-ai/DeepSeek-V3)",
    "hal_generalist_agent_deepseekaideepseekr1": "HAL Generalist Agent (deepseek-ai/DeepSeek-R1)",
    "My Agent(o4-mini-2025-04-16)": "SWE-Agent (o4-mini-2025-04-16)",
    "my_agenttogether_aideepseekaideepseekv3": "SWE-Agent (deepseek-ai/DeepSeek-V3)",
    "My Agent(o1)": "SWE-Agent (o1)",
    "hal_generalist_agent_o3mini20250131_low": "HAL Generalist Agent (o3-mini-2025-01-31 low)",
    "My Agent(together_ai/deepseek-ai/DeepSeek-R1)": "SWE-Agent (deepseek-ai/DeepSeek-R1)",
    "My Agent(gpt-4.1)": "SWE-Agent (gpt-4.1)",
    "colbench_text_o4mini": "Col-bench Text (o4-mini-2025-04-16 low)",
    "colbench_text_gemmini2flash": "Col-bench Text (gemini-2.0-flash)",
    "colbench_text_gpt41": "Col-bench Text (gpt-4.1-2025-04-14)",
    "colbench_text_sonnet37": "Col-bench Text (claude-3-7-sonnet-2025-02-19 low)",
    "colbench_text_deepseekr1": "Col-bench Text (deepseek-ai/DeepSeek-R1)",
    "colbench_text_deepseekv3": "Col-bench Text (deepseek-ai/DeepSeek-V3)",
    "colbench_text_gemini": "Col-bench Text (gemini-2.0-flash)",
    "colbench_text_deepseekv3": "Col-bench Text (deepseek-ai/DeepSeek-V3)",
    "colbench_text_sonnet": "Col-bench Text (claude-3-7-sonnet-2025-02-19)",
    "colbench_text_o4minilow": "Col-bench Text (o4-mini-2025-04-16 low)",
    "colbench_text_deepseekr1": "Col-bench Text (deepseek-ai/DeepSeek-R1)",
    "colbench_text_o4minimedium": "Col-bench Text (o4-mini-2025-04-16 medium)",
    "colbench_text_o4minihigh": "Col-bench Text (o4-mini-2025-04-16 high)",
    "SAB Self-Debug o3 medium": "SAB Example Agent (o3-2025-04-16)",
    "SAB Self-Debug Claude-3-7": "SAB Example Agent (claude-3-7-sonnet-2025-02-19)",
    "SAB Self-Debug o4-mini low": "SAB Example Agent (o4-mini-2025-04-16 low)",
    "SAB Self-Debug gemini-2-0-flash": "SAB Example Agent (gemini-2.0-flash)",
    "SAB Self-Debug Claude-3-7 high": "SAB Example Agent (claude-3-7-sonnet-2025-02-19 high)",
    "sab_selfdebug_gpt_41": "SAB Example Agent (gpt-4.1-2025-04-14)",
    "SAB Self-Debug DS-V3": "SAB Example Agent (deepseek-ai/DeepSeek-V3)",
    "SAB Self-Debug Claude-3-7 low": "SAB Example Agent (claude-3-7-sonnet-2025-02-19 low)",
    "SAB Self-Debug o4-mini high": "SAB Example Agent (o4-mini-2025-04-16 high)",
    "sab_selfdebug_dsr1": "SAB Example Agent (deepseek-ai/DeepSeek-R1)",
    "SAB Self-Debug gemini-2-5-pro": "SAB Example Agent (gemini-2.5-pro-preview)",
    "coreagent": "CORE-Agent",
    'Browser-Use_test(DeepSeek-R1)': 'Browser-Use(DeepSeek-R1)',
    "Browser-Use_test(claude-3-7-sonnet-20250219)": "Browser-Use(claude-3-7-sonnet-20250219)",
    "Browser-Use_test(DeepSeek-V3)": "Browser-Use(DeepSeek-V3)",
}

AGENT_NAME_SHORT_MAP = {
    "Hal Generalist Agent": "HAL Generalist Agent"
}


class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = self._load_data()
        self.config = self.data['config']

        # normalize agent
        orig_agent = self.config.get('agent_name')
        norm_agent = AGENT_NAME_MAP.get(orig_agent, orig_agent)
        self.config['agent_name'] = norm_agent
        # short form
        if norm_agent in ('CORE-Agent', 'HAL Generalist Agent'):
            short_a = norm_agent
        elif any(sub in norm_agent for sub in ('Browser-Use', 'SeeAct')):
            short_a = norm_agent.split('(')[0]
        else:
            cand = norm_agent.split(' (')[-2]
            short_a = AGENT_NAME_SHORT_MAP.get(cand, cand)
        self.config['agent_name_short'] = short_a

        # normalize model
        args = self.config.get('agent_args', {})
        if 'agent.model.name' in args:
            m = args['agent.model.name']
            if 'agent.model.reasoning_effort' in args:
                m = f"{m} {args['agent.model.reasoning_effort']}"
            norm_m = MODEL_NAME_MAP.get(m, m)
        elif 'model_name' in args:
            m = args['model_name']
            if 'reasoning_effort' in args:
                m = f"{m} {args['reasoning_effort']}"
            norm_m = MODEL_NAME_MAP.get(m, m)
        else:
            raise KeyError(f"No model name in {data_path}")
        self.config['model_name_short'] = norm_m

        # write back changes (rename old keys then overwrite)
        self._write_back()

    def _load_data(self):
        with open(self.data_path, 'r') as f:
            return json.load(f)

    def _write_back(self):
        cfg = self.data['config']
        # preserve old
        if 'agent_name' in cfg:
            cfg['agent_name_old'] = cfg['agent_name']
        args = cfg.get('agent_args', {})
        if 'model_name' in args:
            args['model_name_old'] = args['model_name']
        if 'agent.model.name' in args:
            args['agent.model.name_old'] = args['agent.model.name']
        # overwrite
        cfg['agent_name'] = self.config['agent_name']
        if 'agent.model.name' in args:
            args['agent.model.name'] = self.config['model_name_short']
        elif 'model_name' in args:
            args['model_name'] = self.config['model_name_short']
        # dump
        with open(self.data_path, 'w') as f:
            json.dump(self.data, f, indent=2)

def ensure_list(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, str):
        try:
            val = ast.literal_eval(x)
            if isinstance(val, (list, tuple)):
                return list(val)
            else:
                return [val]
        except Exception:
            return []
    return [x]

def main():
    files = glob('/workspaces/hal-frontend/evals_live/*.json')

    records = []
    for fp in files:
        dl = DataLoader(fp)
        records.append({'file': fp, 'model': dl.config['model_name_short']})

    df = pd.DataFrame(records)

    # filtering
    models_to_remove = [
        '2.5-pro', 'o1', 'o3-mini', 'gpt-4o',
        'o3-2025-04-16 low', 'claude-3-7-sonnet-2025-02-19 low'
    ]
    pat = '|'.join(models_to_remove)
    mask = ~df['model'].str.contains(pat, case=False, na=False)
    mask &= (df['model'] != 'o4-mini-2025-04-16')

    to_parse = df[mask]

    out_dir = 'data'
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'jsons_to_update.csv')
    to_parse[['file']].to_csv(out_csv, index=False)
    print(f"Wrote {len(to_parse)} entries to {out_csv}")

if __name__ == '__main__':
    main()
