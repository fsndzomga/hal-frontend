import os
import json
import glob
import shutil
import pandas as pd
from dotenv import load_dotenv

# ———————————————
# 0. Load your name maps
# ———————————————
name_map = pd.read_csv("hal_name_map.csv")
model_map = (
    name_map.query("map_name=='MODEL_NAME_MAP'")
    .set_index("key")["value"]
    .to_dict()
)
agent_map = (
    name_map.query("map_name=='AGENT_NAME_MAP'")
    .set_index("key")["value"]
    .to_dict()
)

# ———————————————
# 1. Prepare folders
# ———————————————
EVAL_DIR    = "evals_live"
ARCHIVE_DIR = "evals_archive"
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# ———————————————
# 2. Scan & remap
# ———————————————
for src_path in glob.glob(os.path.join(EVAL_DIR, "*.json")):
    with open(src_path, "r") as f:
        payload = json.load(f)

    cfg = payload.get("config", {})
    orig_agent = cfg.get("agent_name", "")
    orig_model = cfg.get("agent_args", {}).get("agent.model.name", "")

    # Lookup in your maps
    new_agent = agent_map.get(orig_agent, orig_agent)
    new_model = model_map.get(orig_model, orig_model)

    # Only rewrite if something changed
    if new_agent != orig_agent or new_model != orig_model:
        # 1) Archive the original
        basename = os.path.basename(src_path)
        shutil.move(src_path, os.path.join(ARCHIVE_DIR, basename))

        # 2) Apply the mappings in the payload
        cfg["agent_name"] = new_agent
        cfg.setdefault("agent_args", {})["agent.model.name"] = new_model

        # 3) Write back as a new file with "_mapped" suffix
        name, ext = os.path.splitext(basename)
        dst = os.path.join(EVAL_DIR, f"{name}_mapped{ext}")
        with open(dst, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"Mapped and wrote: {dst}")
    else:
        # No change needed—leave file untouched
        print(f"No mapping needed: {src_path}")
