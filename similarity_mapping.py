import os
import json
import glob
import shutil
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# ———————————————
# 0. Configuration
# ———————————————
load_dotenv()
client    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMB_MODEL = "text-embedding-3-small"  # or "text-embedding-3-large"
EVAL_DIR   = "evals_live"
ARCHIVE_DIR = "evals_archive"

os.makedirs(ARCHIVE_DIR, exist_ok=True)


# ———————————————
# 1. Load canonical maps
# ———————————————
name_map  = pd.read_csv("hal_name_map.csv")
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
# 2. Archive & normalize any JSON needing a map
# ———————————————
for src in glob.glob(os.path.join(EVAL_DIR, "*.json")):
    data = json.load(open(src))
    cfg  = data.get("config", {})
    orig_agent = cfg.get("agent_name", "")
    orig_model = cfg.get("agent_args", {}).get("agent.model.name", "")

    # Apply your maps (fall back to original if not in map)
    new_agent = agent_map.get(orig_agent, orig_agent)
    new_model = model_map.get(orig_model, orig_model)

    # If either changed, archive and write mapped file
    if new_agent != orig_agent or new_model != orig_model:
        # 1) Move original to archive
        shutil.move(src, os.path.join(ARCHIVE_DIR, os.path.basename(src)))

        # 2) Update in-memory JSON
        cfg["agent_name"] = new_agent
        cfg["agent_args"]["agent.model.name"] = new_model

        # 3) Write mapped version in place
        base, _ = os.path.splitext(os.path.basename(src))
        dst = os.path.join(EVAL_DIR, f"{base}_mapped.json")
        with open(dst, "w") as f:
            json.dump(data, f, indent=2)


# ———————————————
# 3. Read & normalize all JSONs in evals_live
# ———————————————
records = []
for path in glob.glob(os.path.join(EVAL_DIR, "*.json")):
    j   = json.load(open(path))
    cfg = j.get("config", {})
    bench = cfg.get("benchmark_name", "")
    agent = cfg.get("agent_name", "")
    model = cfg.get("agent_args", {}).get("agent.model.name", "")
    combined = f"{bench} {agent} {model}"

    records.append({
        "path":          path,
        "benchmark":     bench,
        "agent":         agent,
        "model":         model,
        "combined_norm": combined,
        "run_id":        cfg.get("run_id", ""),
        "date":          cfg.get("date", ""),
    })

files_df = pd.DataFrame(records)


# ———————————————
# 4. Load & filter your agent runs
# ———————————————
agent_df = pd.read_csv("agent_run_status.csv")
uploads  = agent_df[
    agent_df["Status"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.contains("upload", regex=False)
].copy()
uploads["combined"] = (
      uploads["Benchmark"].astype(str) + " "
    + uploads["Agent"].astype(str)     + " "
    + uploads["Model"].astype(str)
)

print(f">> Found {len(uploads)} upload runs to match.")


# ———————————————
# 5. Batched embeddings helper
# ———————————————
def get_embeddings(texts, model=EMB_MODEL, batch_size=500):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = [t.replace("\n", " ") for t in texts[i : i + batch_size]]
        resp = client.embeddings.create(input=batch, model=model)
        embeddings.extend([d.embedding for d in resp.data])
    return embeddings

agent_embs = get_embeddings(uploads["combined"].tolist())
file_embs  = get_embeddings(files_df["combined_norm"].tolist())


# ———————————————
# 6. Cosine‐match each upload to best JSON
# ———————————————
matches = []
for idx, (_, arow) in enumerate(uploads.iterrows()):
    sims   = cosine_similarity([agent_embs[idx]], file_embs)[0]
    best_j = sims.argmax()
    frow   = files_df.iloc[best_j]

    matches.append({
        "Benchmark":    arow["Benchmark"],
        "Agent":        arow["Agent"],
        "Model":        arow["Model"],
        "Acc":          arow["Acc"],
        "Cost":         arow["Cost"],
        "run_combined":  arow["combined"],       # ← agent‐run text
        "file_combined": frow["combined_norm"],  # ← JSON file text
        "Date":         frow["date"],
        "run_id":       frow["run_id"],
        "matched_json": frow["path"],
        "similarity":   sims[best_j],
    })

out_df = pd.DataFrame(matches)
out_df.to_csv("upload_candidate_mapping.csv", index=False)
print("✅ Done! Originals are in evals_archive/, mapped files in evals_live/, and upload_candidate_mapping.csv is ready.")
