import sqlite3
import csv
import glob
import os
from datetime import datetime, timezone

# Folder containing all the .db files
DB_DIR = "preprocessed_traces"

# File to log all actions
LOG_CSV = "model_cleanup_log.csv"

# 1) List of models to remove (exact match or substring)
MODELS_TO_REMOVE = [
    '2.5-pro',
    'o1',
    'o3-mini',
    'gpt-4o',
    'o3-2025-04-16 low',
    'claude-3-7-sonnet-2025-02-19 low',
    'claude-3-7-sonnet-20250219-thinking-low'
]

# 2) Mapping to rename some models
MODEL_MAP = {
    'claude-3-7-sonnet-20250219':               'claude-3-7-sonnet-2025-02-19',
    'claude-3-7-sonnet-20250219-thinking-high': 'claude-3-7-sonnet-2025-02-19 high',
    'claude-3-7-sonnet-20250219-thinking-low':  'claude-3-7-sonnet-2025-02-19 low',
    'o4-mini-2025-04-16-medium':                'o4-mini-2025-04-16',
    'o4-mini-2025-04-16_high_reasoning_effort': 'o4-mini-2025-04-16 high',
    'o4-mini-2025-04-16_low_reasoning_effort': 'o4-mini-2025-04-16 low',
    'claude-3-7-sonnet-20250219_thinking_high_4096': 'claude-3-7-sonnet-2025-02-19 high',
}

with open(LOG_CSV, 'w', newline='') as csvfile:
    log_writer = csv.DictWriter(csvfile, fieldnames=[
        'timestamp', 'db_file', 'action', 'table',
        'run_id', 'old_value', 'new_value'
    ])
    log_writer.writeheader()

    # Iterate over each .db file
    for db_path in glob.glob(os.path.join(DB_DIR, "*.db")):
        db_name = os.path.basename(db_path)
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()

        # --- 1) Identify and remove runs for unwanted models ---
        runs_to_drop = []

        # a) Exact matches via IN
        placeholders = ",".join("?" for _ in MODELS_TO_REMOVE)
        cur.execute(f"""
            SELECT DISTINCT run_id, model_name
            FROM token_usage
            WHERE model_name IN ({placeholders})
        """, MODELS_TO_REMOVE)
        runs_to_drop.extend(cur.fetchall())

        # b) Substring matches via LIKE
        for pattern in MODELS_TO_REMOVE:
            cur.execute("""
                SELECT DISTINCT run_id, model_name
                FROM token_usage
                WHERE model_name LIKE ?
            """, (f"%{pattern}%",))
            runs_to_drop.extend(cur.fetchall())

        # Deduplicate runs
        runs_to_drop = list({(run_id, model): None for run_id, model in runs_to_drop}.keys())

        for run_id, model_name in runs_to_drop:
            ts = datetime.now(timezone.utc).isoformat()
            # Log the removal
            log_writer.writerow({
                'timestamp': ts,
                'db_file': db_name,
                'action': 'REMOVE_RUN',
                'table': 'parsed_results',
                'run_id': run_id,
                'old_value': model_name,
                'new_value': ''
            })
            # Delete from parsed_results
            cur.execute("DELETE FROM parsed_results WHERE run_id = ?", (run_id,))
            # Delete from preprocessed_traces
            cur.execute("DELETE FROM preprocessed_traces WHERE run_id = ?", (run_id,))
            # Delete from token_usage
            cur.execute("DELETE FROM token_usage WHERE run_id = ?", (run_id,))

        # --- 2) Rename model_name in token_usage ---
        for old_model, new_model in MODEL_MAP.items():
            cur.execute(
                "SELECT COUNT(*) FROM token_usage WHERE model_name = ?",
                (old_model,)
            )
            if cur.fetchone()[0] > 0:
                ts = datetime.now(timezone.utc).isoformat()
                log_writer.writerow({
                    'timestamp': ts,
                    'db_file': db_name,
                    'action': 'RENAME_MODEL',
                    'table': 'token_usage',
                    'run_id': '',
                    'old_value': old_model,
                    'new_value': new_model
                })
                cur.execute("""
                    UPDATE token_usage
                       SET model_name = ?
                     WHERE model_name = ?
                """, (new_model, old_model))

        # --- 3) Rename agent_name in all tables ---
        def rename_agent_in_table(table_name):
            for old_model, new_model in MODEL_MAP.items():
                like_pattern = f"%({old_model})%"
                cur.execute(f"""
                    SELECT DISTINCT agent_name
                    FROM {table_name}
                    WHERE agent_name LIKE ?
                """, (like_pattern,))
                for (agent_name,) in cur.fetchall():
                    new_agent_name = agent_name.replace(
                        f"({old_model})",
                        f"({new_model})"
                    )
                    ts = datetime.now(timezone.utc).isoformat()
                    log_writer.writerow({
                        'timestamp': ts,
                        'db_file': db_name,
                        'action': 'RENAME_AGENT',
                        'table': table_name,
                        'run_id': '',
                        'old_value': agent_name,
                        'new_value': new_agent_name
                    })
                    cur.execute(f"""
                        UPDATE {table_name}
                           SET agent_name = ?
                         WHERE agent_name = ?
                    """, (new_agent_name, agent_name))

        for tbl in ('parsed_results', 'preprocessed_traces', 'token_usage'):
            rename_agent_in_table(tbl)

        conn.commit()
        conn.close()

print(f"✅ Cleanup complete — see '{LOG_CSV}' for the full action log.")
