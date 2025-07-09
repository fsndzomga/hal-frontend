import sqlite3
import csv
import glob
import os
from datetime import datetime, timezone

# Folder containing all the .db files
DB_DIR = "preprocessed_traces"

# File to log actions
LOG_CSV = "model_cleanup_log.csv"

# 1) List of models to remove
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
}

with open(LOG_CSV, 'w', newline='') as csvfile:
    log_writer = csv.DictWriter(csvfile, fieldnames=[
        'timestamp', 'db_file', 'action', 'table',
        'run_id', 'old_value', 'new_value'
    ])
    log_writer.writeheader()

    # Go through each db
    for db_path in glob.glob(os.path.join(DB_DIR, "*.db")):
        bench = os.path.basename(db_path)
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()

        # Identify run_ids to remove
        placeholders = ",".join("?" for _ in MODELS_TO_REMOVE)
        cur.execute(f"""
            SELECT DISTINCT run_id, model_name 
            FROM token_usage 
            WHERE model_name IN ({placeholders})
        """, MODELS_TO_REMOVE)
        runs_to_drop = cur.fetchall()

        for run_id, model_name in runs_to_drop:
            # log
            log_writer.writerow({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'db_file': bench,
                'action': 'REMOVE_RUN',
                'table': 'parsed_results',
                'run_id': run_id,
                'old_value': model_name,
                'new_value': ''
            })

            cur.execute("DELETE FROM parsed_results WHERE run_id = ?", (run_id,))

            cur.execute("DELETE FROM preprocessed_traces WHERE run_id = ?", (run_id,))

            cur.execute("DELETE FROM token_usage WHERE run_id = ?", (run_id,))

        # Renaming models
        for old_model, new_model in MODEL_MAP.items():
            cur.execute("SELECT COUNT(*) FROM token_usage WHERE model_name = ?", (old_model,))
            count = cur.fetchone()[0]
            if count:
                # log
                log_writer.writerow({
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'db_file': bench,
                    'action': 'RENAME_MODEL',
                    'table': 'token_usage',
                    'run_id': '',
                    'old_value': old_model,
                    'new_value': new_model
                })
                # update
                cur.execute("""
                    UPDATE token_usage 
                    SET model_name = ? 
                    WHERE model_name = ?
                """, (new_model, old_model))

        def rename_agent(table):
            for old_model, new_model in MODEL_MAP.items():
                # search agent_name containing "(old_model)"
                pattern = f"%({old_model})%"
                cur.execute(f"""
                    SELECT DISTINCT agent_name 
                    FROM {table} 
                    WHERE agent_name LIKE ?
                """, (pattern,))
                rows = cur.fetchall()
                for (agent_name,) in rows:
                    new_agent_name = agent_name.replace(f"({old_model})", f"({new_model})")
                    # log
                    log_writer.writerow({
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'db_file': bench,
                        'action': 'RENAME_AGENT',
                        'table': table,
                        'run_id': '',
                        'old_value': agent_name,
                        'new_value': new_agent_name
                    })
                    # update
                    cur.execute(f"""
                        UPDATE {table}
                        SET agent_name = ?
                        WHERE agent_name = ?
                    """, (new_agent_name, agent_name))

        for tbl in ['parsed_results', 'preprocessed_traces', 'token_usage']:
            rename_agent(tbl)

        conn.commit()
        conn.close()

print(f"✅ Cleanup complete, see {LOG_CSV} for the action log.")
