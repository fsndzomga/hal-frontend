#!/usr/bin/env python3
import sqlite3
import csv
import glob
import os
import re
from datetime import datetime, timezone

DB_DIR   = "preprocessed_traces"
LOG_CSV  = "model_cleanup_log.csv"

MODELS_TO_REMOVE = [
    '2.5-pro',
    'o1',
    'o3-mini',
    'gpt-4o',
    'o3-2025-04-16 low',
    'claude-3-7-sonnet-2025-02-19 low',
    'claude-3-7-sonnet-20250219-thinking-low'
]

MODEL_MAP = {
    'claude-3-7-sonnet-20250219':               'claude-3-7-sonnet-2025-02-19',
    'claude-3-7-sonnet-20250219-thinking-high': 'claude-3-7-sonnet-2025-02-19 high',
    'claude-3-7-sonnet-20250219-thinking-low':  'claude-3-7-sonnet-2025-02-19 low',
    'o4-mini-2025-04-16-medium':                'o4-mini-2025-04-16',
    'o4-mini-2025-04-16_high_reasoning_effort': 'o4-mini-2025-04-16 high',
    'o4-mini-2025-04-16_low_reasoning_effort':  'o4-mini-2025-04-16 low',
    'claude-3-7-sonnet-20250219_thinking_high_4096': 'claude-3-7-sonnet-2025-02-19 high',
}

def extract_model(agent_name):
    m = re.search(r'\(([^)]+)\)\s*$', agent_name or "")
    return m.group(1) if m else None

with open(LOG_CSV, 'w', newline='') as csvfile:
    log_writer = csv.DictWriter(csvfile, fieldnames=[
        'timestamp','db_file','action','table',
        'run_id','old_value','new_value'
    ])
    log_writer.writeheader()

    for db_path in glob.glob(os.path.join(DB_DIR, "*.db")):
        db_name = os.path.basename(db_path)
        conn    = sqlite3.connect(db_path)
        cur     = conn.cursor()

        # 1) Deletion based on true agent model
        cur.execute("PRAGMA table_info(parsed_results)")
        cols = [r[1] for r in cur.fetchall()]
        has_model_col = 'model_name' in cols

        select_cols = "run_id, agent_name" + (", model_name" if has_model_col else "")
        cur.execute(f"SELECT {select_cols} FROM parsed_results")
        runs = cur.fetchall()

        to_drop = []
        for row in runs:
            run_id, agent_name = row[0], row[1]
            model = row[2] if has_model_col else extract_model(agent_name)
            if not model:
                continue
            ml = model.lower()
            if any(ml == pat.lower() or pat.lower() in ml for pat in MODELS_TO_REMOVE):
                to_drop.append((run_id, model))

        to_drop = list({(rid,m):None for rid,m in to_drop}.keys())
        for run_id, model in to_drop:
            ts = datetime.now(timezone.utc).isoformat()
            log_writer.writerow({
                'timestamp': ts,
                'db_file': db_name,
                'action': 'REMOVE_RUN',
                'table': 'parsed_results',
                'run_id': run_id,
                'old_value': model,
                'new_value': ''
            })
            cur.execute("DELETE FROM parsed_results WHERE run_id = ?", (run_id,))

        # 2) Renaming in parsed_results
        # 2a) If model_name column exists, rename there first
        if has_model_col:
            for old_model, new_model in MODEL_MAP.items():
                cur.execute(
                    "SELECT COUNT(*) FROM parsed_results WHERE model_name = ?",
                    (old_model,)
                )
                if cur.fetchone()[0] > 0:
                    ts = datetime.now(timezone.utc).isoformat()
                    log_writer.writerow({
                        'timestamp': ts,
                        'db_file': db_name,
                        'action': 'RENAME_MODEL',
                        'table': 'parsed_results',
                        'run_id': '',
                        'old_value': old_model,
                        'new_value': new_model
                    })
                    cur.execute("""
                        UPDATE parsed_results
                           SET model_name = ?
                         WHERE model_name = ?
                    """, (new_model, old_model))

        # 2b) Rewrite agent_name parentheses to reflect (possibly updated) model_name
        cur.execute("SELECT run_id, agent_name FROM parsed_results")
        for run_id, agent_name in cur.fetchall():
            model = extract_model(agent_name)
            # if we have model_name column, prefer that
            if has_model_col:
                cur.execute("SELECT model_name FROM parsed_results WHERE run_id = ?", (run_id,))
                model = cur.fetchone()[0]
            if not model or model not in MODEL_MAP.values() and model not in MODEL_MAP.keys():
                # still rewrite for any parenthetical change
                pass
            base = re.sub(r'\s*\([^)]*\)\s*$', '', agent_name)
            new_agent = f"{base} ({model})"
            if new_agent != agent_name:
                ts = datetime.now(timezone.utc).isoformat()
                log_writer.writerow({
                    'timestamp': ts,
                    'db_file': db_name,
                    'action': 'RENAME_AGENT',
                    'table': 'parsed_results',
                    'run_id': run_id,
                    'old_value': agent_name,
                    'new_value': new_agent
                })
                cur.execute("""
                    UPDATE parsed_results
                       SET agent_name = ?
                     WHERE run_id = ?
                """, (new_agent, run_id))

        conn.commit()
        conn.close()

print("✅ Cleanup complete — see", LOG_CSV)
