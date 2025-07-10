import sqlite3
import os
import re

DB_PATH = os.path.join("preprocessed_traces", "corebench_hard.db")

def ensure_model_column(cur):
    cur.execute("PRAGMA table_info(parsed_results)")
    cols = [row[1] for row in cur.fetchall()]
    if 'model_name' not in cols:
        cur.execute("ALTER TABLE parsed_results ADD COLUMN model_name TEXT")
        print("→ Added column: model_name")
    else:
        print("→ model_name column already exists")

def populate_model_column(cur):
    cur.execute("""
    UPDATE parsed_results
    SET model_name = (
        SELECT model_name
        FROM token_usage
        WHERE token_usage.benchmark_name = parsed_results.benchmark_name
          AND token_usage.agent_name     = parsed_results.agent_name
          AND token_usage.run_id         = parsed_results.run_id
        LIMIT 1
    )
    WHERE model_name IS NULL;
    """)
    print(f"→ Populated model_name ({cur.rowcount} rows)")

def rewrite_agent_names(cur):
    """
    For each row in parsed_results:
      - Strip any existing '(...)' suffix from agent_name
      - Append ' ({model_name})'
    """
    # Fetch all run_ids + current agent_name + model_name
    cur.execute("SELECT run_id, agent_name, model_name FROM parsed_results")
    rows = cur.fetchall()

    updated = 0
    for run_id, agent_name, model in rows:
        if not model:
            continue
        # Strip any existing parenthetical suffix
        base = re.sub(r'\s*\([^)]*\)\s*$', '', agent_name)
        new_name = f"{base} ({model})"
        if new_name != agent_name:
            cur.execute(
                "UPDATE parsed_results SET agent_name = ? WHERE run_id = ?",
                (new_name, run_id)
            )
            updated += 1

    print(f"→ Rewrote agent_name for {updated} rows")

def main():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"DB not found at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    ensure_model_column(cur)
    populate_model_column(cur)
    rewrite_agent_names(cur)

    conn.commit()
    conn.close()
    print("✅ corebench_hard agent_name rewrite complete.")

if __name__ == "__main__":
    main()
