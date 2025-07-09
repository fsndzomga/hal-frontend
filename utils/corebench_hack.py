"""
Script to add and populate `model_name` column in `parsed_results`
of the corebench_hard SQLite database, based on token_usage table.
"""
import sqlite3
import os

# Path to the corebench_hard database
DB_PATH = os.path.join("preprocessed_traces", "corebench_hard.db")


def main(db_path):
    """Connects to the SQLite DB, adds the model_name column if missing, and populates it."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1) Check existing columns in parsed_results
    cur.execute("PRAGMA table_info(parsed_results)")
    columns = [row[1] for row in cur.fetchall()]

    # 2) Add model_name column if it doesn't exist
    if 'model_name' not in columns:
        cur.execute("ALTER TABLE parsed_results ADD COLUMN model_name TEXT")
        print("Added column: model_name")
    else:
        print("Column 'model_name' already exists")

    # 3) Populate model_name from token_usage
    update_sql = """
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
    """
    cur.execute(update_sql)
    updated_count = conn.total_changes
    print(f"Updated model_name for {updated_count} rows")

    # 4) Verify how many rows remain with NULL model_name
    cur.execute("SELECT COUNT(*) FROM parsed_results WHERE model_name IS NULL")
    missing_count = cur.fetchone()[0]
    print(f"Rows still missing model_name: {missing_count}")

    # Commit and close
    conn.commit()
    conn.close()


if __name__ == '__main__':
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    main(DB_PATH)
