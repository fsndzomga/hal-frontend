# import sqlite3
# import pandas as pd

# conn = sqlite3.connect('/workspaces/hal-frontend/preprocessed_traces/assistantbench.db')
# cursor = conn.cursor()
# cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# tables = [row[0] for row in cursor.fetchall()]
# for table in tables:
#     print(f"\n=== {table} ===")
#     df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 3;", conn)
#     print(df)
# conn.close()

import sqlite3
import pandas as pd

db_path = '/workspaces/hal-frontend/preprocessed_traces/assistantbench.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]

for table in tables:
    print(f"Exporting table: {table}")
    df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
    print(df.head(5))
    # Use table name for CSV file
    # csv_path = f'{table}.csv'
    # df.to_csv(csv_path, index=False)
    # print(f"Saved {csv_path} ({len(df)} rows)")

conn.close()

