import sqlite3
from pprint import pprint

DB_PATH = "data/oltp/events.db"

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # List tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print("Tables:", cur.fetchall())

    # Show schema for events
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='events';")
    print("Schema for events:")
    print(cur.fetchone()[0])

    # Show first 10 rows
    cur.execute("SELECT * FROM events LIMIT 10;")
    rows = cur.fetchall()
    print(f"\nFirst {len(rows)} rows:")
    pprint(rows)

if __name__ == "__main__":
    main()