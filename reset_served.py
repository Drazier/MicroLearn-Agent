"""
reset_served.py — Demo utility: marks all served papers back to unserved
so discovery can re-surface them without re-running populate.

Usage:
    python reset_served.py            # reset all subjects
    python reset_served.py psychology neuroscience  # reset specific subjects
"""
import sys
from populate_db import get_conn, init_db, VALID_SUBJECTS

def reset_served(subjects: list[str] | None = None) -> None:
    init_db()
    conn = get_conn()
    targets = subjects or VALID_SUBJECTS

    for subject in targets:
        result = conn.execute(
            "UPDATE papers SET served=0, date_served=NULL WHERE subject=? AND served=1 AND in_reading_list=0 AND saved=0",
            (subject,)
        )
        print(f"[reset] {subject}: {result.rowcount} papers marked unserved")

    conn.commit()
    conn.close()
    print("\nDone. Run the app and press Show Cards to rediscover.")

if __name__ == "__main__":
    subjects = sys.argv[1:] or None
    if subjects:
        invalid = [s for s in subjects if s not in VALID_SUBJECTS]
        if invalid:
            print(f"Unknown subjects: {invalid}")
            print(f"Valid: {VALID_SUBJECTS}")
            sys.exit(1)
    reset_served(subjects)
