"""
Compatibility wrapper for the old Qwen-only transfer update script.

The transfer analysis is now rebuilt from scratch for all models by
src/rebuild_transfer_analysis.py. Keeping this wrapper prevents accidental
append-style updates to transfer_records.csv.
"""

from __future__ import annotations

from rebuild_transfer_analysis import main


if __name__ == "__main__":
    print("Deprecated: rebuilding all transfer analysis instead of appending Qwen-only rows.")
    main()
