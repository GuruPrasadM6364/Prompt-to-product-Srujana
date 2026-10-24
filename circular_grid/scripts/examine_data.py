"""Examine all uploaded xlsx files to understand their structure."""
import sys
import os

# Find openpyxl
try:
    import openpyxl
    print("openpyxl available")
except ImportError:
    print("openpyxl NOT available")
    sys.exit(1)

try:
    import pandas as pd
    print(f"pandas {pd.__version__}")
except ImportError:
    print("pandas NOT available")
    sys.exit(1)

upload_dir = r"C:\Users\Guru Prasad M\.gemini\antigravity\brain\430a98f7-876c-4d4a-9647-91604f46ab99\.user_uploaded"

files = {
    "buildings": "media_1792821560942_512cf9d6.xlsx",
    "calendar": "media_1792821560942_231837c3.xlsx",
    "daily_summary": "media_1792821560944_41eb6649.xlsx",
    "hourly_streams": "media_1792821560946_8006d6d8.xlsx",
    "manual_logs": "media_1792821560948_725f035c.xlsx",
}

# Note: waste_events_train and waste_events_test not in the 5 files listed
# We have 5 files - need to figure out which is which

for name, fname in files.items():
    path = os.path.join(upload_dir, fname)
    print(f"\n{'='*60}")
    print(f"FILE: {name} ({fname})")
    print(f"Size: {os.path.getsize(path):,} bytes")
    try:
        xl = pd.ExcelFile(path)
        print(f"Sheets: {xl.sheet_names}")
        for sheet in xl.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet, nrows=5)
            print(f"\n  Sheet '{sheet}':")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Shape (5 rows): {df.shape}")
            print(f"  Dtypes: {dict(df.dtypes)}")
            # Full shape
            df_full = pd.read_excel(path, sheet_name=sheet)
            print(f"  Full shape: {df_full.shape}")
            print(f"  First row: {df_full.iloc[0].to_dict() if len(df_full) > 0 else 'empty'}")
    except Exception as e:
        print(f"  ERROR: {e}")
