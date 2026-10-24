"""
Quick deeper audit of what's in each file -- check for waste events embedded.
"""
import pandas as pd
import numpy as np
from pathlib import Path

upload_dir = Path(r"C:\Users\Guru Prasad M\.gemini\antigravity\brain\430a98f7-876c-4d4a-9647-91604f46ab99\.user_uploaded")

files = {
    "buildings":      "media_1792821560942_512cf9d6.xlsx",
    "calendar":       "media_1792821560942_231837c3.xlsx",
    "daily_summary":  "media_1792821560944_41eb6649.xlsx",
    "hourly_streams": "media_1792821560946_8006d6d8.xlsx",
    "manual_logs":    "media_1792821560948_725f035c.xlsx",
}

# Check all sheets in each file
for name, fname in files.items():
    path = upload_dir / fname
    xl = pd.ExcelFile(path)
    print(f"\n{name}: sheets={xl.sheet_names}")
    for sh in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sh)
        print(f"  '{sh}': shape={df.shape}, cols={list(df.columns)}")

print("\n\n=== DETAILED DATA CHECKS ===")

# Hourly streams details
df_h = pd.read_excel(upload_dir / files["hourly_streams"])
df_h.columns = [c.lower() for c in df_h.columns]
print(f"\nhourly_streams: {df_h.shape}")
print(f"  timestamp range: {df_h['timestamp'].min()} to {df_h['timestamp'].max()}")
print(f"  buildings: {sorted(df_h['building_id'].unique())}")
print(f"  duplicates (ts+bid): {df_h.duplicated(['timestamp','building_id']).sum()}")
for col in ['electricity_kwh','water_litres','solar_generation_kwh','footfall']:
    print(f"  missing {col}: {df_h[col].isna().sum()}")

# Manual logs building aliases
df_m = pd.read_excel(upload_dir / files["manual_logs"])
print(f"\nmanual_logs: {df_m.shape}")
print(f"  building unique values: {sorted(df_m['building'].dropna().unique())}")
print(f"  streams: {df_m['stream'].unique()}")
print(f"  units: {df_m['unit'].dropna().unique()}")

# Calendar day types
df_c = pd.read_excel(upload_dir / files["calendar"])
print(f"\ncalendar: {df_c.shape}")
print(f"  day_type values: {df_c['day_type'].value_counts().to_dict()}")
print(f"  distinct events: {df_c['event'].dropna().nunique()}")
print(f"  temp range: {df_c['avg_temp_c'].min():.1f} to {df_c['avg_temp_c'].max():.1f}")

# Buildings
df_b = pd.read_excel(upload_dir / files["buildings"])
print(f"\nbuildings: {df_b.shape}")
print(df_b.to_string())

# Daily summary
df_d = pd.read_excel(upload_dir / files["daily_summary"])
print(f"\ndaily_summary: {df_d.shape}")
print(f"  Missing counts: {df_d.isna().sum().to_dict()}")
print(f"  date range: {df_d['date'].min()} to {df_d['date'].max()}")
print(f"  buildings: {sorted(df_d['building_id'].unique())}")
