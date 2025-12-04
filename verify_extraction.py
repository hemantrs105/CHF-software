import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_fetcher import fetch_metrics
import ee

def verify_extraction():
    # 1. Initialize GEE
    try:
        ee.Initialize(project='nrscworks')
    except Exception:
        print("Warning: explicit project 'nrscworks' init failed. Trying default.")
        try:
            ee.Initialize()
        except Exception as e:
            print("GEE Authentication failed. Please run `earthengine authenticate`.")
            print(e)
            return

    # 2. Define Test Inputs
    roi_asset_id = 'projects/ee-odishagee/assets/Bundi_district_Village_boundary_withUUID'

    # Years to test: Just testing one year (2018)
    year = 2018
    crop_map_asset = 'projects/ee-odishagee/assets/Bundi_supervised_cropmap_1soybean_2maize_3blackgram_4padddy_final'

    # NEW: Dates Config (Kharif Season approx)
    dates_config = {
        'season_start': f'{year}-06-01',
        'season_end': f'{year}-11-30',
        'peak_start': f'{year}-08-01',
        'peak_end': f'{year}-10-31'
    }

    # NEW: Target Crop Class (assuming 4 = Paddy based on asset name)
    # The asset name is "...1soybean_2maize_3blackgram_4padddy_final"
    # So Paddy is likely 4.
    target_crop_class = 4

    output_dir = 'outputs/raw_data'

    print("Starting verification extraction...")
    print(f"Year: {year}")
    print(f"Dates: {dates_config}")
    print(f"Target Class: {target_crop_class}")

    fetch_metrics(
        year=year,
        crop_map_asset=crop_map_asset,
        roi_asset_id=roi_asset_id,
        output_dir=output_dir,
        dates_config=dates_config,
        target_crop_class=target_crop_class,
        chunk_size=10
    )

    # Verify Output
    expected_file = os.path.join(output_dir, f'indicators_{year}.csv')
    if os.path.exists(expected_file):
        print(f"SUCCESS: File created at {expected_file}")
        import pandas as pd
        df = pd.read_csv(expected_file)
        print("Columns found:", df.columns.tolist())
        print("Rows:", len(df))
    else:
        print("FAILURE: Output file not found.")

if __name__ == "__main__":
    verify_extraction()
