import os
import ee
from src.data_fetcher import fetch_metrics
from src.chf_engine import CHFEngine

# ==========================================
# CONFIGURATION
# ==========================================

# 1. GEE Project & Authentication
GEE_PROJECT = 'nrscworks'  # Change if needed

# 2. Input Assets
SHAPEFILE_ASSET = 'projects/ee-odishagee/assets/Bundi_district_Village_boundary_withUUID'

# Dictionary mapping Year -> Crop Map Asset
CROP_MAPS = {
    2018: 'projects/ee-odishagee/assets/Bundi_supervised_cropmap_1soybean_2maize_3blackgram_4padddy_final',
    2019: 'projects/ee-odishagee/assets/Bundi_supervised_cropmap_1soybean_2maize_3blackgram_4padddy_final',
    2020: 'projects/ee-odishagee/assets/Bundi_supervised_cropmap_1soybean_2maize_3blackgram_4padddy_final',
    2023: 'projects/ee-odishagee/assets/Bundi_supervised_cropmap_1soybean_2maize_3blackgram_4padddy_final'
}

# 3. Analysis Parameters
TRAINING_YEARS = [2018, 2019, 2020]
ALL_YEARS = [2018, 2019, 2020, 2023]
TARGET_CROP_CLASS = 4  # e.g., 4 for Paddy

# 4. Season Dates (Kharif Example)
DATES_CONFIG = {
    'season_start': 'YYYY-06-01',  # YYYY will be replaced dynamically
    'season_end':   'YYYY-11-30',
    'peak_start':   'YYYY-08-01',
    'peak_end':     'YYYY-10-31'
}

# 5. Output Directories
RAW_DATA_DIR = 'outputs/raw_data'
MODEL_DIR = 'outputs/model'
RESULTS_DIR = 'outputs/results'

# 6. Execution Flags
RUN_PHASE_1_EXTRACTION = True
RUN_PHASE_2_TRAINING = True
RUN_PHASE_3_SCORING = True

# ==========================================
# MAIN EXECUTION
# ==========================================

def get_dates_for_year(year, config_template):
    """Replaces YYYY in the date config with the specific year."""
    return {k: v.replace('YYYY', str(year)) for k, v in config_template.items()}

def main():
    # Initialize Earth Engine
    try:
        ee.Initialize(project=GEE_PROJECT)
        print(f"Successfully initialized GEE with project: {GEE_PROJECT}")
    except Exception as e:
        print("Warning: explicit project init failed. Trying default.")
        ee.Initialize()

    # PHASE 1: DATA EXTRACTION
    if RUN_PHASE_1_EXTRACTION:
        print("\n--- PHASE 1: BATCH EXTRACTION ---")
        for year in ALL_YEARS:
            print(f"Processing Year: {year}")

            if year not in CROP_MAPS:
                print(f"Skipping {year}: No crop map defined.")
                continue

            dates = get_dates_for_year(year, DATES_CONFIG)

            fetch_metrics(
                year=year,
                crop_map_asset=CROP_MAPS[year],
                roi_asset_id=SHAPEFILE_ASSET,
                output_dir=RAW_DATA_DIR,
                dates_config=dates,
                target_crop_class=TARGET_CROP_CLASS,
                chunk_size=50  # Adjust based on timeout risk
            )

    # PHASE 2: WEIGHT TRAINING
    if RUN_PHASE_2_TRAINING:
        print("\n--- PHASE 2: WEIGHT TRAINING ---")
        try:
            CHFEngine.train_model(
                training_years=TRAINING_YEARS,
                input_dir=RAW_DATA_DIR,
                output_dir=MODEL_DIR
            )
        except Exception as e:
            print(f"Training failed: {e}")

    # PHASE 3: SCORING
    if RUN_PHASE_3_SCORING:
        print("\n--- PHASE 3: SCORING ---")
        try:
            CHFEngine.calculate_scores(
                years_list=ALL_YEARS,
                input_dir=RAW_DATA_DIR,
                model_dir=MODEL_DIR,
                output_dir=RESULTS_DIR
            )
        except Exception as e:
            print(f"Scoring failed: {e}")

if __name__ == "__main__":
    main()
