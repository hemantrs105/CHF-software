import ee
import geemap
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from src.gee_utils import GEEUtils

# Strict Schema Definition
CORE_COLUMNS = ['Unit_ID', 'Strata_ID']
BAND_NAMES = [
    'max_ndvi',
    'max_lswi',
    'max_backscatter',
    'integrated_backscatter',
    'integrated_fapar',
    'rainy_days',
    'adjusted_rainfall'
]

def fetch_metrics(year, crop_map_asset, roi_asset_id, output_dir, dates_config, target_crop_class=None, chunk_size=50):
    """
    Fetches metrics for a given year using Client-Side Chunking.

    Args:
        year (int): Year to process.
        crop_map_asset (str): GEE Asset ID for the crop map of that year.
        roi_asset_id (str): GEE Asset ID for the Insurance Units (Shapefile).
        output_dir (str): Directory to save the CSV.
        dates_config (dict): Dictionary containing 'season_start', 'season_end', 'peak_start', 'peak_end'.
        target_crop_class (int, optional): Specific class value to filter from the crop map.
        chunk_size (int): Number of units to process per batch.
    """

    # 1. Load ROI
    roi_collection = ee.FeatureCollection(roi_asset_id)

    # 2. Get list of Unit_IDs
    try:
        unit_ids = roi_collection.aggregate_array('Unit_ID').getInfo()
        unit_ids = sorted(list(set(unit_ids)))
    except Exception as e:
        print(f"Error fetching Unit_IDs: {e}")
        return

    print(f"Total Units to process for {year}: {len(unit_ids)}")

    # 3. Prepare Image
    full_roi_geom = roi_collection.geometry()

    image = GEEUtils.get_all_indicators(
        year=year,
        roi=full_roi_geom,
        crop_map_asset=crop_map_asset,
        dates_config=dates_config,
        target_crop_class=target_crop_class
    )

    # 4. Prepare Output CSV
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, f'indicators_{year}.csv')

    if os.path.exists(output_csv):
        os.remove(output_csv)

    # Build Expected Columns List
    # We expect mean and stdDev for every band
    metric_columns = []
    for band in BAND_NAMES:
        metric_columns.append(f"{band}_mean")
        metric_columns.append(f"{band}_stdDev")

    # Add calculated column
    final_columns = CORE_COLUMNS + metric_columns + ['condition_variability']

    # 5. Loop in Batches
    total_batches = (len(unit_ids) + chunk_size - 1) // chunk_size

    for i in tqdm(range(0, len(unit_ids), chunk_size), desc=f"Processing {year}", total=total_batches):
        batch_ids = unit_ids[i : i + chunk_size]

        # Filter ROI for this batch
        batch_fc = roi_collection.filter(ee.Filter.inList('Unit_ID', batch_ids))

        # Combined Reducer (Mean + StdDev for CV)
        # sharedInputs=True because we want both Mean and StdDev for each band of the input image.
        reducer = ee.Reducer.mean().combine(
            reducer2=ee.Reducer.stdDev(),
            sharedInputs=True
        )

        try:
            # reduceRegions
            stats = image.reduceRegions(
                collection=batch_fc,
                reducer=reducer,
                scale=20,
                tileScale=4
            )

            # Remove geometry
            stats_no_geo = stats.select(['.*'], None, False)

            # Fetch data
            df_batch = geemap.ee_to_df(stats_no_geo)

            if df_batch.empty:
                continue

            # Calculate Spatial CV
            # CV = StdDev / Mean (of Max NDVI)
            # Handle potential missing columns if image was empty or masked out
            if 'max_ndvi_stdDev' in df_batch.columns and 'max_ndvi_mean' in df_batch.columns:
                # Use numpy for safe division (handles division by zero -> inf, NaN -> NaN)
                df_batch['condition_variability'] = df_batch['max_ndvi_stdDev'].div(df_batch['max_ndvi_mean'])
                # Replace inf with NaN or 0?
                # If mean is 0, CV is undefined.
                # User asked to "handle NaNs gracefully".
                # Let's fill inf with NaN so it's consistent.
                df_batch['condition_variability'] = df_batch['condition_variability'].replace([np.inf, -np.inf], np.nan)
            else:
                df_batch['condition_variability'] = np.nan

            # STRICT SCHEMA ENFORCEMENT
            # This reorders columns and fills missing ones with NaN
            df_batch = df_batch.reindex(columns=final_columns)

            # Append to CSV
            header = not os.path.exists(output_csv)
            df_batch.to_csv(output_csv, mode='a', header=header, index=False)

        except Exception as e:
            print(f"Error in batch {i//chunk_size}: {e}")
            continue

    print(f"Finished processing {year}. Saved to {output_csv}")
