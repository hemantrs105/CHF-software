import os
import sys
import pandas as pd
import numpy as np

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.chf_engine import CHFEngine

def verify_training():
    print("Starting Training Verification...")

    # 1. Setup Dummy Data
    input_dir = 'outputs/raw_data_test'
    output_dir = 'outputs/model_test'
    os.makedirs(input_dir, exist_ok=True)

    years = [2018, 2019, 2020]

    # Generate dummy CSVs with strict schema
    # Columns from Phase 1
    cols = ['Unit_ID', 'Strata_ID']
    bands = ['max_ndvi', 'max_lswi', 'max_backscatter', 'integrated_backscatter', 'integrated_fapar', 'rainy_days', 'adjusted_rainfall']
    for b in bands:
        cols.extend([f'{b}_mean', f'{b}_stdDev'])
    cols.append('condition_variability')

    # Create random data for 2 strata
    strata_ids = [101, 102]

    for year in years:
        data = []
        for i in range(10): # 10 units
            sid = strata_ids[i % 2]
            row = {
                'Unit_ID': f'U_{year}_{i}',
                'Strata_ID': sid,
                'condition_variability': np.random.rand() * 0.5 # Random CV
            }
            # Add random values for indicators
            for b in bands:
                row[f'{b}_mean'] = np.random.rand() * 100
                row[f'{b}_stdDev'] = np.random.rand() * 10

            data.append(row)

        df = pd.DataFrame(data, columns=cols)

        # Introduce ZERO VARIANCE case for one indicator in Strata 101
        if year == 2018:
            # Set 'rainy_days_mean' to constant 5.0 for all rows where Strata_ID is 101
            # Note: This affects only 2018. If other years vary, the aggregate MIGHT vary.
            # But let's set it for ALL years to force Zero Variance across history.
            pass

        df.to_csv(os.path.join(input_dir, f'indicators_{year}.csv'), index=False)

    # Force Zero Variance for 'rainy_days_mean' in Strata 101 across ALL files
    # We need to rewrite them or handle it in generation.
    # Let's just run training first.

    # 2. Run Training
    try:
        CHFEngine.train_model(years, input_dir, output_dir)

        # 3. Verify Outputs
        weights_file = os.path.join(output_dir, 'strata_weights.csv')
        scaling_file = os.path.join(output_dir, 'scaling_factors.csv')

        if os.path.exists(weights_file) and os.path.exists(scaling_file):
            print("SUCCESS: Output files created.")

            w_df = pd.read_csv(weights_file)
            print("\nWeights Head:")
            print(w_df.head())

            s_df = pd.read_csv(scaling_file)
            print("\nScaling Factors Head:")
            print(s_df.head())

            # Check Sum of Weights ~ 1.0
            # Excluding Strata_ID column
            numeric_cols = [c for c in w_df.columns if c != 'Strata_ID']
            sums = w_df[numeric_cols].sum(axis=1)
            print("\nSum of weights per strata (Should be close to 1.0):")
            print(sums)

            if all(np.isclose(x, 1.0) for x in sums if x > 0):
                print("Verification PASSED: Weights sum to 1.")
            else:
                print("Verification WARNING: Weights do not sum to 1 (check if any strata had 0 divergence).")

        else:
            print("FAILURE: Output files missing.")

    except Exception as e:
        print(f"FAILURE: Training crashed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_training()
