import pandas as pd
import numpy as np
import os

class CHFEngine:
    # Define Indicators and their types
    POSITIVE_INDICATORS = [
        'max_ndvi_mean',
        'max_lswi_mean',
        'max_backscatter_mean',
        'integrated_backscatter_mean',
        'integrated_fapar_mean',
        'rainy_days_mean',
        'adjusted_rainfall_mean'
    ]

    NEGATIVE_INDICATORS = [
        'condition_variability'
    ]

    ALL_INDICATORS = POSITIVE_INDICATORS + NEGATIVE_INDICATORS

    @staticmethod
    def train_model(training_years, input_dir, output_dir):
        """
        Loads historic data, learns entropy weights, and saves model artifacts.

        Args:
            training_years (list): List of years (int) to use for training.
            input_dir (str): Directory containing indicators_{year}.csv.
            output_dir (str): Directory to save model outputs.
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. Load Data
        df_list = []
        for year in training_years:
            filepath = os.path.join(input_dir, f'indicators_{year}.csv')
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                df['Year'] = year
                df_list.append(df)
            else:
                print(f"Warning: Data for {year} not found at {filepath}")

        if not df_list:
            raise ValueError("No training data found.")

        df_history = pd.concat(df_list, ignore_index=True)

        # 2. Group by Strata
        # We need to calculate weights PER STRATA.
        unique_strata = df_history['Strata_ID'].unique()

        weights_list = []
        scaling_list = []

        for strata in unique_strata:
            df_strata = df_history[df_history['Strata_ID'] == strata].copy()

            # Skip if not enough data? (e.g. 1 row can't calc entropy effectively but math works)

            strata_weights = {'Strata_ID': strata}

            # Temporary storage for Divergence to calculate final weights
            divergences = {}

            # 3. Process Each Indicator
            for indicator in CHFEngine.ALL_INDICATORS:
                # Handle Missing Data: Fill with mean or drop?
                # Entropy requires complete data. Let's drop rows with NaN for this indicator
                # OR fill with mean of the strata.
                # PRD says "Gracefully handle NaNs" earlier.
                # Let's drop NaNs for the calculation to be safe, or fill.
                # Filling with mean is safer to preserve sample size.
                if df_strata[indicator].isnull().any():
                     df_strata[indicator].fillna(df_strata[indicator].mean(), inplace=True)

                series = df_strata[indicator]

                # Step A: Min/Max
                min_val = series.min()
                max_val = series.max()

                # Save Scaling Factor
                scaling_list.append({
                    'Strata_ID': strata,
                    'Indicator': indicator,
                    'Min': min_val,
                    'Max': max_val
                })

                # Zero Variance Check
                if max_val == min_val:
                    # Edge Case: Zero Variance -> Weight = 0
                    divergences[indicator] = 0.0
                    continue

                # Normalize
                if indicator in CHFEngine.POSITIVE_INDICATORS:
                    normalized = (series - min_val) / (max_val - min_val)
                else: # Negative
                    normalized = (max_val - series) / (max_val - min_val)

                # Step B: Probability (Pij)
                # Pij = xij / sum(xij)
                # If sum is 0 (all xij are 0), then Pij is 0 (or undefined).
                # If normalized values are all 0, it means all original values were Min (Zero Variance case handled above).
                # But individual pixels could be min.

                total_norm = normalized.sum()
                if total_norm == 0:
                     # Should theoretically be covered by Zero Variance if all are 0?
                     # Not necessarily. e.g. [0, 0, 0] -> Min=0, Max=0 -> Zero Variance.
                     # [0, 1] -> Min=0, Max=1. Norm -> [0, 1]. Sum=1. P -> [0, 1].
                     divergences[indicator] = 0.0 # Fallback
                     continue

                probs = normalized / total_norm

                # Step C: Entropy (Ej)
                # Ej = -k * sum(P * ln(P))
                # k = 1 / ln(n) where n is number of samples (rows)
                n = len(df_strata)
                if n <= 1:
                     # Only 1 sample -> Entropy is 0? Or 1?
                     # If n=1, k is undefined (1/0).
                     # We cannot calculate entropy with 1 sample.
                     # Default weight to 0 or equal?
                     divergences[indicator] = 0.0
                     continue

                k = 1 / np.log(n)

                # Handle log(0) -> 0 in entropy calculation logic (lim x->0 of x*ln(x) = 0)
                # mask 0 probabilities
                valid_probs = probs[probs > 0]
                entropy_sum = np.sum(valid_probs * np.log(valid_probs))

                ej = -k * entropy_sum

                # Step D: Divergence (Dj)
                dj = 1 - ej
                divergences[indicator] = dj

            # Step E: Final Weights (wj)
            total_divergence = sum(divergences.values())

            for indicator in CHFEngine.ALL_INDICATORS:
                if total_divergence == 0:
                    # If all indicators have 0 variance or divergence, equal weights? Or 0?
                    # Let's set equal weights as fallback or 0.
                    w = 0.0
                else:
                    w = divergences.get(indicator, 0.0) / total_divergence

                strata_weights[indicator] = w

            weights_list.append(strata_weights)

        # 4. Save Outputs
        df_weights = pd.DataFrame(weights_list)
        df_scaling = pd.DataFrame(scaling_list)

        weights_path = os.path.join(output_dir, 'strata_weights.csv')
        scaling_path = os.path.join(output_dir, 'scaling_factors.csv')

        df_weights.to_csv(weights_path, index=False)
        df_scaling.to_csv(scaling_path, index=False)

        print(f"Model trained successfully.")
        print(f"Weights saved to: {weights_path}")
        print(f"Scaling factors saved to: {scaling_path}")
