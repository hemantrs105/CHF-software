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

            strata_weights = {'Strata_ID': strata}

            # Temporary storage for Divergence to calculate final weights
            divergences = {}

            # 3. Process Each Indicator
            for indicator in CHFEngine.ALL_INDICATORS:
                # Handle Missing Data: Fill with mean
                if df_strata[indicator].isnull().any():
                     df_strata[indicator] = df_strata[indicator].fillna(df_strata[indicator].mean())

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
                total_norm = normalized.sum()
                if total_norm == 0:
                     divergences[indicator] = 0.0
                     continue

                probs = normalized / total_norm

                # Step C: Entropy (Ej)
                n = len(df_strata)
                if n <= 1:
                     divergences[indicator] = 0.0
                     continue

                k = 1 / np.log(n)

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

    @staticmethod
    def calculate_scores(years_list, input_dir, model_dir, output_dir):
        """
        Applies learned weights to calculate CHF scores for all specified years.

        Args:
            years_list (list): List of years (int) to score.
            input_dir (str): Directory containing indicators_{year}.csv.
            model_dir (str): Directory containing strata_weights.csv and scaling_factors.csv.
            output_dir (str): Directory to save final scores.
        """
        # Load Artifacts
        weights_path = os.path.join(model_dir, 'strata_weights.csv')
        scaling_path = os.path.join(model_dir, 'scaling_factors.csv')

        if not os.path.exists(weights_path) or not os.path.exists(scaling_path):
            raise FileNotFoundError("Model artifacts (weights/scaling) not found. Run training first.")

        weights_df = pd.read_csv(weights_path)
        scaling_df = pd.read_csv(scaling_path)

        results = []

        for year in years_list:
            file_path = os.path.join(input_dir, f'indicators_{year}.csv')
            if not os.path.exists(file_path):
                 print(f"Warning: Data for {year} not found. Skipping.")
                 continue

            df = pd.read_csv(file_path)

            # Iterate through strata present in this year's data
            for strata in df['Strata_ID'].unique():
                # Filter units for this strata
                df_strata = df[df['Strata_ID'] == strata].copy()

                # Get Model Metadata
                w_row = weights_df[weights_df['Strata_ID'] == strata]
                s_rows = scaling_df[scaling_df['Strata_ID'] == strata]

                if w_row.empty:
                    print(f"Warning: No weights found for Strata {strata}. Skipping units.")
                    continue

                if s_rows.empty:
                    print(f"Warning: No scaling factors found for Strata {strata}. Skipping units.")
                    continue

                # Initialize Scores
                chf_scores = 0.0

                for indicator in CHFEngine.ALL_INDICATORS:
                    # Get Min/Max/Weight
                    s_row = s_rows[s_rows['Indicator'] == indicator]
                    if s_row.empty:
                        # Should not happen if model is complete
                        continue

                    min_val = s_row.iloc[0]['Min']
                    max_val = s_row.iloc[0]['Max']
                    weight = w_row.iloc[0][indicator]

                    # Skip if weight is 0 to save computation
                    if weight == 0:
                        continue

                    # Handle Missing Data in Application Phase
                    # If data is missing for a unit, we can fill with Historic Mean?
                    # Or current Strata Mean?
                    # Using current strata mean is safer for application.
                    if df_strata[indicator].isnull().any():
                        df_strata[indicator] = df_strata[indicator].fillna(df_strata[indicator].mean())

                    series = df_strata[indicator]

                    # Normalize
                    # Safety check for Zero Variance (Min == Max)
                    if max_val == min_val:
                        normalized = 0.0
                    else:
                        if indicator in CHFEngine.POSITIVE_INDICATORS:
                            normalized = (series - min_val) / (max_val - min_val)
                        else: # Negative
                            normalized = (max_val - series) / (max_val - min_val)

                    # Add weighted component
                    chf_scores += (normalized * weight)

                df_strata['CHF_Score'] = chf_scores
                # Keep relevant columns
                results.append(df_strata[['Unit_ID', 'Strata_ID', 'CHF_Score']].assign(Year=year))

        # Save Final Results
        if results:
            final_df = pd.concat(results, ignore_index=True)
            # Reorder columns
            final_df = final_df[['Year', 'Unit_ID', 'Strata_ID', 'CHF_Score']]

            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'chf_scores_all_years.csv')
            final_df.to_csv(output_path, index=False)
            print(f"Scoring complete. Results saved to: {output_path}")
        else:
            print("No scores calculated.")
