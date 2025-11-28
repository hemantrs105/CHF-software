# Mathematical Logic (Entropy Weighting)
1. **Normalization:**
   - If parameter is positive for yield (NDVI, LSWI): Use Min-Max normalization.
   - If parameter is negative for yield (Condition Variability): Use Inverted Min-Max.

2. **Entropy Calculation:**
   - Input: Matrix X (Rows=Insurance Units, Cols=Parameters).
   - Step A: Calculate Probability Matrix P_ij = X_ij / sum(X_ij).
   - Step B: Calculate Entropy E_j = -k * sum(P_ij * ln(P_ij)).
   - Step C: Calculate Weights W_j = (1 - E_j) / sum(1 - E_j).

3. **CHF Score:**
   - CHF_i = Sum(Weight_j * Normalized_Value_ij).
