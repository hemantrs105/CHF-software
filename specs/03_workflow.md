# Execution Workflow
1. Load Shapefile of Insurance Units (IUs).
2. For each IU, fetch satellite time-series (lazy loading via Dask).
3. Compute the 8 metrics defined in the research paper.
4. Group IUs by "Homogeneous Strata" (e.g., Soil Type).
5. Apply Entropy Weighting per group.
6. Export CSV with CHF values.
