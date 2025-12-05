# CHF-Automator

**CHF-Automator** is a Python tool designed to automate the **Crop Health Factor (CHF)** insurance model using **Google Earth Engine (GEE)**. It implements the "N+1" strategy, extracting satellite-based crop indicators for historical years plus the current assessment year, learning weights from historical variance (Entropy Method), and generating final risk scores.

## Key Features

*   **N+1 Strategy:** Supports dynamic inputs where every year utilizes a specific **Crop Map Asset**.
*   **Decoupled Architecture:** Separates heavy GEE data extraction (Phase 1) from mathematical modeling (Phase 2 & 3).
*   **Robust Extraction:** Uses **Client-Side Chunking** to fetch data in batches, preventing GEE timeouts on large datasets.
*   **Entropy Weighting:** Automatically calculates weights for each strata based on historical data variance using the Entropy Weight Method (EWM).
*   **Strict Schema:** Enforces consistent CSV output schemas to prevent data misalignment.

## Project Structure

```text
CHF-software/
├── inputs/                # Place for local input files (if any)
├── outputs/
│   ├── raw_data/          # Phase 1: Extracted GEE indicators (CSVs)
│   ├── model/             # Phase 2: Learned Weights & Scaling Factors
│   └── results/           # Phase 3: Final CHF Scores
├── src/
│   ├── gee_utils.py       # GEE logic (Band math, Cloud Masking, etc.)
│   ├── data_fetcher.py    # Batch extraction logic
│   └── chf_engine.py      # Entropy Model & Scoring Engine
├── main.py                # Orchestration script (Configuration & Execution)
├── requirements.txt       # Python dependencies
└── CHF_technical_document.ipynb # Absolute Source of Truth (PRD)
```

## Prerequisites

1.  **Google Earth Engine Account:** You must have a GEE account and project enabled.
2.  **Python 3.9+**

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Authenticate with Google Earth Engine:
    ```bash
    earthengine authenticate
    ```

## Usage

The entire workflow is orchestrated by `main.py`.

### 1. Configure the Project
Open `main.py` and update the **Configuration Section**:

*   **GEE_PROJECT:** Your GEE Project ID (e.g., `'nrscworks'`).
*   **SHAPEFILE_ASSET:** GEE Asset ID for your Insurance Units (must contain `Unit_ID` and `Strata_ID`).
*   **CROP_MAPS:** Dictionary mapping each `Year` to its specific Crop Map Asset ID.
*   **TRAINING_YEARS:** List of years to use for learning weights (e.g., `[2018, 2019, 2020]`).
*   **ALL_YEARS:** List of all years to process (Historical + Current).
*   **DATES_CONFIG:** Season dates (automatically updated for each year).

### 2. Run the Tool
Execute the main script:
```bash
python main.py
```

You can toggle specific phases on/off using the flags in `main.py`:
```python
RUN_PHASE_1_EXTRACTION = True
RUN_PHASE_2_TRAINING = True
RUN_PHASE_3_SCORING = True
```

## Methodology

### Phase 1: Batch Extraction
*   **Input:** Shapefile + Crop Maps + Satellite Data (Sentinel-2, Sentinel-1, MODIS, CHIRPS).
*   **Process:** Iterates through units in batches (default 50), applies cloud masks (`MSK_CLDPRB`), filters for specific crop pixels, and calculates 8 indicators (NDVI, LSWI, Backscatter, Rainfall, etc.).
*   **Output:** `outputs/raw_data/indicators_{YEAR}.csv`

### Phase 2: Weight Training
*   **Input:** Historical CSVs (defined in `TRAINING_YEARS`).
*   **Process:**
    1. Groups data by `Strata_ID`.
    2. Calculates **Scaling Factors** (Min/Max) for normalization.
    3. Calculates **Entropy Weights** based on data variance.
    4. Handles "Zero Variance" edge cases (Weight = 0).
*   **Output:** `outputs/model/strata_weights.csv`, `outputs/model/scaling_factors.csv`

### Phase 3: Scoring
*   **Input:** All CSVs + Learned Model (Weights & Factors).
*   **Process:**
    1. Normalizes data using **Historical** Min/Max factors.
    2. Applies learned Weights to calculate the weighted sum.
*   **Output:** `outputs/results/chf_scores_all_years.csv`

## Indicators

1.  **Max NDVI** (Sentinel-2)
2.  **Max LSWI** (Sentinel-2)
3.  **Max Backscatter** (Sentinel-1 VH)
4.  **Integrated Backscatter** (Sentinel-1 VH Sum)
5.  **Integrated FAPAR** (MODIS)
6.  **Condition Variability** (Spatial CV of Max NDVI) - *Negative Indicator*
7.  **Rainy Days** (CHIRPS)
8.  **Adjusted Rainfall** (CHIRPS vs 10-yr Normal)
