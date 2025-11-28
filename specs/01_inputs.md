# Data Sources
1. **Sentinel-2 (Optical):**
   - Source: Microsoft Planetary Computer STAC (`sentinel-2-l2a`).
   - Indices: 
     - **NDVI** (Season Max): (NIR - Red) / (NIR + Red).
     - **LSWI** (Season Max): (NIR - SWIR) / (NIR + SWIR).
   - Logic: Use "Scene Classification Layer" (SCL) to mask clouds (values 3, 8, 9, 10).

2. **Sentinel-1 (Radar):**
   - Source: Microsoft Planetary Computer STAC (`sentinel-1-rtc`).
   - Band: **VH** polarization.
   - Indices:
     - **Max Backscatter:** Peak dB value during crop season.
     - **Integrated Backscatter:** Area under the curve (sum of dB) for the season.

3. **Rainfall:**
   - Source: CHIRPS or ERA5 (Gridded).
   - Logic: Cap rainfall at 150% of normal (excess rain does not equal linear yield increase).
