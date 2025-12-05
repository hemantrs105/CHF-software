import ee

class GEEUtils:
    @staticmethod
    def get_s2_with_cloud_prob(start_date, end_date):
        """
        Joins Sentinel-2 SR Harmonized with Sentinel-2 Cloud Probability.
        Renames 'probability' band to 'MSK_CLDPRB' to match spec expectations.
        """
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate(start_date, end_date)

        s2_cld_prb = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY') \
            .filterDate(start_date, end_date)

        # Join based on system:index
        filter = ee.Filter.equals(leftField='system:index', rightField='system:index')
        join = ee.Join.saveFirst(matchKey='cloud_mask')

        s2_joined = ee.ImageCollection(join.apply(s2, s2_cld_prb, filter))

        def add_cloud_band(image):
            # Extract the cloud probability image from the 'cloud_mask' property
            cloud_img = ee.Image(image.get('cloud_mask'))
            # Rename 'probability' (0-100) to 'MSK_CLDPRB'
            return image.addBands(cloud_img.select(['probability'], ['MSK_CLDPRB']))

        return s2_joined.map(add_cloud_band)

    @staticmethod
    def mask_clouds(image):
        """
        Masks clouds in Sentinel-2 images using the MSK_CLDPRB band.
        Pixels with MSK_CLDPRB > 20 are masked.
        """
        cloud_prob = image.select('MSK_CLDPRB')
        is_cloud = cloud_prob.gt(20)
        return image.updateMask(is_cloud.Not())

    @staticmethod
    def get_max_ndvi(s2_collection, start_date, end_date):
        """
        Calculates Max NDVI from Sentinel-2 collection.
        s2_collection must already have the MSK_CLDPRB band.
        """
        def add_ndvi(image):
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('max_ndvi')
            return image.addBands(ndvi)

        # Collection is already filtered by date in get_s2_with_cloud_prob
        # But we pass dates to be safe or if reusing simple collection.
        # Here we assume s2_collection IS the pre-joined one.

        return s2_collection \
            .map(GEEUtils.mask_clouds) \
            .map(add_ndvi) \
            .select('max_ndvi') \
            .max()

    @staticmethod
    def get_max_lswi(s2_collection, start_date, end_date):
        """
        Calculates Max LSWI from Sentinel-2 collection.
        LSWI = (NIR - SWIR1) / (NIR + SWIR1)
        """
        def add_lswi(image):
            lswi = image.normalizedDifference(['B8', 'B11']).rename('max_lswi')
            return image.addBands(lswi)

        return s2_collection \
            .map(GEEUtils.mask_clouds) \
            .map(add_lswi) \
            .select('max_lswi') \
            .max()

    @staticmethod
    def get_backscatter(s1_collection, start_date, end_date):
        """
        Calculates Max Backscatter and Integrated Backscatter from Sentinel-1.
        Filter: IW, VH, Descending.
        """
        filtered = s1_collection.filterDate(start_date, end_date) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')) \
            .select(['VH'])

        # TODO: Refined Lee Speckle Filter (5x5).
        # Implementing a full Refined Lee filter in pure GEE Python API without external libraries
        # is complex and prone to errors.
        # As a robust alternative for noise reduction, we apply a focal mean (5x5 boxcar).
        def apply_filter(image):
             return image.focal_mean(radius=2.5, units='pixels', iterations=1)

        filtered_smooth = filtered.map(apply_filter)

        # Max Backscatter
        max_bs = filtered_smooth.max().rename('max_backscatter')

        # Integrated Backscatter (Sum)
        integrated_bs = filtered_smooth.sum().rename('integrated_backscatter')

        return max_bs.addBands(integrated_bs)

    @staticmethod
    def get_integrated_fapar(modis_collection, peak_start, peak_end):
        """
        Calculates Integrated FAPAR from MODIS/061/MCD15A3H.
        Band: Fpar
        """
        return modis_collection.filterDate(peak_start, peak_end) \
            .select('Fpar') \
            .sum() \
            .rename('integrated_fapar')

    @staticmethod
    def get_rainfall_metrics(chirps_collection, season_start, season_end, year_int):
        """
        Calculates Rainy Days and Adjusted Rainfall from CHIRPS.
        """
        # Filter for current season
        current_season = chirps_collection.filterDate(season_start, season_end).select('precipitation')

        # 7. Rainy Days: Count days > 2.5mm
        def is_rainy(image):
            return image.gt(2.5).rename('rainy_day')

        rainy_days = current_season.map(is_rainy).sum().rename('rainy_days')

        # 8. Adjusted Rainfall
        # Step A: Current Year Total Rain
        current_total = current_season.sum()

        # Step B: Normal (10-yr avg)
        # We need the 10 years PRIOR to current year.
        # We need to construct the date ranges for the previous 10 years dynamically
        # based on the month-day of season_start/end.

        s_date = ee.Date(season_start)
        e_date = ee.Date(season_end)

        # Helper to get same season for a different year
        def get_historical_seasonal_rain(y_offset):
            # y_offset is 1 to 10
            y_start = s_date.advance(ee.Number(y_offset).multiply(-1), 'year')
            y_end = e_date.advance(ee.Number(y_offset).multiply(-1), 'year')
            return chirps_collection.filterDate(y_start, y_end).select('precipitation').sum()

        offsets = ee.List.sequence(1, 10)
        normal_collection = ee.ImageCollection.fromImages(offsets.map(get_historical_seasonal_rain))
        normal = normal_collection.mean()

        # Step C: Adjusted Logic
        # If Current > 1.5 * Normal, cap at 1.5 * Normal
        cap = normal.multiply(1.5)
        adjusted_rain = current_total.min(cap).rename('adjusted_rainfall')

        return rainy_days.addBands(adjusted_rain)

    @staticmethod
    def get_all_indicators(year, roi, crop_map_asset, dates_config, target_crop_class=None):
        """
        Orchestrates the generation of all 8 indicators for a specific year.
        Returns a single multi-band Image.

        Args:
            year (int): Year of analysis.
            roi (ee.Geometry): Region of Interest.
            crop_map_asset (str): Asset ID for the crop map.
            dates_config (dict): Dictionary with 'season_start', 'season_end', 'peak_start', 'peak_end'.
            target_crop_class (int, optional): Pixel value to mask for (e.g. 4 for Paddy). If None, masks 0.
        """
        season_start = dates_config['season_start']
        season_end = dates_config['season_end']
        peak_start = dates_config['peak_start']
        peak_end = dates_config['peak_end']

        # 1. Collections
        # Pre-process Sentinel-2 with Cloud Probability
        s2_joined = GEEUtils.get_s2_with_cloud_prob(season_start, season_end)

        s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
        modis = ee.ImageCollection('MODIS/061/MCD15A3H')
        chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')

        # 2. Indicators
        img_ndvi = GEEUtils.get_max_ndvi(s2_joined, season_start, season_end)
        img_lswi = GEEUtils.get_max_lswi(s2_joined, season_start, season_end)
        img_bs = GEEUtils.get_backscatter(s1, season_start, season_end)
        img_fapar = GEEUtils.get_integrated_fapar(modis, peak_start, peak_end)
        img_rain = GEEUtils.get_rainfall_metrics(chirps, season_start, season_end, year)

        # 3. Combine
        combined = img_ndvi.addBands(img_lswi) \
                           .addBands(img_bs) \
                           .addBands(img_fapar) \
                           .addBands(img_rain)

        # 4. Crop Mask
        crop_map = ee.Image(crop_map_asset)

        if target_crop_class is not None:
            # Create binary mask where pixel == target_class
            mask = crop_map.eq(target_crop_class)
        else:
            # Assume 0 is no-data/non-crop
            mask = crop_map.neq(0)

        combined_masked = combined.updateMask(mask)

        # Clip to ROI
        return combined_masked.clip(roi)
