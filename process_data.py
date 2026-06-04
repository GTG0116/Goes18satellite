import boto3
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import numpy as np
import os
import json
import urllib.request
from datetime import datetime, timezone, timedelta
from botocore import UNSIGNED
from botocore.config import Config
from scipy.ndimage import zoom

OUTPUT_DIR = 'site/data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BUCKET = 'noaa-goes18'
MAX_FRAMES = 10  # rolling frame buffer per product

# Geographic extent: [west_lon, east_lon, south_lat, north_lat]
# This MUST match the imageBounds in site/index.html
# GOES-18 Full Disk covers the full hemisphere from ~142°E to ~56°W.
# Expressed as extended western longitudes: 142°E = -218°W (i.e. -218).
# This avoids splitting the disk at the antimeridian.
SATELLITE_LON = -137.0   # approximate GOES-18 nadir longitude
EXTENT = [-220, -55, -80, 80]

# Full-disk bands are enormous (Band 2: ~21 696×21 696; Band 13: ~5 424×5 424).
# Subsample so that neither dimension exceeds this value before rendering.
# 2048 gives excellent visual quality at the output DPI while keeping render
# times under a minute.
MAX_PIXELS = 2048

# Tropical cyclone sector configuration
NHC_PACIFIC_URL = (
    'https://raw.githubusercontent.com/GTG0116/JTWCTyphoonData/'
    'claude/jtwc-forecast-viewer-NQqQA/data/nhc_pacific.json'
)
SECTOR_DEG = 6.0          # degrees of lat/lon around the storm centre
SECTOR_MAX_PIXELS = 4096  # keep native resolution for small sectors


# ---------------------------------------------------------------------------
# Custom colour maps
# ---------------------------------------------------------------------------

def _ir_colormap():
    """NWS-style rainbow IR enhancement.

    Maps brightness temperature (190 K → 310 K):
      Cold cloud tops  (190–220 K) → white / magenta / red / orange
      Moderate clouds  (230–260 K) → orange / green / cyan
      Warm clear sky   (270–310 K) → blue → dark blue → near-black
    """
    return LinearSegmentedColormap.from_list('ir_enhancement', [
        (0.00, '#ffffff'),  # 190 K  – white  (extreme cold tops)
        (0.07, '#dd00dd'),  # 199 K  – magenta
        (0.17, '#ff0000'),  # 210 K  – red
        (0.27, '#ff5500'),  # 222 K  – orange-red
        (0.37, '#ff8800'),  # 233 K  – orange (was amber, removed yellow cast)
        (0.45, '#44cc00'),  # 244 K  – green  (was yellow #ffff00 – removed)
        (0.53, '#00cc00'),  # 254 K  – green
        (0.62, '#00cccc'),  # 264 K  – cyan
        (0.72, '#0066ff'),  # 276 K  – blue
        (0.87, '#001177'),  # 294 K  – dark blue
        (1.00, '#060606'),  # 310 K  – near-black
    ])


def _wv_colormap():
    """Water-vapour enhancement colormap.

    Maps brightness temperature (195 K → 280 K):
      Cold / moist upper troposphere (195–225 K) → deep navy → royal blue
      Moderate moisture              (225–250 K) → medium blue → teal
      Warm / dry troposphere         (250–280 K) → green → orange → red
    """
    return LinearSegmentedColormap.from_list('wv_enhancement', [
        (0.00, '#00003c'),  # 195 K  – deep navy
        (0.18, '#0000cc'),  # 209 K  – royal blue
        (0.35, '#0066ee'),  # 222 K  – medium blue
        (0.50, '#00bbdd'),  # 233 K  – light blue / cyan
        (0.63, '#00bb66'),  # 242 K  – teal-green
        (0.74, '#22cc00'),  # 250 K  – green (was yellow-green #aadd00 – removed)
        (0.84, '#ff8800'),  # 258 K  – orange (was yellow #ffcc00 – removed)
        (0.92, '#ff5500'),  # 265 K  – deep orange
        (1.00, '#cc1100'),  # 280 K  – red-orange (warm / dry)
    ])


# ---------------------------------------------------------------------------
# Frame management
# ---------------------------------------------------------------------------

def shift_frames(product_base):
    """Shift existing frames back one slot to make room for a new _00 frame.

    _00 is always the newest frame; _{MAX_FRAMES-1} is the oldest.
    When the buffer is full the oldest frame is deleted before shifting.
    A legacy single-file (product.png) is migrated to the oldest slot on
    the first call so no historical imagery is lost.
    """
    legacy   = os.path.join(OUTPUT_DIR, f'{product_base}.png')
    frame_00 = os.path.join(OUTPUT_DIR, f'{product_base}_00.png')

    # One-time migration: seed the oldest slot with the pre-frame-buffer image
    if os.path.exists(legacy) and not os.path.exists(frame_00):
        seed = os.path.join(OUTPUT_DIR, f'{product_base}_{MAX_FRAMES - 1:02d}.png')
        os.rename(legacy, seed)
        print(f"  Migrated legacy {product_base}.png → {os.path.basename(seed)}")

    # Count how many frame files currently exist
    n_existing = sum(
        1 for i in range(MAX_FRAMES)
        if os.path.exists(os.path.join(OUTPUT_DIR, f'{product_base}_{i:02d}.png'))
    )

    # Drop the oldest frame only when the buffer is already at capacity
    if n_existing >= MAX_FRAMES:
        oldest = os.path.join(OUTPUT_DIR, f'{product_base}_{MAX_FRAMES - 1:02d}.png')
        if os.path.exists(oldest):
            os.remove(oldest)

    # Shift _08→_09, _07→_08, …, _00→_01
    for i in range(MAX_FRAMES - 2, -1, -1):
        src = os.path.join(OUTPUT_DIR, f'{product_base}_{i:02d}.png')
        dst = os.path.join(OUTPUT_DIR, f'{product_base}_{i + 1:02d}.png')
        if os.path.exists(src):
            os.rename(src, dst)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_latest_goes_file(s3_client, band, domain='F'):
    """Find the most recent GOES-18 ABI CMIP file for a given band.

    Searches backwards up to 6 hours to find the latest available file.
    Domain 'F' = Full Disk, 'C' = CONUS, 'M' = Mesoscale.
    """
    now = datetime.now(timezone.utc)

    for hour_offset in range(6):
        t = now - timedelta(hours=hour_offset)
        year = t.strftime('%Y')
        doy  = t.strftime('%j')
        hour = t.strftime('%H')

        prefix   = f'ABI-L2-CMIP{domain}/{year}/{doy}/{hour}/'
        band_str = f'C{band:02d}_G18'

        try:
            resp  = s3_client.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
            files = [
                obj['Key'] for obj in resp.get('Contents', [])
                if band_str in obj['Key']
            ]
            if files:
                latest = sorted(files)[-1]
                print(f"  Found: {os.path.basename(latest)}")
                return latest
        except Exception as e:
            print(f"  Warning: could not list {prefix}: {e}")

    return None


def _make_figure():
    """Create a figure rendered in satellite-centred Web Mercator to match Leaflet.

    Leaflet uses spherical Web Mercator (EPSG:3857) with a fixed sphere radius
    of 6 378 137 m.  The GOES-18 full disk spans from ~142°E to ~56°W, which
    straddles the antimeridian in standard Mercator (central_longitude=0°).
    Using central_longitude=SATELLITE_LON shifts the x-axis so the entire
    disk falls within a single continuous x range, avoiding the antimeridian
    split.  The formula is identical to EPSG:3857 — just a constant x-offset —
    so Leaflet's imageBounds (which use geographic lat/lon) still align the
    overlay pixel-perfectly with the basemap at all latitudes.
    """
    R = 6378137.0  # Web Mercator / EPSG:3857 sphere radius in metres

    # x limits in satellite-centred Mercator metres.
    # EXTENT longitudes are geographic; subtract SATELLITE_LON to get the
    # offset from the projection's central meridian.
    x_min = R * np.radians(EXTENT[0] - SATELLITE_LON)  # west edge
    x_max = R * np.radians(EXTENT[1] - SATELLITE_LON)  # east edge
    y_min = R * np.log(np.tan(np.pi / 4 + np.radians(EXTENT[2]) / 2))  # south lat
    y_max = R * np.log(np.tan(np.pi / 4 + np.radians(EXTENT[3]) / 2))  # north lat

    # Aspect ratio in metres (height / width) for the correct figure proportions
    mercator_aspect = (y_max - y_min) / (x_max - x_min)

    fig_width  = 12.0
    fig_height = fig_width * mercator_aspect

    # Spherical Mercator centred on the satellite nadir — matches EPSG:3857
    # but shifted so the full disk doesn't straddle the antimeridian.
    web_mercator = ccrs.Mercator(
        central_longitude=SATELLITE_LON,
        globe=ccrs.Globe(semimajor_axis=R, semiminor_axis=R)
    )

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax  = fig.add_axes([0, 0, 1, 1], projection=web_mercator)
    # Set limits directly in projected metres — avoids set_extent padding/rounding
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('auto')  # fill the axes exactly; no equal-aspect padding
    ax.set_axis_off()
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    return fig, ax


def _download_band(s3_client, band_num):
    """Download a GOES-18 ABI band (Full Disk).

    Returns (data_array, x_metres, y_metres, goes_proj).
    data_array contains raw float values with NaNs intact (no fill applied).
    Returns (None, None, None, None) on any failure.
    """
    print(f"  Downloading Band {band_num}...")
    key = get_latest_goes_file(s3_client, band_num)
    if key is None:
        print(f"  ERROR: No Band {band_num} data found in the last 6 hours.")
        return None, None, None, None

    local_file = f'/tmp/goes_band{band_num}.nc'
    try:
        s3_client.download_file(BUCKET, key, local_file)
        ds = xr.open_dataset(local_file, engine='netcdf4')

        cmi = ds['CMI'].values.astype(np.float32)

        proj_var  = ds['goes_imager_projection']
        sat_h     = float(proj_var.attrs['perspective_point_height'])
        sat_lon   = float(proj_var.attrs['longitude_of_projection_origin'])
        sat_sweep = str(proj_var.attrs['sweep_angle_axis'])
        x = ds['x'].values * sat_h
        y = ds['y'].values * sat_h
        ds.close()

        # Subsample full-disk arrays to MAX_PIXELS so rendering stays fast.
        h, w = cmi.shape
        sy = max(1, h // MAX_PIXELS)
        sx = max(1, w // MAX_PIXELS)
        if sy > 1 or sx > 1:
            cmi = cmi[::sy, ::sx]
            x   = x[::sx]
            y   = y[::sy]
            print(f"  Subsampled Band {band_num}: {h}×{w} → {cmi.shape[0]}×{cmi.shape[1]}")

        goes_proj = ccrs.Geostationary(
            central_longitude=sat_lon,
            satellite_height=sat_h,
            sweep_axis=sat_sweep,
        )
        return cmi, x, y, goes_proj

    except Exception as e:
        print(f"  ERROR loading Band {band_num}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

    finally:
        if os.path.exists(local_file):
            os.remove(local_file)


# ---------------------------------------------------------------------------
# Single-band renderer
# ---------------------------------------------------------------------------

def process_goes_band(s3_client, band, output_filename, colormap, vmin, vmax, gamma=1.0):
    """Download and render a single GOES-18 ABI Full Disk band as a transparent PNG.

    gamma – optional power-law correction applied after normalising to [0, 1].
            gamma < 1 brightens the image (e.g. 0.5 = square-root stretch).
    """
    print(f"\n--- Band {band}: {output_filename} ---")

    key = get_latest_goes_file(s3_client, band)
    if key is None:
        print(f"  ERROR: No Band {band} data found in the last 6 hours. Skipping.")
        return

    local_file = f'/tmp/goes_band{band}.nc'
    try:
        print(f"  Downloading...")
        s3_client.download_file(BUCKET, key, local_file)

        ds       = xr.open_dataset(local_file, engine='netcdf4')
        cmi_data = ds['CMI'].values.astype(np.float32)  # Reflectance or BT [K]

        # --- Projection parameters from the file ---
        proj_var  = ds['goes_imager_projection']
        sat_h     = float(proj_var.attrs['perspective_point_height'])
        sat_lon   = float(proj_var.attrs['longitude_of_projection_origin'])
        sat_sweep = str(proj_var.attrs['sweep_angle_axis'])

        # Convert scan angles (radians) → projection coordinates (meters)
        x = ds['x'].values * sat_h
        y = ds['y'].values * sat_h
        ds.close()

        # Subsample full-disk arrays to MAX_PIXELS — pcolormesh on 21 k×21 k
        # quads would never finish; imshow + subsampling is orders of magnitude
        # faster and produces identical visual output at the output DPI.
        h, w = cmi_data.shape
        sy = max(1, h // MAX_PIXELS)
        sx = max(1, w // MAX_PIXELS)
        if sy > 1 or sx > 1:
            cmi_data = cmi_data[::sy, ::sx]
            x        = x[::sx]
            y        = y[::sy]
            print(f"  Subsampled: {h}×{w} → {cmi_data.shape[0]}×{cmi_data.shape[1]}")

        # Apply gamma correction if requested (normalise → gamma → restore range)
        if gamma != 1.0:
            normed   = np.clip((cmi_data - vmin) / (vmax - vmin), 0.0, 1.0)
            cmi_data = np.power(normed, gamma) * (vmax - vmin) + vmin

        goes_proj = ccrs.Geostationary(
            central_longitude=sat_lon,
            satellite_height=sat_h,
            sweep_axis=sat_sweep
        )

        fig, ax = _make_figure()

        # imshow is dramatically faster than pcolormesh for regular grids.
        # extent = (left, right, bottom, top) in projection coordinates;
        # y[0] is the northernmost scan line (largest y value).
        ax.imshow(
            cmi_data,
            origin='upper',
            extent=(x[0], x[-1], y[-1], y[0]),
            transform=goes_proj,
            aspect='auto',
            interpolation='nearest',
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
        )

        product_base = output_filename.replace('.png', '')
        shift_frames(product_base)
        output_path = os.path.join(OUTPUT_DIR, f'{product_base}_00.png')
        plt.savefig(output_path, dpi=150, transparent=True)
        plt.close()
        print(f"  Saved: {output_path}")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if os.path.exists(local_file):
            os.remove(local_file)


# ---------------------------------------------------------------------------
# GeoColor composite (day / night)
# ---------------------------------------------------------------------------

def process_geocolor(s3_client):
    """GeoColor RGB composite.

    Daytime  – pseudo-natural colour from Bands 1 and 2 with gamma correction.
    Nighttime – IR cloud layer (Band 13) blended with city-lights proxy
                (Band 7 minus thermal background) on a transparent background.
    """
    print(f"\n--- GeoColor RGB Composite ---")

    # Fetch the two visible bands needed for the RGB composite
    b1, x1, y1, goes_proj = _download_band(s3_client, 1)
    if b1 is None:
        print("  ERROR: Missing Band 1. Skipping GeoColor.")
        return

    b2, x2, y2, _ = _download_band(s3_client, 2)
    if b2 is None:
        print("  ERROR: Missing Band 2. Skipping GeoColor.")
        return

    # Band 2 is 0.5 km (2× resolution) – downsample to match Band 1 (1 km)
    b2 = np.nan_to_num(b2, nan=0.0)
    if b2.shape != b1.shape:
        zy = b1.shape[0] / b2.shape[0]
        zx = b1.shape[1] / b2.shape[1]
        b2 = zoom(b2, (zy, zx), order=1)

    b1 = np.nan_to_num(b1, nan=0.0)

    # Determine day vs night: at night Band 2 visible reflectance ≈ 0
    mean_ref = float(np.nanmean(b2))
    is_daytime = mean_ref > 0.05
    print(f"  Band 2 mean reflectance: {mean_ref:.4f}  →  {'DAYTIME' if is_daytime else 'NIGHTTIME'}")

    if is_daytime:
        # Fetch Band 13 for cloud-top enhancement (failure is non-fatal)
        bt13, *_ = _download_band(s3_client, 13)
        _render_geocolor_day(b1, b2, x1, y1, goes_proj, bt13)
    else:
        _render_geocolor_night(s3_client)


def _render_geocolor_day(b1, b2, x, y, goes_proj, bt13=None):
    """Pseudo-natural colour composite for daytime.

    bt13 is an optional Band 13 brightness-temperature array (same spatial
    footprint after resampling).  When provided, very cold cloud tops are
    blended towards bright white so deep convective anvils are clearly
    visible against land/ocean backgrounds.
    """
    R = np.clip(b2, 0, 1)
    # Synthetic green: average of red and blue channels only.
    # Omitting NIR (Band 3 / 0.86 µm) prevents the yellow cast that NIR
    # introduces over vegetated and arid land surfaces.
    G = np.clip(0.5 * b2 + 0.5 * b1, 0, 1)
    B = np.clip(b1, 0, 1)

    # Gamma correction for natural brightness
    gamma = 0.5
    R = np.power(R, gamma)
    G = np.power(G, gamma)
    B = np.power(B, gamma)

    # --- Cloud enhancement via Band 13 IR ---
    # Pixels colder than ~255 K (high cloud tops) are blended towards bright
    # white, making anvils and deep convection clearly pop out.
    if bt13 is not None:
        bt13_f = np.where(np.isnan(bt13), 320.0, bt13)
        # Band 13 is 2 km; Band 1/2/3 composite is at 1 km – upsample to match
        if bt13_f.shape != R.shape:
            zy = R.shape[0] / bt13_f.shape[0]
            zx = R.shape[1] / bt13_f.shape[1]
            bt13_f = zoom(bt13_f, (zy, zx), order=1)
        # 255 K → 0 (no enhancement); 200 K → 1 (pure-white cloud tops)
        cloud_enhance = np.clip((255.0 - bt13_f) / 55.0, 0.0, 1.0)
        strength = 0.85
        R = np.clip(R + cloud_enhance * (1.0 - R) * strength, 0.0, 1.0)
        G = np.clip(G + cloud_enhance * (1.0 - G) * strength, 0.0, 1.0)
        B = np.clip(B + cloud_enhance * (1.0 - B) * (strength + 0.05), 0.0, 1.0)

    rgb = np.dstack([R, G, B])
    fig, ax = _make_figure()
    img_extent = (x[0], x[-1], y[-1], y[0])
    ax.imshow(rgb, origin='upper', extent=img_extent,
              transform=goes_proj, aspect='auto', interpolation='none')

    shift_frames('geocolor')
    output_path = os.path.join(OUTPUT_DIR, 'geocolor_00.png')
    plt.savefig(output_path, dpi=150, transparent=True)
    plt.close()
    print(f"  Saved (daytime): {output_path}")


def _render_geocolor_night(s3_client):
    """Nighttime GeoColor composite.

    Cloud layer  – derived from Band 13 (10.35 µm clean IR window).
                   Cold temperatures → bright blue-white clouds.
    City lights  – derived from Band 7 (3.9 µm shortwave IR) minus the
                   thermal background estimated from Band 13.  At night,
                   cities, fires, and industrial heat sources emit
                   anomalously in 3.9 µm.
    Background   – fully transparent so the dark basemap shows through.
    """
    # Band 13: brightness temperature (K), same 2 km resolution as Band 7
    bt13, x13, y13, goes_proj = _download_band(s3_client, 13)
    if bt13 is None:
        print("  ERROR: Missing Band 13 for nighttime GeoColor. Skipping.")
        return

    # Fill off-earth NaNs with a warm value so they don't look like cloud tops
    bt13 = np.where(np.isnan(bt13), 320.0, bt13)

    # Band 7: 3.9 µm shortwave IR — optional; gracefully absent
    bt7, x7, y7, _ = _download_band(s3_client, 7)

    h, w = bt13.shape

    # ------------------------------------------------------------------
    # Cloud layer
    # ------------------------------------------------------------------
    # 275 K → no cloud (opacity 0); 220 K → deep convection (opacity 1)
    cloud_opacity = np.clip((275.0 - bt13) / 55.0, 0.0, 1.0)

    # Clouds rendered as cool blue-white (scattered light / natural night look)
    cloud_R = cloud_opacity * 0.80
    cloud_G = cloud_opacity * 0.88
    cloud_B = cloud_opacity * 1.00

    # ------------------------------------------------------------------
    # City lights layer
    # ------------------------------------------------------------------
    if bt7 is not None:
        bt7 = np.where(np.isnan(bt7), 0.0, bt7)

        # Match Band 7 resolution to Band 13 if needed
        if bt7.shape != (h, w):
            zy = h / bt7.shape[0]
            zx = w / bt7.shape[1]
            bt7 = zoom(bt7, (zy, zx), order=1)

        # At typical surface temperatures Band 7 BT runs ~12 K cooler than
        # Band 13 due to Planck function differences.  Where Band 7 exceeds
        # this offset (i.e. Band7 > Band13 − 12) there is anomalous emission
        # from city lights, fires, or industrial heat.
        city_raw = np.clip((bt7 - (bt13 - 12.0)) / 25.0, 0.0, 1.0)

        # Only paint city lights over clear, warm surface pixels
        surface_clear = (bt13 > 265.0).astype(np.float32)
        city_lights   = city_raw * surface_clear
    else:
        city_lights = np.zeros((h, w), dtype=np.float32)
        print("  WARNING: Band 7 unavailable; city lights layer disabled.")

    # ------------------------------------------------------------------
    # Compose RGBA image
    # ------------------------------------------------------------------
    # Clouds: blue-white  |  City lights: warm yellow-orange
    R = np.clip(cloud_R + city_lights * 1.00, 0.0, 1.0)
    G = np.clip(cloud_G + city_lights * 0.75, 0.0, 1.0)
    B = np.clip(cloud_B + city_lights * 0.10, 0.0, 1.0)

    # Alpha: transparent where there is nothing to show
    A = np.clip(cloud_opacity + city_lights * 2.0, 0.0, 1.0)

    rgba = np.dstack([R, G, B, A]).astype(np.float32)

    fig, ax = _make_figure()
    img_extent = (x13[0], x13[-1], y13[-1], y13[0])
    ax.imshow(rgba, origin='upper', extent=img_extent,
              transform=goes_proj, aspect='auto', interpolation='none')

    shift_frames('geocolor')
    output_path = os.path.join(OUTPUT_DIR, 'geocolor_00.png')
    plt.savefig(output_path, dpi=150, transparent=True)
    plt.close()
    print(f"  Saved (nighttime): {output_path}")


# ---------------------------------------------------------------------------
# Tropical cyclone sector helpers
# ---------------------------------------------------------------------------

def fetch_cyclone_list():
    """Fetch active tropical cyclones from the NHC Pacific JSON feed.

    Returns a list of dicts with keys: id, name, lat, lon.
    Returns [] on failure or when no storms are active.
    """
    try:
        req = urllib.request.Request(
            NHC_PACIFIC_URL,
            headers={'User-Agent': 'goes18-processor/1.0'},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"  WARNING: Could not fetch NHC Pacific cyclone list: {e}")
        return []

    raw_storms = data.get('storms', data.get('activeStorms', []))
    storms = []
    for s in raw_storms:
        # The NHC feed nests the live position under a "current" object
        # ({"current": {"lat": 12.5, "lon": -130.5, ...}}); fall back to the
        # storm dict itself for flatter feed variants.
        pos = s.get('current') if isinstance(s.get('current'), dict) else s

        # Try several common field names for numeric coordinates
        lat = pos.get('lat', pos.get('latitudeNumeric'))
        lon = pos.get('lon', pos.get('longitudeNumeric'))

        # Fall back to parsing string forms like "20.5N" / "97.5W"
        if lat is None and 'latitude' in pos:
            try:
                raw = str(pos['latitude'])
                lat = float(raw.replace('N', '').replace('S', ''))
                if 'S' in raw:
                    lat = -lat
            except ValueError:
                pass
        if lon is None and 'longitude' in pos:
            try:
                raw = str(pos['longitude'])
                lon = float(raw.replace('E', '').replace('W', ''))
                if 'W' in raw:
                    lon = -lon
            except ValueError:
                pass

        if lat is None or lon is None:
            print(f"  WARNING: Could not parse coordinates for storm: {s}")
            continue

        storm_id   = str(s.get('id', s.get('stormId', f'tc{len(storms):02d}'))).lower()
        storm_name = str(s.get('name', s.get('stormName', storm_id.upper())))
        storms.append({'id': storm_id, 'name': storm_name,
                       'lat': float(lat), 'lon': float(lon)})

    print(f"  {len(storms)} active storm(s) in NHC Pacific feed.")
    return storms


def _make_figure_sector(west, east, south, north):
    """Create a figure for a small cyclone sector in satellite-centred Web Mercator.

    Mirrors _make_figure() but accepts explicit geographic bounds instead of
    using the global EXTENT.
    """
    R = 6378137.0
    x_min = R * np.radians(west  - SATELLITE_LON)
    x_max = R * np.radians(east  - SATELLITE_LON)
    y_min = R * np.log(np.tan(np.pi / 4 + np.radians(south) / 2))
    y_max = R * np.log(np.tan(np.pi / 4 + np.radians(north) / 2))

    mercator_aspect = (y_max - y_min) / (x_max - x_min)
    fig_width  = 8.0
    fig_height = fig_width * mercator_aspect

    web_mercator = ccrs.Mercator(
        central_longitude=SATELLITE_LON,
        globe=ccrs.Globe(semimajor_axis=R, semiminor_axis=R),
    )
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax  = fig.add_axes([0, 0, 1, 1], projection=web_mercator)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('auto')
    ax.set_axis_off()
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    return fig, ax


def _download_band_sector(s3_client, band_num, lat, lon):
    """Download a GOES-18 full-disk band and crop to the cyclone sector.

    The sector is a ±SECTOR_DEG lat/lon box around (lat, lon).
    Returns the cropped array at native resolution (no subsampling unless
    the crop still exceeds SECTOR_MAX_PIXELS on either axis).
    Returns (None, None, None, None) on failure or if the storm is outside
    the GOES-18 field of view.
    """
    print(f"  Downloading Band {band_num} (sector)...")
    key = get_latest_goes_file(s3_client, band_num)
    if key is None:
        print(f"  ERROR: No Band {band_num} data found.")
        return None, None, None, None

    local_file = f'/tmp/goes_band{band_num}_sector.nc'
    try:
        s3_client.download_file(BUCKET, key, local_file)
        ds = xr.open_dataset(local_file, engine='netcdf4')

        cmi = ds['CMI'].values.astype(np.float32)
        proj_var  = ds['goes_imager_projection']
        sat_h     = float(proj_var.attrs['perspective_point_height'])
        sat_lon   = float(proj_var.attrs['longitude_of_projection_origin'])
        sat_sweep = str(proj_var.attrs['sweep_angle_axis'])
        x = ds['x'].values * sat_h
        y = ds['y'].values * sat_h
        ds.close()

        goes_proj = ccrs.Geostationary(
            central_longitude=sat_lon,
            satellite_height=sat_h,
            sweep_axis=sat_sweep,
        )

        # Transform the four sector corners from lat/lon → GOES projection metres
        pc = ccrs.PlateCarree()
        west  = lon - SECTOR_DEG;  east  = lon + SECTOR_DEG
        south = lat - SECTOR_DEG;  north = lat + SECTOR_DEG
        corners = np.array([[west, south], [east, south],
                             [west, north], [east, north]])
        corners_xy = goes_proj.transform_points(pc, corners[:, 0], corners[:, 1])

        xp_min = np.nanmin(corners_xy[:, 0])
        xp_max = np.nanmax(corners_xy[:, 0])
        yp_min = np.nanmin(corners_xy[:, 1])
        yp_max = np.nanmax(corners_xy[:, 1])

        # x increases west→east; y decreases north→south in GOES arrays
        xi = np.where((x >= xp_min) & (x <= xp_max))[0]
        yi = np.where((y >= yp_min) & (y <= yp_max))[0]

        if xi.size == 0 or yi.size == 0:
            print(f"  WARNING: Storm sector outside GOES-18 coverage for Band {band_num}.")
            return None, None, None, None

        cmi_crop = cmi[yi[0]:yi[-1] + 1, xi[0]:xi[-1] + 1]
        x_crop   = x[xi[0]:xi[-1] + 1]
        y_crop   = y[yi[0]:yi[-1] + 1]

        h, w = cmi_crop.shape
        sy = max(1, h // SECTOR_MAX_PIXELS)
        sx = max(1, w // SECTOR_MAX_PIXELS)
        if sy > 1 or sx > 1:
            cmi_crop = cmi_crop[::sy, ::sx]
            x_crop   = x_crop[::sx]
            y_crop   = y_crop[::sy]
            print(f"  Sector Band {band_num}: {h}×{w} → {cmi_crop.shape[0]}×{cmi_crop.shape[1]}")
        else:
            print(f"  Sector Band {band_num}: {h}×{w} (native resolution)")

        return cmi_crop, x_crop, y_crop, goes_proj

    except Exception as e:
        print(f"  ERROR loading Band {band_num} sector: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

    finally:
        if os.path.exists(local_file):
            os.remove(local_file)


def process_cyclone_band(s3_client, storm_key, band, output_base,
                         colormap, vmin, vmax, gamma, lat, lon):
    """Render a single GOES-18 band cropped to a cyclone sector."""
    cmi, x, y, goes_proj = _download_band_sector(s3_client, band, lat, lon)
    if cmi is None:
        return

    if gamma != 1.0:
        normed = np.clip((cmi - vmin) / (vmax - vmin), 0.0, 1.0)
        cmi = np.power(normed, gamma) * (vmax - vmin) + vmin

    west, east   = lon - SECTOR_DEG, lon + SECTOR_DEG
    south, north = lat - SECTOR_DEG, lat + SECTOR_DEG

    fig, ax = _make_figure_sector(west, east, south, north)
    ax.imshow(cmi, origin='upper', extent=(x[0], x[-1], y[-1], y[0]),
              transform=goes_proj, aspect='auto', interpolation='nearest',
              cmap=colormap, vmin=vmin, vmax=vmax)

    shift_frames(output_base)
    output_path = os.path.join(OUTPUT_DIR, f'{output_base}_00.png')
    plt.savefig(output_path, dpi=150, transparent=True)
    plt.close()
    print(f"  Saved: {output_path}")


def process_cyclone_geocolor(s3_client, storm_key, lat, lon):
    """GeoColor RGB composite cropped to a cyclone sector.

    Day/night logic mirrors process_geocolor() but uses sector downloads
    and _make_figure_sector() for the output extent.
    """
    print(f"  --- GeoColor sector ({storm_key}) ---")
    output_base  = f'geocolor_tc_{storm_key}'
    west, east   = lon - SECTOR_DEG, lon + SECTOR_DEG
    south, north = lat - SECTOR_DEG, lat + SECTOR_DEG

    b1, x1, y1, goes_proj = _download_band_sector(s3_client, 1, lat, lon)
    if b1 is None:
        print("  ERROR: Missing Band 1. Skipping GeoColor sector.")
        return

    b2, x2, y2, _ = _download_band_sector(s3_client, 2, lat, lon)
    if b2 is None:
        print("  ERROR: Missing Band 2. Skipping GeoColor sector.")
        return

    b2 = np.nan_to_num(b2, nan=0.0)
    if b2.shape != b1.shape:
        b2 = zoom(b2, (b1.shape[0] / b2.shape[0], b1.shape[1] / b2.shape[1]), order=1)
    b1 = np.nan_to_num(b1, nan=0.0)

    mean_ref   = float(np.nanmean(b2))
    is_daytime = mean_ref > 0.05
    print(f"  Band 2 mean reflectance: {mean_ref:.4f}  →  {'DAYTIME' if is_daytime else 'NIGHTTIME'}")

    fig, ax = _make_figure_sector(west, east, south, north)

    if is_daytime:
        bt13, *_ = _download_band_sector(s3_client, 13, lat, lon)

        R = np.clip(b2, 0, 1)
        G = np.clip(0.5 * b2 + 0.5 * b1, 0, 1)
        B = np.clip(b1, 0, 1)
        gamma = 0.5
        R, G, B = np.power(R, gamma), np.power(G, gamma), np.power(B, gamma)

        if bt13 is not None:
            bt13_f = np.where(np.isnan(bt13), 320.0, bt13)
            if bt13_f.shape != R.shape:
                bt13_f = zoom(bt13_f,
                              (R.shape[0] / bt13_f.shape[0],
                               R.shape[1] / bt13_f.shape[1]), order=1)
            ce = np.clip((255.0 - bt13_f) / 55.0, 0.0, 1.0)
            s  = 0.85
            R  = np.clip(R + ce * (1.0 - R) * s,        0.0, 1.0)
            G  = np.clip(G + ce * (1.0 - G) * s,        0.0, 1.0)
            B  = np.clip(B + ce * (1.0 - B) * (s + 0.05), 0.0, 1.0)

        rgb = np.dstack([R, G, B])
        ax.imshow(rgb, origin='upper', extent=(x1[0], x1[-1], y1[-1], y1[0]),
                  transform=goes_proj, aspect='auto', interpolation='none')

    else:
        bt13, x13, y13, _ = _download_band_sector(s3_client, 13, lat, lon)
        if bt13 is None:
            plt.close()
            print("  ERROR: Missing Band 13 for nighttime GeoColor sector. Skipping.")
            return
        bt13 = np.where(np.isnan(bt13), 320.0, bt13)
        bt7, *_ = _download_band_sector(s3_client, 7, lat, lon)

        h, w = bt13.shape
        cloud_opacity = np.clip((275.0 - bt13) / 55.0, 0.0, 1.0)
        cloud_R = cloud_opacity * 0.80
        cloud_G = cloud_opacity * 0.88
        cloud_B = cloud_opacity * 1.00

        if bt7 is not None:
            bt7 = np.where(np.isnan(bt7), 0.0, bt7)
            if bt7.shape != (h, w):
                bt7 = zoom(bt7, (h / bt7.shape[0], w / bt7.shape[1]), order=1)
            city_raw    = np.clip((bt7 - (bt13 - 12.0)) / 25.0, 0.0, 1.0)
            city_lights = city_raw * (bt13 > 265.0).astype(np.float32)
        else:
            city_lights = np.zeros((h, w), dtype=np.float32)

        R = np.clip(cloud_R + city_lights * 1.00, 0.0, 1.0)
        G = np.clip(cloud_G + city_lights * 0.75, 0.0, 1.0)
        B = np.clip(cloud_B + city_lights * 0.10, 0.0, 1.0)
        A = np.clip(cloud_opacity + city_lights * 2.0, 0.0, 1.0)
        rgba = np.dstack([R, G, B, A]).astype(np.float32)

        ax.imshow(rgba, origin='upper', extent=(x13[0], x13[-1], y13[-1], y13[0]),
                  transform=goes_proj, aspect='auto', interpolation='none')

    shift_frames(output_base)
    output_path = os.path.join(OUTPUT_DIR, f'{output_base}_00.png')
    plt.savefig(output_path, dpi=150, transparent=True)
    plt.close()
    print(f"  Saved: {output_path}")


def process_cyclones(s3_client):
    """Fetch active tropical cyclones and generate a sector image set for each.

    Products per storm (±SECTOR_DEG box, max native resolution):
      • GeoColor   — 2 km  (Band 1 @ 1 km + Band 2 @ 0.5 km)
      • Visible    — 0.5 km (Band 2)
      • Infrared   — 2 km  (Band 13)
      • Water Vapor — 2 km (Band 9)

    Also writes site/data/cyclones.json so the web viewer can build overlays
    dynamically without hardcoding storm names or positions.
    """
    print("\n--- Tropical Cyclone Sectors ---")
    storms   = fetch_cyclone_list()
    manifest = {'sector_deg': SECTOR_DEG, 'storms': []}

    for storm in storms:
        sid  = storm['id']
        name = storm['name']
        lat  = storm['lat']
        lon  = storm['lon']

        print(f"\n  Storm: {name} ({sid.upper()})  lat={lat:.1f}  lon={lon:.1f}")

        # Visible — Band 2 (0.5 km native resolution)
        process_cyclone_band(
            s3_client, sid, 2, f'visible_tc_{sid}',
            'gray', vmin=0.0, vmax=1.0, gamma=0.5, lat=lat, lon=lon,
        )
        # Infrared — Band 13 (2 km)
        process_cyclone_band(
            s3_client, sid, 13, f'infrared_tc_{sid}',
            _ir_colormap(), vmin=190, vmax=310, gamma=1.0, lat=lat, lon=lon,
        )
        # Water Vapor — Band 9 (2 km)
        process_cyclone_band(
            s3_client, sid, 9, f'water_vapor_tc_{sid}',
            _wv_colormap(), vmin=195, vmax=280, gamma=1.0, lat=lat, lon=lon,
        )
        # GeoColor — 2 km effective
        process_cyclone_geocolor(s3_client, sid, lat, lon)

        manifest['storms'].append({
            'id':   sid,
            'name': name,
            'lat':  lat,
            'lon':  lon,
        })

    manifest_path = os.path.join(OUTPUT_DIR, 'cyclones.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)
    print(f"\n  Cyclone manifest written: {manifest_path}  ({len(storms)} storm(s))")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("GOES-18 Satellite Image Processor (Full Disk)")
    print("=" * 40)
    print(f"Extent: {EXTENT}")
    print(f"Bucket: s3://{BUCKET}")

    # Anonymous access — GOES-18 bucket is publicly readable.
    # Full-disk files can be several hundred MB; set generous timeouts so the
    # client doesn't hang silently and allows GitHub Actions to fail fast.
    s3 = boto3.client(
        's3',
        region_name='us-east-1',
        config=Config(
            signature_version=UNSIGNED,
            connect_timeout=30,
            read_timeout=600,   # 10 min — large full-disk files can be 200+ MB
            retries={'max_attempts': 2},
        )
    )

    # Band 2  — Visible (0.64 µm)              reflectance [0.0 – 1.0]
    # 'gray' maps 0→black (clear sky) and 1→white (bright cloud).
    # gamma=0.5 (square-root stretch) matches conventional satellite display.
    process_goes_band(s3, 2,  'visible.png',  'gray',        vmin=0.0, vmax=1.0, gamma=0.5)

    # Band 13 — Clean IR Longwave (10.35 µm)   brightness temp [K]
    # Custom NWS-style rainbow: cold tops → red/orange, warm surface → dark blue/black
    process_goes_band(s3, 13, 'infrared.png', _ir_colormap(), vmin=190, vmax=310)

    # Band 9  — Mid-Level Water Vapor (6.95 µm) brightness temp [K]
    # Custom enhancement: cold/moist → navy/blue, warm/dry → orange/red
    process_goes_band(s3, 9,  'water_vapor.png', _wv_colormap(), vmin=195, vmax=280)

    # GeoColor — natural colour (day) or IR+city-lights composite (night)
    process_geocolor(s3)

    # Tropical cyclone sectors — max-resolution crops around each active storm
    process_cyclones(s3)

    # Write a plain-text timestamp so the website can show freshness
    ts_path = os.path.join(OUTPUT_DIR, 'last_updated.txt')
    with open(ts_path, 'w') as f:
        f.write(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'))
    print(f"\nTimestamp written: {ts_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
