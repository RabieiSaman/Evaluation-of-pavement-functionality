import laspy
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import substring
import numpy as np
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer
from scipy.signal import savgol_filter
from tqdm import tqdm
from pyproj import CRS

# -----------------------------
# --- USER PATHS & TUNABLES ---
# -----------------------------
LAS_FILE   = r"C:\PhD_files\PhD_thesis\Postdoc - Positions\New Hampshire\Post-Doc_Activity_Data\Post-Doc_Activity_Data\rt26_sec2_31424 3.las"          # LiDAR raster (from LAS)
CENTERLINE = r"C:\PhD_files\PhD_thesis\Postdoc - Positions\New Hampshire\Post-Doc_Activity_Data\Post-Doc_Activity_Data\Post-Doc_Activity_Data\Road_CenterLine.shp"                   # polyline for roadway center
BUFFER_M = 2.0            # buffer around centerline to select points
PROFILE_STEP_M = 0.03     # 3 cm sampling

# --- LOAD CENTERLINE ---
cl = gpd.read_file(CENTERLINE)
line = cl.geometry.iloc[0]   # take first line
crs_line = cl.crs

# --- LOAD LAS ---
las = laspy.read(LAS_FILE)
x = las.x
y = las.y
z = las.z
crs_las = CRS.from_epsg(32619)   # <-- set to your LAS CRS (check metadata)

if CRS.from_user_input(crs_line) != crs_las:
    print('it was needed')
    cl = cl.to_crs(crs_las)
    line = cl.geometry.iloc[0]    

# --- CLIP POINTS TO ROAD CORRIDOR ---
road_buffer = line.buffer(BUFFER_M)
mask = []
for px, py in tqdm(zip(x, y), total=len(x), desc="Clipping points"):
    mask.append(road_buffer.contains(Point(px, py)))

mask = np.array(mask, dtype=bool)
x, y, z = x[mask], y[mask], z[mask]

print(f"Points after clipping: {len(z)}")

#-------------------------------------------------------------------------------------------------------

# --- PROJECT POINTS ONTO CENTERLINE ---
# distance along line for each point
distances = [line.project(Point(px, py)) for px, py in zip(x, y)]
distances = np.array(distances)

# Sort by distance
order = np.argsort(distances)
distances = distances[order]
z = z[order]

# --- BIN TO REGULAR PROFILE ---
max_dist = distances.max()
bins = np.arange(0, max_dist, PROFILE_STEP_M)
z_profile = np.full_like(bins, np.nan, dtype=float)

# average z within each bin
for i in tqdm(range(len(bins)-1)):
    in_bin = (distances >= bins[i]) & (distances < bins[i+1])
    if np.any(in_bin):
        z_profile[i] = np.mean(z[in_bin])

mask = np.isfinite(z_profile)
x_m = bins[mask]
z_m = z_profile[mask]

plt.plot(x_m, z_m, '.-')
plt.xlabel("Distance along road (m)")
plt.ylabel("Elevation (m)")
plt.title("Longitudinal profile from LAS")
plt.show()


# So now the results showed the Longitudinal profile across the road and now I want to provide the detrending and IRI calculation
#-------------------------------------------------------------------------------------------------------

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# --- PARAMETERS ---
PROFILE_STEP_M = 0.03    # sampling step used earlier (m)
SEG_LEN_M = 25.0         # segment length for reporting IRI (m)
IRI_SPEED_KMH = 80.0     # standard speed
OUT_DIR        = r"C:\PhD_files\PhD_thesis\Postdoc - Positions\New Hampshire\Post-Doc_Activity_Data\Post-Doc_Activity_Data\Output"  # existing folder
CSV_PATH       = rf"{OUT_DIR}/rt26_segments_metrics.csv"
SHP_SEG_PATH   = os.path.join(OUT_DIR, "segments_25m.shp")
SHP_PTS_PATH   = os.path.join(OUT_DIR, "segments_25m_midpoints.shp")
LAYER_NAME     = "segments_25m"

# --- DETREND PROFILE ---
def detrend_profile(x_m, z_m, method="poly3", savgol_win_m=6.0, dx=0.03):
    if method == "poly3":
        coeffs = np.polyfit(x_m, z_m, 3)
        trend = np.polyval(coeffs, x_m)
        return z_m - trend, trend
    elif method == "savgol":
        win = int(round(savgol_win_m / dx))
        if win % 2 == 0: win += 1
        win = max(win, 5)
        trend = savgol_filter(z_m, window_length=win, polyorder=3, mode="interp")
        return z_m - trend, trend
    else:
        return z_m.copy(), np.zeros_like(z_m)

z_detrended, trend = detrend_profile(x_m, z_m, method="poly3", dx=PROFILE_STEP_M)

# --- QUARTER-CAR IRI MODEL ---
m_s = 250.0; m_u = 30.0
k_s = 63300.0; k_t = 653000.0; c_s = 6000.0

def compute_iri(profile_z, dx_m, speed_kmh):
    v = speed_kmh / 3.6
    dt = dx_m / v
    z_s = z_u = z_s_dot = z_u_dot = 0.0
    acc_abs_susp_vel = 0.0
    for z_r in profile_z:
        z_s_ddot = (-k_s*(z_s - z_u) - c_s*(z_s_dot - z_u_dot)) / m_s
        z_u_ddot = ( k_s*(z_s - z_u) + c_s*(z_s_dot - z_u_dot) - k_t*(z_u - z_r)) / m_u
        z_s_dot += z_s_ddot * dt; z_u_dot += z_u_ddot * dt
        z_s += z_s_dot * dt; z_u += z_u_dot * dt
        acc_abs_susp_vel += abs(z_s_dot - z_u_dot) * dt
    iri_m_per_m = acc_abs_susp_vel / (len(profile_z) * dx_m)
    return iri_m_per_m * 1000.0   # m/km

IRI_total = compute_iri(z_detrended, PROFILE_STEP_M, IRI_SPEED_KMH)

# --- PER-SEGMENT IRI ---
def segment_stats(x_m, z_detrended, seg_len_m, dx_m, speed_kmh):
    n_per_seg = max(1, int(round(seg_len_m / dx_m)))
    segs = []
    for a in range(0, len(z_detrended), n_per_seg):
        b = min(len(z_detrended), a + n_per_seg)
        if b - a < 10: continue
        x_seg = x_m[a:b]; z_seg = z_detrended[a:b]
        iri = compute_iri(z_seg, dx_m, speed_kmh)
        rms_h = np.sqrt(np.mean(z_seg**2))
        rms_slope = np.sqrt(np.mean(np.gradient(z_seg, dx_m)**2))
        segs.append({
            "start_m": float(x_seg[0]),
            "end_m": float(x_seg[-1]),
            "IRI_m_per_km": float(iri),
            "RMS_height_cm": float(rms_h*100),
            "RMS_slope": float(rms_slope)
        })
    return segs

segs = segment_stats(x_m, z_detrended, SEG_LEN_M, PROFILE_STEP_M, IRI_SPEED_KMH)

# --- PRINT RESULTS ---
print(f"Overall IRI: {IRI_total:.2f} m/km @ {IRI_SPEED_KMH:.0f} km/h")
for s in segs:
    print(f"{s['start_m']:.1f}-{s['end_m']:.1f} m | IRI {s['IRI_m_per_km']:.2f} m/km | "
          f"RMS_h {s['RMS_height_cm']:.2f} cm | RMS_slope {s['RMS_slope']:.5f}")

# --- PLOTS ---
plt.figure(figsize=(10,4))
plt.plot(x_m, z_m, label="Raw elevation")
plt.plot(x_m, trend, label="Trend")
plt.xlabel("Distance (m)"); plt.ylabel("Elevation (m)")
plt.title("Raw Profile + Trend")
plt.legend(); plt.show()

plt.figure(figsize=(10,4))
plt.plot(x_m, z_detrended, label="Detrended profile")
plt.xlabel("Distance (m)"); plt.ylabel("Height (m)")
plt.title("Detrended Longitudinal Profile")
plt.legend(); plt.show()

# Bar chart of segment IRI
plt.figure(figsize=(10,4))
xs = [(s["start_m"]+s["end_m"])/2 for s in segs]
iri_vals = [s["IRI_m_per_km"] for s in segs]
plt.bar(xs, iri_vals, width=SEG_LEN_M*0.8, align="center")
plt.xlabel("Distance (m)"); plt.ylabel("IRI (m/km)")
plt.title(f"IRI per {SEG_LEN_M} m Segment")
plt.show()



# -------- OVERALL SUMMARY ----------
dz = np.gradient(z_detrended, PROFILE_STEP_M)
RMS_height_m = float(np.sqrt(np.mean(z_detrended**2)))
Std_height_m = float(np.std(z_detrended))
RMS_slope    = float(np.sqrt(np.mean(dz**2)))
IRI_total    = compute_iri(z_detrended, PROFILE_STEP_M, IRI_SPEED_KMH)

print("=== OVERALL ROUGHNESS ===")
print(f"Length          : {x_m[-1]-x_m[0]:.2f} m")
print(f"IRI (m/km)      : {IRI_total:.2f}")
print(f"RMS height (cm) : {RMS_height_m*100:.2f}")
print(f"Std. height (cm): {Std_height_m*100:.2f}")
print(f"RMS slope (m/m) : {RMS_slope:.5f}")

# -------- PER-SEGMENT STATS ----------
def segment_stats_table(x_m, z_detrended, seg_len_m, dx_m, speed_kmh):
    n_per_seg = max(1, int(round(seg_len_m / dx_m)))
    rows = []
    seg_id = 1
    for a in range(0, len(z_detrended), n_per_seg):
        b = min(len(z_detrended), a + n_per_seg)
        if b - a < 10: continue
        xs = x_m[a:b]; zs = z_detrended[a:b]
        iri = compute_iri(zs, dx_m, speed_kmh)
        dz = np.gradient(zs, dx_m)
        rows.append({
            "seg_id"     : seg_id,
            "start_m"    : float(xs[0]),
            "end_m"      : float(xs[-1]),
            "length_m"   : float(xs[-1] - xs[0]),
            "mid_m"      : float(0.5*(xs[0]+xs[-1])),
            "IRI"        : float(iri),
            "RMS_h_m"    : float(np.sqrt(np.mean(zs**2))),
            "STD_h_m"    : float(np.std(zs)),
            "RMS_slope"  : float(np.sqrt(np.mean(dz**2))),
        })
        seg_id += 1
    return pd.DataFrame(rows)

seg_df = segment_stats_table(x_m, z_detrended, SEG_LEN_M, PROFILE_STEP_M, IRI_SPEED_KMH)

# -------- SAVE CSV ----------
seg_df.to_csv(CSV_PATH, index=False)
print(f"Saved CSV → {CSV_PATH}")

# -------- SEGMENT GEOMETRIES ----------
geoms = []
for _, r in seg_df.iterrows():
    g = substring(line, r["start_m"], r["end_m"], normalized=False)
    geoms.append(g)

gseg = gpd.GeoDataFrame(seg_df.copy(), geometry=geoms, crs="EPSG:32619")  # update CRS if needed

# Midpoints
midpoints = [substring(line, d, d, normalized=False) for d in seg_df["mid_m"]]
gpts = gpd.GeoDataFrame(seg_df[["seg_id","mid_m","IRI"]],
                        geometry=[Point(p.coords[0]) for p in midpoints],
                        crs=gseg.crs)

# -------- SAVE SHAPEFILES ----------
gseg.to_file(SHP_SEG_PATH)
gpts.to_file(SHP_PTS_PATH)

print(f"Saved shapefiles → {SHP_SEG_PATH}, {SHP_PTS_PATH}")



#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------






















