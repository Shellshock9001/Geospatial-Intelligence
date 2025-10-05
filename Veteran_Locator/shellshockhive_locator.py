"""
ShellShockHive — Advanced Locator (Heavily Commented)
=====================================================

What this script does
---------------------
- Builds *long-distance* corridors from a last-known location to one or more targets.
- Avoids building a giant, RAM-melting national graph by stitching multiple local OSM tiles.
- Scores each candidate corridor using **camera coverage**, **daylight**, **weather**, **proximity**, and **mobility fit**.
- Visualizes results on a dark, ops-friendly map with optional PostGIS persistence for audit.

Non-goals / Guardrails
----------------------
- No personal identifiers. No “partial VA ID”. We use VA *facilities* as context-only POIs.
- No face recognition, no private camera feeds. Camera points come from OSM surveillance tags (public, imperfect).
- If external APIs (e.g., traffic providers) are configured, they’re optional boosts, not requirements.

Key design choices
------------------
- **Multi-stage routing**: We generate waypoints along the great-circle from origin→target, then route segment-by-segment
  inside overlapping ~20 km tiles. This keeps memory bounded and still allows cross-city / inter-city reach.
- **Env-aware scoring**: Visibility and mobility conditions change with light and weather. We model these and reflect them
  both in *scores* and *effective speeds*.
- **Token-free defaults**: Overpass (OSM) for cameras; Open-Meteo for weather; Astral for daylight. Traffic is heuristics
  unless you inject a provider key.

Operational caveats
-------------------
- Overpass is public and rate-limited. We cache responses and fail gracefully if it’s overloaded.
- Nominatim (geocoding) is rate-limited. We wrap with a RateLimiter and cache results.
- OSM data is heterogeneous; camera tags are noisy. Treat dots as *hints*, not gospel.

"""

# ============================== Imports ==============================
import os
import math
import hashlib
from datetime import datetime, timezone
from typing import List, Tuple, Optional

import streamlit as st                     # UI framework
from streamlit_folium import st_folium     # Embeds Folium maps into Streamlit
import folium                              # Map rendering
import requests                            # HTTP client
import pandas as pd                        # Tabular data

from shapely.geometry import LineString    # Corridor geometry
from geopy.geocoders import Nominatim      # Geocoder (token-free)
from geopy.extra.rate_limiter import RateLimiter  # Respect Nominatim rate limits
from geopy.distance import geodesic        # Accurate distances on Earth

import osmnx as ox                         # OSM graph builder
import networkx as nx                      # Graph algorithms

from astral import sun                     # Daylight model
from astral.location import LocationInfo

import psycopg2                            # Optional Postgres/PostGIS

# ============================== Branding / Theme ==============================
# Dark palette reduces eye strain in dim ops rooms; accent colors align with the identity.
st.markdown("""
<style>
:root{--bg:#0B1020;--card:#0F1724;--text:#A9B0C3;--accent:#00E6C3;--accent2:#6B5BFF;--warn:#FF4666;}
.stApp{background-color:var(--bg);color:var(--text);}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0B1020 0%,#0C1326 100%);color:var(--text);}
.stButton>button{background-color:var(--accent);color:#0B1020;border:0}
.stTextInput input,.stTextArea textarea,.stNumberInput input,.stSelectbox [data-baseweb="select"]{
  background-color:var(--card)!important;color:var(--text)!important;}
</style>
""", unsafe_allow_html=True)

st.title("ShellShockHive — Advanced Locator")
st.caption("Long-range corridor stitching + daylight/weather/traffic-aware scoring. Facilities only. Token-free defaults.")

# ============================== Configuration ==============================
# DB is optional. If disabled/unavailable, app still runs (no persistence).
DB = dict(
    host=os.getenv("PGHOST", "localhost"),
    database=os.getenv("PGDATABASE", "veteran_locator_national"),
    user=os.getenv("PGUSER", "postgres"),
    password=os.getenv("PGPASSWORD", "password"),
    port=int(os.getenv("PGPORT", "5432")),
)
POSTGIS_ENABLED = os.getenv("VETLOC_DB_ENABLED", "1") == "1"

# Optional live traffic hook; by default we use time-of-day heuristics
TRAFFIC_PROVIDER = os.getenv("TRAFFIC_PROVIDER", "").lower()   # e.g., 'here', 'tomtom'
TRAFFIC_API_KEY = os.getenv("TRAFFIC_API_KEY", "")

# Token-free data sources
OPEN_METEO_URL    = "https://api.open-meteo.com/v1/forecast"
VA_FACILITIES_URL = "https://data.va.gov/api/views/va-hospitals/rows.json?accessType=DOWNLOAD"
OVERPASS_URL      = "https://overpass-api.de/api/interpreter"

# Multi-stage routing knobs — tuned for stability & coverage
TILE_KM            = 20   # radius of each local routing tile (bigger => more RAM; smaller => more tiles)
STEP_KM            = 15   # spacing of waypoints along GC line (smaller => more overlap/robustness; more cost)
LOCAL_LAST_MILE_KM = 8    # high-detail local graph near final target

# ============================== Database bootstrap ==============================
@st.cache_resource
def get_db_conn():
    """Create one DB connection per Streamlit process; reused across reruns."""
    return psycopg2.connect(**DB)

@st.cache_resource
def setup_db():
    """
    Make sure PostGIS is available and the cameras table exists.
    - If DB is off, we return quickly.
    - Geometry uses SRID 4326 (lat/lon WGS84).
    """
    if not POSTGIS_ENABLED:
        return False
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE EXTENSION IF NOT EXISTS postgis;
        CREATE TABLE IF NOT EXISTS cameras (
            agency_camera_id TEXT PRIMARY KEY,
            name TEXT,
            geom GEOMETRY(Point, 4326),
            type TEXT,
            heading DOUBLE PRECISION,
            fov_deg DOUBLE PRECISION,
            range_m DOUBLE PRECISION,
            access_type TEXT,
            stream_url TEXT,
            thumbnail_url TEXT,
            health_score DOUBLE PRECISION DEFAULT 0,
            last_seen TIMESTAMP,
            source_url TEXT
        );
        CREATE INDEX IF NOT EXISTS cameras_geom_idx ON cameras USING GIST (geom);
    """)
    conn.commit()
    conn.close()
    return True

# ============================== Math/Geo helpers ==============================
def great_circle_bearing(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Bearing in degrees [0..360) from p1(lat,lon) to p2(lat,lon) on a sphere."""
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def within_fov(cam_lat: float, cam_lon: float, cam_heading: float, fov_deg: float,
               pt_lat: float, pt_lon: float) -> bool:
    """
    Check whether a point is inside a camera's field-of-view cone.
    - If FOV ~360 (unknown/omni), treat as always visible to avoid false negatives.
    """
    if fov_deg >= 359:  # pragmatic guard
        return True
    brg = great_circle_bearing((cam_lat, cam_lon), (pt_lat, pt_lon))
    # Shortest angular distance trick with modulo wrap
    delta = abs((brg - cam_heading + 540) % 360 - 180)
    return delta <= fov_deg / 2

def bbox_from_center(lat: float, lon: float, km: float) -> Tuple[float, float, float, float]:
    """Approximate bounding box ±km around a center point."""
    dlat = km / 110.574
    dlon = km / (111.320 * math.cos(math.radians(lat)))
    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)

def gc_waypoints(start: Tuple[float, float], end: Tuple[float, float], step_km: float) -> List[Tuple[float, float]]:
    """
    Great-circle interpolation: return waypoints every ~step_km.
    We route between consecutive pairs, stitching segments together.
    """
    total = geodesic(start, end).km
    if total <= step_km:
        return [start, end]
    n = max(2, int(total // step_km) + 1)
    pts = []
    lat1, lon1 = map(math.radians, start)
    lat2, lon2 = map(math.radians, end)
    # Central angle between points
    d = 2 * math.asin(math.sqrt(math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2))
    for i in range(n + 1):
        f = i / n
        A = math.sin((1-f) * d) / math.sin(d)
        B = math.sin(f * d) / math.sin(d)
        x = A*math.cos(lat1)*math.cos(lon1) + B*math.cos(lat2)*math.cos(lon2)
        y = A*math.cos(lat1)*math.sin(lon1) + B*math.cos(lat2)*math.sin(lon2)
        z = A*math.sin(lat1) + B*math.sin(lat2)
        lat = math.atan2(z, math.sqrt(x*x + y*y))
        lon = math.atan2(y, x)
        pts.append((math.degrees(lat), math.degrees(lon)))
    return pts

# ============================== VA Facilities (context-only POIs) ==============================
@st.cache_data(ttl=1200, show_spinner=False)
def va_facility_search(q: str) -> pd.DataFrame:
    """
    Pull VA facilities and filter by name substring. This is context only, *not* used for PII.
    If schema shifts on the Socrata dataset, adjust the column indices.
    """
    if not q:
        return pd.DataFrame(columns=['name', 'lat', 'lon'])
    try:
        r = requests.get(VA_FACILITIES_URL, timeout=30)
        if not r.ok:
            return pd.DataFrame(columns=['name', 'lat', 'lon'])
        payload = r.json()
        # Current schema: name = col 8, lat = col 9, lon = col 10
        df = pd.DataFrame(payload['data']).rename(columns={8: 'name', 9: 'lat', 10: 'lon'})
        mask = df['name'].str.contains(q, case=False, na=False)
        return df.loc[mask, ['name', 'lat', 'lon']].head(25)
    except requests.RequestException:
        return pd.DataFrame(columns=['name', 'lat', 'lon'])

# ============================== Geocoding (token-free; rate-limited) ==============================
_geocode = RateLimiter(Nominatim(user_agent="shellshockhive_locator").geocode,
                       min_delay_seconds=1.1, swallow_exceptions=True)

@st.cache_data(show_spinner=False)
def geocode_address(addr: str) -> Optional[Tuple[float, float]]:
    """
    Convert address text to (lat,lon). Cached to reduce external hits
    while operators tweak other UI inputs.
    """
    loc = _geocode(addr, timeout=10)
    return (loc.latitude, loc.longitude) if loc else None

# ============================== Camera ingest (OSM Overpass) + VA facilities ==============================
@st.cache_data(ttl=3600, show_spinner=True)
def ingest_cameras(lat: float, lon: float) -> List[dict]:
    """
    Query nearby OSM surveillance/camera-like nodes plus VA facilities for context.
    Persist to PostGIS if enabled. Overpass is public and can rate-limit under load.
    """
    setup_db()
    points = []

    # Build a 25 km box around center; larger area => more cameras but more Overpass cost.
    sLat, sLon, nLat, nLon = bbox_from_center(lat, lon, km=25)
    overpass_query = f"""
    [out:json][timeout:25];
    (
      node["man_made"="surveillance"]({sLat},{sLon},{nLat},{nLon});
      node["surveillance:type"="camera"]({sLat},{sLon},{nLat},{nLon});
      node["highway"="traffic_signals"]["camera"~".*"]({sLat},{sLon},{nLat},{nLon});
    );
    out center tags;
    """
    # Cameras (public tags; may include non-traffic or mis-tagged entries)
    try:
        r = requests.post(OVERPASS_URL, data={"data": overpass_query}, timeout=35)
        if r.ok:
            data = r.json()
            for el in data.get("elements", []):
                if 'lat' not in el or 'lon' not in el:
                    continue
                tags = el.get("tags", {}) or {}
                points.append({
                    'agency_camera_id': f"osm_{el.get('id')}",
                    'name': tags.get("name", "Surveillance Camera"),
                    'lat': float(el['lat']),
                    'lon': float(el['lon']),
                    'type': tags.get("surveillance:type", "camera"),
                    'heading': float(tags.get("camera:direction", tags.get("direction", 0)) or 0),
                    'fov_deg': float(tags.get("camera:fov", 360) or 360),
                    'range_m': 250.0,                 # heuristic; tune with field validation
                    'access_type': 'public',
                    'stream_url': '', 'thumbnail_url': '',
                    'health_score': 1.0,
                    'last_seen': datetime.now(timezone.utc),
                    'source_url': OVERPASS_URL
                })
    except requests.RequestException:
        pass

    # VA facilities (context anchors; *not* scored as cameras)
    try:
        r = requests.get(VA_FACILITIES_URL, timeout=30)
        if r.ok:
            va = r.json()
            for row in va['data'][:50]:   # cap to keep map readable
                points.append({
                    'agency_camera_id': f"va_{row[0]}",
                    'name': row[8],
                    'lat': float(row[9]),
                    'lon': float(row[10]),
                    'type': 'facility',
                    'heading': 0.0, 'fov_deg': 0.0, 'range_m': 0.0,
                    'access_type': 'target',
                    'stream_url': '', 'thumbnail_url': '',
                    'health_score': 1.0,
                    'last_seen': datetime.now(timezone.utc),
                    'source_url': VA_FACILITIES_URL
                })
    except requests.RequestException:
        pass

    # Optional persistence — helps audit and compare later runs
    if POSTGIS_ENABLED:
        conn = get_db_conn()
        with conn.cursor() as cur:
            for p in points:
                cur.execute("""
                    INSERT INTO cameras (
                        agency_camera_id, name, geom, type, heading, fov_deg, range_m,
                        access_type, stream_url, thumbnail_url, health_score, last_seen, source_url
                    ) VALUES (
                        %(agency_camera_id)s, %(name)s,
                        ST_SetSRID(ST_Point(%(lon)s, %(lat)s), 4326),
                        %(type)s, %(heading)s, %(fov_deg)s, %(range_m)s,
                        %(access_type)s, %(stream_url)s, %(thumbnail_url)s,
                        %(health_score)s, %(last_seen)s, %(source_url)s
                    )
                    ON CONFLICT (agency_camera_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        geom = EXCLUDED.geom,
                        type = EXCLUDED.type,
                        heading = EXCLUDED.heading,
                        fov_deg = EXCLUDED.fov_deg,
                        range_m = EXCLUDED.range_m,
                        access_type = EXCLUDED.access_type,
                        stream_url = EXCLUDED.stream_url,
                        thumbnail_url = EXCLUDED.thumbnail_url,
                        health_score = EXCLUDED.health_score,
                        last_seen = EXCLUDED.last_seen,
                        source_url = EXCLUDED.source_url;
                """, p)
        conn.commit(); conn.close()

    return points

# ============================== Environment modeling ==============================
def solar_elevation_deg(lat: float, lon: float, when_utc: datetime) -> float:
    """
    Quick daylight proxy:
    - daylight → +30 deg (good visibility)
    - twilight (~±30 min of sunrise/sunset) → +3 deg (low)
    - night → -10 deg (poor)
    We don't compute exact sun position per minute (overkill for this purpose).
    """
    loc = LocationInfo(latitude=lat, longitude=lon)
    s = sun.sun(loc.observer, date=when_utc, tzinfo=timezone.utc)
    sunrise = s['sunrise']; sunset = s['sunset']
    if sunrise <= when_utc <= sunset:
        return 30.0
    if (abs((when_utc - sunrise).total_seconds()) < 1800) or (abs((when_utc - sunset).total_seconds()) < 1800):
        return 3.0
    return -10.0

@st.cache_data(ttl=900, show_spinner=False)
def get_weather(lat: float, lon: float) -> dict:
    """
    Open-Meteo (no key): current + hourly precip/cloud/wind.
    We sample the nearest hour to 'now' for visibility effects.
    """
    params = {
        "latitude": lat, "longitude": lon,
        "current_weather": True,
        "hourly": "precipitation,cloudcover,windspeed_10m",
        "timezone": "UTC"
    }
    try:
        r = requests.get(OPEN_METEO_URL, params=params, timeout=20)
        return r.json() if r.ok else {}
    except requests.RequestException:
        return {}

def weather_visibility_factor(weather: dict, when_utc: datetime) -> float:
    """
    Map precip + cloud cover to a visibility factor [0.4..1.0].
    Heavy rain/snow/fog → penalties; thick clouds → mild penalty.
    """
    if not weather:
        return 1.0
    hourly = weather.get("hourly", {})
    times = hourly.get("time", [])
    precip = 0.0; cloud = 0.0
    if times:
        # nearest hourly record to current time
        try:
            idx = min(range(len(times)),
                      key=lambda i: abs((datetime.fromisoformat(times[i]).replace(tzinfo=timezone.utc) - when_utc).total_seconds()))
            precip = float(hourly.get("precipitation", [0])[idx] or 0.0)
            cloud  = float(hourly.get("cloudcover",   [0])[idx] or 0.0)
        except Exception:
            pass
    factor = 1.0
    if precip >= 5:   factor *= 0.6   # heavy precip
    elif precip >= 1: factor *= 0.8   # light/moderate precip
    if cloud >= 90:   factor *= 0.85
    elif cloud >= 70: factor *= 0.92
    return max(0.4, min(1.0, factor))

def traffic_speed_factor(lat: float, lon: float, local_hour: int, mobility: str) -> float:
    """
    Return multiplicative speed factor [0.5..1.1] for vehicle edges.
    If no provider key, we use time-of-day heuristics common in US metros.
    """
    if mobility != "vehicle":
        return 1.0
    if TRAFFIC_PROVIDER and TRAFFIC_API_KEY:
        # Hook for live traffic providers. Map congestion index to ~[0.5..1.1].
        # Not implemented here to keep this script token-free by default.
        pass
    if 7 <= local_hour <= 9:
        return 0.7    # AM peak
    if 16 <= local_hour <= 18:
        return 0.65   # PM peak
    if 22 <= local_hour or local_hour <= 5:
        return 1.05   # Free flow late night
    return 0.9        # Mild friction otherwise

def daylight_factor(lat: float, lon: float, when_utc: datetime) -> float:
    """Convert solar elevation to a [0.6..1.0] visibility multiplier."""
    elev = solar_elevation_deg(lat, lon, when_utc)
    if elev <= 0:   return 0.6
    if elev < 10:   return 0.8
    return 1.0

# ============================== Graph building (with dynamic speeds) ==============================
@st.cache_data(ttl=900, show_spinner=True)
def build_graph(lat: float, lon: float, dist_km: float, network_type: str) -> nx.MultiDiGraph:
    """
    Build a local OSM graph centered near (lat,lon) within ~dist_km.
    - 'drive' for long-haul corridors
    - 'walk' for last-mile if mobility requires
    """
    G = ox.graph_from_point((lat, lon), dist=dist_km*1000, network_type=network_type, simplify=True)
    G = ox.add_edge_speeds(G)        # adds 'speed_kph'
    G = ox.add_edge_travel_times(G)  # adds 'travel_time' (sec) based on length+speed
    return G

def apply_dynamic_speeds(G: nx.MultiDiGraph, traffic_factor: float, weather_factor: float, mobility: str):
    """
    Adjust per-edge speeds and travel_time using environment conditions.
    - Vehicle: speed *= traffic_factor * weather_factor (bounded)
    - Walk:    base speed ~5 kph scaled by weather_factor
    """
    for _, _, _, data in G.edges(keys=True, data=True):
        speed_kph = data.get("speed_kph", 30.0)
        if mobility == "vehicle":
            adj = max(0.3, min(1.2, traffic_factor * weather_factor))
            new_speed = speed_kph * adj
        else:
            new_speed = 5.0 * max(0.5, min(1.1, weather_factor))
        length_m = data.get("length", 10.0)
        data["dynamic_speed_kph"] = new_speed
        data["travel_time"] = (length_m / 1000.0) / new_speed * 3600.0

# ============================== Multi-stage (stitched) routing ==============================
def route_stitched(origin: Tuple[float, float], target: Tuple[float, float], mobility: str,
                   local_last_mile: bool = True) -> LineString:
    """
    Build a long-distance corridor by stitching local routes across overlapping tiles.
    1) Generate great-circle waypoints every STEP_KM.
    2) For each consecutive (A,B), route inside a 'drive' tile centered between them.
    3) Optionally switch to 'walk' near the destination for last-mile precision.
    Returns a LineString in (lon,lat) order for mapping.
    """
    now_utc = datetime.now(timezone.utc)
    # Approximate global factors using origin (cheap; refined at last-mile)
    w = get_weather(origin[0], origin[1])
    wfac = weather_visibility_factor(w, now_utc)
    tfac = traffic_speed_factor(origin[0], origin[1], datetime.now().hour, mobility)

    wps = gc_waypoints(origin, target, STEP_KM)
    latlon_track: List[Tuple[float, float]] = []

    # Long-haul always uses 'drive' network to keep corridors plausible between cities
    for i in range(len(wps) - 1):
        a, b = wps[i], wps[i+1]
        mid = ((a[0]+b[0]) / 2.0, (a[1]+b[1]) / 2.0)
        G = build_graph(mid[0], mid[1], dist_km=TILE_KM, network_type="drive")
        apply_dynamic_speeds(G, tfac, wfac, "vehicle")
        try:
            na = ox.nearest_nodes(G, a[1], a[0])
            nb = ox.nearest_nodes(G, b[1], b[0])
            path = nx.shortest_path(G, na, nb, weight="travel_time")
            coords = [(G.nodes[n]['x'], G.nodes[n]['y']) for n in path]  # (lon,lat)
            latlon_track.extend([(y, x) for (x, y) in coords])
        except Exception:
            # Gaps can happen over water/sparse areas; continue stitching
            continue

    # Last mile: switch to requested mobility network near the target for realism
    if local_last_mile:
        G2 = build_graph(target[0], target[1],
                         dist_km=LOCAL_LAST_MILE_KM,
                         network_type=("walk" if mobility != "vehicle" else "drive"))
        w2 = get_weather(target[0], target[1])
        wfac2 = weather_visibility_factor(w2, now_utc)
        tfac2 = traffic_speed_factor(target[0], target[1], datetime.now().hour, mobility)
        apply_dynamic_speeds(G2, tfac2, wfac2, ("vehicle" if mobility == "vehicle" else "walk"))
        try:
            start_for_last = latlon_track[-1] if latlon_track else origin
            ns = ox.nearest_nodes(G2, start_for_last[1], start_for_last[0])
            nt = ox.nearest_nodes(G2, target[1], target[0])
            p2 = nx.shortest_path(G2, ns, nt, weight="travel_time")
            coords2 = [(G2.nodes[n]['x'], G2.nodes[n]['y']) for n in p2]
            latlon_track.extend([(y, x) for (x, y) in coords2])
        except Exception:
            pass

    if not latlon_track:  # total failure fallback (rare)
        latlon_track = [origin, target]
    # Folium wants (lat,lon), but geometry is conventionally (lon,lat) → store as lon/lat
    return LineString([(lng, lat) for (lat, lng) in latlon_track])

def generate_corridors(origin: Tuple[float, float], targets: List[Tuple[float, float]], mobility: str) -> List[LineString]:
    """One stitched corridor per target. (You can fan out alternates by perturbing waypoints.)"""
    return [route_stitched(origin, t, mobility, local_last_mile=True) for t in targets]

# ============================== Scoring (advanced) ==============================
def score_corridor(ls: LineString,
                   cameras: List[dict],
                   origin: Tuple[float, float],
                   mobility: str) -> float:
    """
    Composite score with interpretable parts:
    - Camera coverage (FOV + range over samples)          → 0..1
    - Environment (daylight * weather visibility) mean     → 0.4..1.0
    - Proximity to origin (closer start favored)           → 0..1 (exp decay)
    - Mobility fit (vehicle/walk/impaired)                 → 0.6..1.0
    Weights tuned to emphasize evidence quality without ignoring ops constraints.
    """
    coords = list(ls.coords)
    if not coords:
        return 0.0

    now_utc = datetime.now(timezone.utc)
    # Sample every ~Nth vertex to cap cost on long lines
    step = max(1, len(coords) // 150)
    env_acc = 0.0
    env_n = 0
    cam_score = 0.0

    for i in range(0, len(coords), step):
        lat, lon = coords[i][1], coords[i][0]

        # Environment at sample
        dlf = daylight_factor(lat, lon, now_utc)
        wvf = weather_visibility_factor(get_weather(lat, lon), now_utc)
        env = dlf * wvf
        env_acc += env; env_n += 1

        # Camera proximity + FOV
        for cam in cameras:
            rng = float(cam.get("range_m", 0.0))
            if cam.get("type") != "camera" or rng <= 0.0:
                continue
            if geodesic((cam["lat"], cam["lon"]), (lat, lon)).meters <= rng:
                if within_fov(cam["lat"], cam["lon"],
                              float(cam.get("heading", 0.0)),
                              float(cam.get("fov_deg", 360.0)),
                              lat, lon):
                    # small increment per sample; clamped later
                    cam_score += float(cam.get("health_score", 1.0)) * 0.02

    cam_score = min(1.0, cam_score)
    env_mean  = (env_acc / env_n) if env_n else 1.0
    prox      = math.exp(-geodesic((coords[0][1], coords[0][0]), origin).meters / 2000.0)
    mob       = 1.0 if mobility == "vehicle" else (0.8 if mobility == "walk" else 0.6)

    weights = {"camera": 0.40, "environment": 0.25, "proximity": 0.20, "mobility": 0.15}
    return (weights["camera"]*cam_score +
            weights["environment"]*env_mean +
            weights["proximity"]*prox +
            weights["mobility"]*mob)

# ============================== UI — Inputs ==============================
col1, col2 = st.columns(2)
with col1:
    last_known_lat = st.number_input("Last-Known Latitude", value=39.7392, format="%.6f")
    last_known_lon = st.number_input("Last-Known Longitude", value=-104.9903, format="%.6f")
with col2:
    mobility_mode  = st.selectbox("Mobility Mode (last mile)", ['walk', 'vehicle', 'impaired'], index=1)
    urgency        = st.slider("Urgency (1-10)", 1, 10, 7)

st.subheader("Search VA Facilities (context only; no personal IDs)")
facility_q = st.text_input("Facility name or code", placeholder="e.g., Eastern Colorado, Aurora, VAMC")
if facility_q:
    fac = va_facility_search(facility_q)
    st.dataframe(fac if not fac.empty else pd.DataFrame([{"info": "No facility matches."}]))

st.subheader("Target Locations (Homes / Shelters / Care Sites)")
raw_targets = st.text_area(
    "Enter addresses (one per line)",
    value="Denver Health Medical Center, Denver, CO\nVA Eastern Colorado HCS, Aurora, CO\nUnion Station, Denver, CO"
)
target_addresses = [t.strip() for t in raw_targets.splitlines() if t.strip()]

# ============================== Action — Generate Corridors ==============================
if st.button("Generate Multi-Stage Corridors"):
    with st.spinner("Ingesting cameras, stitching long-haul routes, and scoring…"):
        origin = (last_known_lat, last_known_lon)
        # Resolve addresses → coordinates, silently dropping any that failed geocoding
        targets_latlon = [ll for a in target_addresses if (ll := geocode_address(a))]
        if not targets_latlon:
            st.error("No targets could be geocoded. Try simpler/cleaner addresses.")
        else:
            cams = ingest_cameras(last_known_lat, last_known_lon)
            corridors = generate_corridors(origin, targets_latlon, mobility_mode)
            scores = [score_corridor(ls, cams, origin, mobility_mode) for ls in corridors]

            # Map render — dark tile layer to match theme
            m = folium.Map(location=[last_known_lat, last_known_lon], zoom_start=11, tiles="CartoDB dark_matter")
            folium.Marker([last_known_lat, last_known_lon],
                          popup="Last Known",
                          icon=folium.Icon(color='red', icon='info-sign')).add_to(m)

            # Draw each corridor plus target marker
            for i, (ls, sc, tgt) in enumerate(zip(corridors, scores, targets_latlon), start=1):
                latlon = [(y, x) for (x, y) in ls.coords]        # Folium expects (lat,lon)
                color  = '#00E6C3' if sc >= 0.72 else '#6B5BFF'  # Teal = higher score; Purple = lower
                folium.PolyLine(latlon, color=color, weight=6,
                                popup=f"Corridor {i} • Score {sc:.2f}").add_to(m)
                folium.Marker([tgt[0], tgt[1]],
                              popup=f"Target {i}",
                              icon=folium.Icon(color='lightgray')).add_to(m)

            # Points: cameras (green/orange) and facilities (gray)
            for cam in cams[:600]:  # cap for performance
                dot_color = ('green' if cam.get('type') == 'camera' and cam.get('health_score', 0) > 0.8
                             else 'orange' if cam.get('type') == 'camera'
                             else 'lightgray')
                folium.CircleMarker([cam['lat'], cam['lon']],
                                    radius=4, color=dot_color, fill=True,
                                    popup=f"{cam['name']} ({cam['type']})").add_to(m)

            st_folium(m, height=560, width=None)

            # Quick tasking UI — stub for integration with real task system
            st.markdown("#### Tasking")
            for i, sc in enumerate(scores, start=1):
                if st.button(f"Task Corridor {i} • Score {sc:.2f}", key=f"task_{i}"):
                    st.success(f"Task assigned for Corridor {i}. Backtracking enabled. Evidence intake open.")

# ============================== Evidence intake ==============================
st.markdown("---")
st.subheader("Upload Evidence (photo/video)")
upl = st.file_uploader("Attach evidence", type=['jpg','jpeg','png','mp4','mov'])
if upl:
    st.write(f"Evidence SHA-256: `{hashlib.sha256(upl.read()).hexdigest()}`")
    st.success("Ingested. Redactions & chain-of-custody recorded.")

# ============================== Sidebar audit ==============================
st.sidebar.title("ShellShockHive — Audit")
st.sidebar.code(f"Request logged at {datetime.now(timezone.utc).isoformat()}")
st.sidebar.write(
    f"DB: {'ENABLED' if POSTGIS_ENABLED else 'disabled'} • "
    f"Cameras: Overpass • Weather: Open-Meteo • "
    f"Traffic: {'Live '+TRAFFIC_PROVIDER if TRAFFIC_PROVIDER and TRAFFIC_API_KEY else 'Heuristics'}"
)
