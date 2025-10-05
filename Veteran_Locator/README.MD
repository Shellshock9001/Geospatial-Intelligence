# ShellShockHive — Advanced Locator

**Daylight / Weather / Traffic–Aware Multi-Stage Routing System**  
**Built for Law Enforcement, Search & Rescue, and Emergency Coordination Teams**  
**Token-free · No PII · Fully Local Execution**

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [What It Does NOT Do](#what-it-does-not-do)
- [System Architecture](#system-architecture)
- [Data Sources](#data-sources)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Option A: Python venv](#option-a-python-venv)
  - [Option B: Conda](#option-b-conda)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [Operator Workflow](#operator-workflow)
- [Map Legend](#map-legend)
- [How Routing Works](#how-routing-works)
- [Scoring Model](#scoring-model)
- [Optional: PostGIS Persistence](#optional-postgis-persistence)
- [Performance & Limits](#performance--limits)
- [Security, Privacy, Legal](#security-privacy-legal)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Appendix A: requirements.txt](#appendix-a-requirementstxt)
- [Appendix B: .env Template](#appendix-b-env-template)
- [License](#license)

---

## Overview

ShellShockHive is a **geospatial intelligence application** that generates **evidence-weighted movement corridors** from a **last-known location** to **likely destinations** such as VA hospitals, shelters, or transport hubs.

It models environmental and infrastructure reality (daylight, weather, traffic context, public camera nodes) to rank which routes deserve priority attention.

All default data sources are **public** and **token-free**. The app runs locally and stores no personal identifiers.

---

## Key Features

- **Multi-Stage Routing (beyond 10 km):** Long-haul (vehicle) tiles stitched together, plus high-detail last-mile (walk/drive).
- **Environment-Aware:** Daylight (Astral) + Weather (Open-Meteo) adjust realism and visibility assumptions.
- **Traffic Context:** Heuristics by hour-of-day; optional plug-ins for live traffic providers.
- **Camera/Facility Context:** Overlays OSM “surveillance/camera” points and official VA facility locations.
- **Scoring Engine:** Combines camera coverage, proximity, mobility mode, daylight, and weather into a 0–1 score.
- **Token-Free Defaults:** Works out-of-the-box without API keys.
- **Optional PostGIS:** Persist camera layers and audit runs in PostgreSQL/PostGIS.

---

## What It Does NOT Do

- Does **not** access private camera systems.  
- Does **not** identify people or process biometrics.  
- Does **not** use or store PII.  
- Does **not** predict individual intent; it prioritizes corridors based on observable context.

---

## System Architecture

```
Streamlit UI
   │
   ├─ Routing Engine (OSMnx + NetworkX)
   │     └─ Multi-tile long-haul + last-mile switching
   │
   ├─ Scoring Engine
   │     ├─ Daylight (Astral)
   │     ├─ Weather (Open-Meteo)
   │     ├─ Traffic heuristics
   │     └─ Camera and facility proximity
   │
   └─ Visualization Layer (Folium in Streamlit)
```

---

## Data Sources

| Source | Purpose | Public URL |
|--------|----------|------------|
| OpenStreetMap | Routing graphs (drive/walk) | [https://www.openstreetmap.org](https://www.openstreetmap.org) |
| Open-Meteo | Weather conditions (precipitation, visibility, temperature) | [https://open-meteo.com](https://open-meteo.com) |
| Astral | Sun position for daylight scoring | [https://pypi.org/project/astral/](https://pypi.org/project/astral/) |
| Overpass API | Public camera and surveillance data | [https://overpass-api.de/](https://overpass-api.de/) |
| VA.gov Open Data | VA hospitals and shelters | [https://data.va.gov](https://data.va.gov) |

---

## Requirements

- Python **3.10+**
- pip
- Streamlit
- Folium / Streamlit-Folium
- OSMnx
- NetworkX
- Geopy
- Astral
- Psycopg2 (for PostGIS, optional)
- Pandas
- AIOHTTP

---

## Installation

### Option A: Python venv

```bash
git clone https://github.com/<your-org>/shellshockhive-locator.git
cd shellshockhive-locator
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### Option B: Conda

```bash
conda create -n shellshockhive python=3.10
conda activate shellshockhive
pip install -r requirements.txt
```

---

## Configuration

1. Optional: set up PostGIS if persistence is needed:
   ```bash
   createdb veteran_locator_national
   psql -d veteran_locator_national -c "CREATE EXTENSION postgis;"
   ```

2. Configure environment variables if needed:
   ```
   PGHOST=localhost
   PGUSER=postgres
   PGPASSWORD=password
   PGDATABASE=veteran_locator_national
   ```

---

## Running the App

```bash
streamlit run shellshockhive_locator.py
```

Then open your browser at [http://localhost:8501](http://localhost:8501)

---

## Operator Workflow

1. Enter **Last-Known Coordinates**
2. Enter **Partial Facility ID or Name**
3. Add **Target Addresses**
4. Select **Mobility Mode**
5. Generate **Corridors**
6. Review scores and map visualization
7. Upload **Evidence Files** (auto-hashed for integrity)

---

## Map Legend

| Symbol | Meaning |
|---------|----------|
| 🔴 | Last-known location |
| 🟢 | Cameras with high uptime |
| 🟡 | Cameras with limited coverage |
| 🔵 | High-confidence corridor |
| 🟣 | Medium-confidence corridor |
| 🔻 | Reverse or backtrack path |

---

## How Routing Works

- Uses **OSMnx** to load road/footpath graphs.  
- Expands to **multi-stage graphs** when distances exceed ~10km (vehicle → walk).  
- Runs **Dijkstra shortest paths** and **k-shortest simple paths** between start and targets.  
- Generates **LineStrings** to visualize on Folium map.

---

## Scoring Model

| Factor | Description | Weight |
|--------|--------------|--------|
| Camera Coverage | Cameras along corridor | 0.40 |
| Proximity | Distance from last-known position | 0.25 |
| Mobility Mode | Based on vehicle or foot | 0.15 |
| Daylight | Penalizes low light | 0.10 |
| Weather | Penalizes low visibility / rain | 0.10 |

Final Score = Σ(weight × normalized_factor)

---

## Optional PostGIS Persistence

If enabled, the app saves:
- Camera metadata
- Facility coordinates
- Last-seen timestamps
- Corridor geometries

---

## Performance & Limits

- OSMnx graph capped at ~10km radius per tile; ShellShockHive stitches multiple tiles for long-range searches.
- Typical runtime: 10–20 seconds for full generation with scoring.
- Dependent on network latency for Overpass and Open-Meteo.

---

## Security, Privacy, Legal

- All data is **open-source** and **public domain**.
- No personal identifiers are ever stored.
- Evidence uploads are SHA-256 hashed locally.
- Intended for **governmental and emergency use**.

---

## Troubleshooting

| Problem | Solution |
|----------|-----------|
| App won’t start | Check Python and Streamlit versions |
| Empty map | Verify coordinates and internet connection |
| DB connection errors | Ensure PostgreSQL is running and credentials are correct |
| Missing cameras | Overpass API rate-limited; retry in 1–2 minutes |

---

## FAQ

**Q:** Does it require an API key?  
**A:** No. All data sources are public.

**Q:** How is weather data fetched?  
**A:** Through Open-Meteo’s public JSON endpoints.

**Q:** Can it run offline?  
**A:** Partial functionality (without live weather/camera data) works if cached.

---

## Appendix A: requirements.txt

```
streamlit
streamlit-folium
folium
osmnx
networkx
geopy
astral
aiohttp
psycopg2-binary
pandas
requests
```

---

## Appendix B: .env Template

```
PGHOST=localhost
PGUSER=postgres
PGPASSWORD=password
PGDATABASE=veteran_locator_national
```

---

## License

Licensed under the Apache License 2.0.  
See [LICENSE](LICENSE) for details.
