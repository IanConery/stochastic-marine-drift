
Stochastic Marine Drift (SMD)

A bi-directional, ensemble-based machine learning and physics framework designed to locate submerged archaeological sites and historical shipwrecks in the Caribbean Basin.
By decoupling chaotic environmental vectors (wind, surface shear, hurricanes, and deep-water currents) into distinct modules, this system simulates maritime drift trajectories using stochastic Monte Carlo simulations to generate highly accurate 2D seafloor probability maps.

🏗️ System Architecture & Specialization

This repository implements a three-part modular toolkit that bridges the gap between historical archival text, paleoclimate data, and fluid dynamics:
                  ┌──────────────────────────────────────┐
                  │      LAST KNOWN POSITION & TIME      │
                  └──────────────────┬───────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   WIND MODULE   │         │ SURFACE CURRENT │         │  EXTREME EVENT  │
│  (High Chaos /  │         │(Moderate Chaos /│         │ (Binary Filter /│
│  Weibull Noise) │         │Ekman Transport) │         │Non-Linear Drift)│
└────────┬────────┘         └────────┬────────┘         └────────┬────────┘
         │                           │                           │
         └─────────────────┬─────────┴───────────────────────────┘
                           ▼
              ┌─────────────────────────┐
              │  SURFACE DRIFT ENGINE   │ ──► Computes Point-of-Sink (SPS)
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │    DEEP WATER COLUMN    │ ──► Calculates 3D trajectory 
              │  (Low Chaos / Tidal)    │     using Terminal Descent Profiles
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   FINAL SEAFLOOR MAP    │ ──► Verified via Bayesian 
              └─────────────────────────┘     Constraint Filters

The Model Ecosystem

1. The Forward Drift Forecaster (Chaos-Ratio Engine): Simulates a vessel floating on the surface from its Last Known Position (LKP). It uses a stochastic particle filter, applying high-variance Gaussian noise to windage vectors and moderate noise to Ekman currents, continuously updating local wave conditions until a dynamic hull-fatigue sinking threshold is breached.
2. The Shipwreck Backtracker (Inverse-Time Optimizer): Reverses the arrow of time. Starting from known seafloor coordinates, it computes a Terminal Ascent Profile through deep-water columns (\(-\vec{V}_{deep}\)) to find the Surface Point of Sink (SPS). From there, it fan-searches backward through surface vectors using a Bayesian framework to bound the search area and reconstruct missing historical timelines.
3. The Foundering Classifier (Supervised Threshold Predictor): Uses the output of backtracked, discovered wrecks to train a Random Survival Forest. It isolates the non-linear thresholds (e.g., continuous wave height vs. hull mass) that trigger a ship's transition from surface drift to deep-water sinking.

📦 Data Feature Engineering & Sources

To execute without expensive physical surveys, this framework utilizes entirely open-source, public reanalysis data:
• Atmospheric Vectors (\(U_{10}, V_{10}\)): Hourly wind speed and significant wave height grids extracted from ECMWF ERA5.
• Hydrodynamic Vectors (\(\vec{V}_{current}\)): Multi-depth ocean current velocity vectors obtained via the Copernicus Marine Service.
• 3D Seafloor Topography: High-resolution digital elevation models from GEBCO (General Bathymetric Chart of the Oceans) used to map wreck settling locations and debris field scattering.
• Extreme Weather Anchors: Historical storm paths, maximum winds, and barometric pressures queried from the NOAA IBTrACS database.
• Paleoclimate Calibration: Annually resolved coral geochemical proxy records from NOAA NCEI Paleo Data to scale back modern baselines to the cooler, denser Caribbean water profiles of the 1700s.
• Ground Truth Labels & Cargo NLP: Core tabular frames built via Wrecksite.eu and the NOAA Ocean Explorer Project, processed through an LLM agent script to parse messy 18th-century logbook descriptions into structured numerical parameters (buoyancy_factor, cargo_mass).

🚀 Dependencies & Technology Stack

The pipeline is written entirely in Python, utilizing specialized geospatial and maritime frameworks:
bash
# Core Simulation & Particle Tracking
pip install opendrift oceanparcels

# Multi-dimensional Data Wrangling (NetCDF & GeoTIFF)
pip install xarray netCDF4 rasterio

# Machine Learning & Probability Mapping
pip install scikit-learn xgboost scipy
Use code with caution.
• OpenDrift: Drives the core physics implementation, natively ingestion Copernicus and ERA5 NetCDF files to compute leeway coefficients.
• MaxEnt (Dismo): Used as a baseline spatial suitability classifier to map geographical choke points.

📂 Repository Structure

text
├── data/
│   ├── raw/                # Placeholder for raw NetCDF weather/ocean files
│   └── processed/          # Standardized tabular datasets containing wreck profiles
├── src/
│   ├── agents/             # LLM scraping scripts for text-to-feature classification
│   ├── modules/
│   │   ├── wind.py         # High-chaos atmospheric vector updates
│   │   ├── current.py      # Ekman surface layer mechanics
│   │   └── deep_water.py   # 3D descent/ascent physics matrices
│   ├── forward_drift.py    # Monte Carlo simulation script
│   └── backtrack.py        # Inverse-time Lagrangian optimization script
├── notebooks/              # Exploration, EDA, and model validation plots
├── README.md
└── requirements.txt
Use code with caution.

🛠️ Verification & Next Steps

Current developments are focused on optimizing the Reverse-Time Stochastic Framework to prevent numerical diffusion (exponential search area blow-up) by constraining backward velocity vectors based on historical ship performance limits. Model validation is executed dynamically by running known Caribbean wreck coordinates through the backtracker to check the distance error between the simulated Surface Point of Sink and documented archival logs.
Would you like to build out the src/modules/wind.py script next to establish your high-chaos Weibull noise generation, or do you want to create the LLM agent prompt template to begin extracting numerical cargo metrics from your raw text descriptions?
