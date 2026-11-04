# Stochastic Marine Drift (SMD)

A bi-directional, ensemble-based fluid dynamics and machine learning framework for reconstructing historical maritime drift trajectories and predicting shipwreck resting locations in the Caribbean Basin.

The framework couples forward Monte Carlo Lagrangian particle tracking with reverse-time trajectory reconstruction. Environmental forcing mechanisms—including atmospheric wind shear, surface currents, Stokes wave drift, episodic tropical cyclones, and deep-water currents—are decoupled into independent stochastic modules. This architecture addresses spatial-temporal scale differences, mitigates reverse-time numerical diffusion, and yields calibrated 2D seafloor probability density maps and 3D descent scatter footprints.

---

## Theoretical Foundations

Maritime search and salvage operations typically rely on forward leeway drift equations that assume known abandonment coordinates and timestamps. Historical maritime losses, particularly during the 17th through 19th centuries, lack these boundary conditions. Archival accounts rely on dead reckoning, imprecise chronologies, and qualitative storm notes.

SMD addresses these constraints through three structural methodologies:

1. **Decoupled Stochastic Forcing**: Environmental forces operate across distinct spatial and temporal scales. Windage carries high-frequency stochastic turbulence; ocean currents exhibit mesoscale stability; wave-induced Stokes drift provides independent mass transport; and extreme hurricane events impose highly non-linear cyclonic perturbations. Isolating these forces into dedicated numerical solvers prevents scale error propagation.
2. **Bi-Directional Optimization Loop**: Backtracking documented, discovered shipwrecks from seafloor coordinates calculates their Surface Point of Sink (SPS). Intersecting computed SPS coordinates with historical climate reanalyses extracts empirical environmental stress profiles (wave steepness, continuous wind shear, and storm duration) at the moment of loss. These synthesized profiles serve as supervised training labels for survival models that govern forward simulations.
3. **Continuous Adjoint Dynamics & Bayesian Bounds**: Standard reverse-time particle integration suffers from exponential numerical diffusion. SMD constrains the reverse search cone by combining Physics-Informed Neural Ordinary Differential Equations (PINN-ODEs) with empirical naval performance limits (maximum hull speed, leeway divergence angles, and sailing corridors).

---

## Subsystem Architecture

```
                       Last Known Position & Time
                    (Seeded via LLM Epistemic Priors)
                                   |
         +-------------------------+-------------------------+
         |                         |                         |
         v                         v                         v
+-----------------+       +-----------------+       +-----------------+
|   Wind Module   |       | Surface Current |       |  Extreme Events |
| (Weibull Noise) |       | (Ekman & Stokes)|       | (IBTrACS Tracks)|
+--------+--------+       +--------+--------+       +--------+--------+
         |                         |                         |
         +-------------------------+-------------------------+
                                   |
                                   v
                      +-------------------------+
                      |  Surface Drift Engine   | ---> Bathymetric Collision Check
                      +------------+------------+      (Depth <= Draft -> Grounded)
                                   |
                                   v
                      +-------------------------+
                      |   Foundering Trigger    | ---> Dynamic Hull Fatigue
                      +------------+------------+      Survival Model
                                   |
                                   v
                      +-------------------------+
                      |    Deep Water Column    | ---> 3D Hydrodynamic Descent
                      +------------+------------+      (Terminal Velocity & Glide)
                                   |
                                   v
                      +-------------------------+
                      |   Seafloor Deposition   | ---> Bivariate Gaussian Footprint
                      +-------------------------+      (Ballast Mound vs. Scatter)
```

### Core Subsystems

* **Forward Drift Simulator (`src/models/forward_drift.py`)**: Executes ensemble Monte Carlo particle integration using parameterized leeway coefficients, Stokes drift vectors, and stochastic turbulence. Incorporates a discrete crew behavioral state machine (`HEAVED_TO`, `DERELICT_DEAD_DRIFT`, `GROUNDED`, `FOUNDERED`).
* **Physics-Informed Reverse Backtracker (`src/models/neural_ode.py`, `src/models/backtrack.py`)**: Reverses time steps from known seafloor coordinates to compute the Surface Point of Sink. Constrains reverse velocity vectors via Neural ODE adjoint sensitivity and hydrodynamic boundary limits.
* **Dynamic Foundering Model (`src/models/foundering.py`)**: Uses a Random Survival Forest trained on backtracked historical wrecks to evaluate continuous hull fatigue integrals against cumulative wave stress ($H_s, T_p$), replacing arbitrary fixed time-to-sink heuristics.
* **Epistemic Source Uncertainty Calibrator (`src/agents/epistemic.py`)**: Natural language processing pipeline that evaluates archival logbook provenance, categorizing records into structured credibility tiers to seed initial spatial covariance matrices ($\Sigma_0$) and timestamp variances ($\sigma_t$).
* **Colonial Convoy Breakdown Network (`src/models/convoy_gnn.py`)**: Spatio-Temporal Graph Neural Network representing historical Caribbean shipping routes. Predicts fleet detachment probabilities and straggler drift vectors when storm anomalies cross convoy paths.
* **Bathymetric Grounding Filter (`src/modules/collision.py`)**: Evaluates instantaneous particle depth against high-resolution GEBCO elevation rasters, halting drift upon shallow reef or sandbar contact.
* **Debris Field Dispersion Model (`src/models/debris_field.py`)**: Generates dual-component bivariate Gaussian scatter footprints, separating high-density primary ballast mounds from lighter down-current debris trails.

---

## Environmental Data Ingestion

The framework ingests publicly available reanalysis and observational datasets:

| Data Layer | Source | Spatial / Temporal Resolution | Model Parameter |
| :--- | :--- | :--- | :--- |
| Atmospheric Forcing | ECMWF ERA5 Reanalysis | 0.25 deg / Hourly | $U_{10}, V_{10}$, Surface Pressure |
| Surface Waves & Stokes Drift | ECMWF ERA5 / WaveWatch III | 0.5 deg / Hourly | $H_s, T_p, \vec{U}_{Stokes}$ |
| Ocean Hydrodynamics | Copernicus Marine Service (CMEMS) | 0.083 deg / Multi-Depth Hourly | Horizontal Currents ($\vec{V}$), Salinity |
| Digital Elevation Model | GEBCO 2023 Grid | 15 arc-second | Bathymetric Depth ($Z_{bathymetry}$) |
| Tropical Cyclones | NOAA IBTrACS Database | Point / 3-Hourly | Storm Tracks, $V_{max}$, Central Pressure |
| Paleoclimate Adjustment | NOAA NCEI Coral Geochemical Proxies | Annually Resolved | Sea Surface Temperature Offset ($-2^\circ\text{C}$) |
| Archival Loss Ground Truth | Wrecksite.eu / NOAA Ocean Explorer | Text / Tabular | Reference Coordinates, Loss Accounts |

---

## Repository Structure

```text
stochastic-marine-drift/
├── data/
│   ├── raw/                  # Atmospheric, hydrodynamic, and bathymetric grids
│   ├── processed/            # Structured wreck inventories and training features
│   └── networks/             # Caribbean shipping network definitions (GeoJSON)
├── src/
│   ├── profiles/             # Vessel taxonomy and hydrodynamics
│   │   ├── vessel_class.py   # VesselProfile dataclass and leeway definitions
│   │   └── presets.py        # Calibrated vessel parameters (Galleon, Sloop, etc.)
│   ├── modules/              # Decoupled physics engines
│   │   ├── wind.py           # ERA5 atmospheric ingestion and Weibull turbulence
│   │   ├── current.py        # Copernicus multi-depth velocity and Ekman shear
│   │   ├── stokes.py         # Wave mass transport (Stokes drift)
│   │   ├── deep_water.py     # 3D descent/ascent physics and hydrodynamic glide
│   │   ├── extreme_event.py  # IBTrACS cyclonic storm field modeling
│   │   └── collision.py      # GEBCO bathymetric grounding filter
│   ├── agents/               # Archival intelligence and NLP
│   │   ├── parser.py         # Unstructured logbook parameter extractor
│   │   └── epistemic.py      # Source reliability tiering and prior covariance
│   ├── models/               # Core simulation and learning engines
│   │   ├── forward_drift.py  # Monte Carlo surface drift simulation
│   │   ├── backtrack.py      # Discrete Lagrangian reverse-time optimizer
│   │   ├── neural_ode.py     # Continuous Physics-Informed Neural ODE
│   │   ├── foundering.py     # Random Survival Forest dynamic failure model
│   │   ├── convoy_gnn.py     # Spatio-temporal graph network for convoy dispersion
│   │   └── debris_field.py   # Bivariate Gaussian artifact scatter generator
│   └── utils/                # Utility and evaluation routines
│       ├── geo.py            # Geodesic metrics and raster spatial sampling
│       ├── paleo.py          # Coral proxy temperature and density corrections
│       └── metrics.py        # Containment rate and Search Area Reduction (SAR)
├── notebooks/                # Model calibration, verification, and visual analytics
├── tests/                    # Unit and integration test suites
├── PLAN.md                   # Detailed implementation roadmap and specifications
├── summary.md                # Mathematical formulations and reference catalog
├── environment.yml           # Conda environment specification (recommended)
└── requirements.txt          # Pip package dependencies
```

---

## Installation & Environment Setup

### Option A: Conda (Recommended)

Conda (or Mamba) is strongly recommended for this framework because core dependencies (`OpenDrift`, `GDAL`, `GEOS`, `PROJ`, `netCDF4`, `Cartopy`, `rasterio`) require compiled C/Fortran libraries. Conda manages these binaries natively across Linux and macOS.

```bash
# Clone the repository
git clone https://github.com/your-org/stochastic-marine-drift.git
cd stochastic-marine-drift

# Create and activate the conda environment
conda env create -f environment.yml
conda activate smd-drift

# Verify test suite
pytest
```

### Option B: Python venv & Pip

If using standard virtual environments, ensure underlying system development headers (`libgdal-dev`, `libgeos-dev`, `libproj-dev`, `libnetcdf-dev`) are installed on your OS prior to running `pip install`:

```bash
# Clone the repository
git clone https://github.com/your-org/stochastic-marine-drift.git
cd stochastic-marine-drift

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify test suite
pytest
```

---

## Validation & Benchmarks

Model performance is benchmarked against a curated cohort of historical Caribbean losses with confirmed archaeological site coordinates:

* **Nuestra Senora de Atocha (1622)**: Florida Keys (Marquesas Keys); hurricane loss, known primary ballast mound and secondary scatter trail.
* **Santa Margarita (1622)**: Florida Keys; hurricane dispersion benchmark.
* **1715 Spanish Plate Fleet**: Florida Atlantic Coast; multi-vessel hurricane reef grounding benchmark.
* **HMS Looe (1744)**: Looe Key, Florida; barrier reef collision.
* **SS El Faro (2015)**: Bahamas / Atlantic approach; modern container vessel lost during Hurricane Joaquin (deep-water foundering and glide benchmark).

Quantitative verification relies on two formal search theory metrics:
1. **Containment Probability Rate ($P_{contain}$)**: Frequency with which known wreck coordinates fall within the 50%, 80%, and 95% simulated probability contours.
2. **Search Area Reduction (SAR) Ratio**: Ratio of the naive circular search area (derived from LKP uncertainty) to the area bounded by the model's 95% contour. Target: $\ge 85\text{–}95\%$ reduction.

---

## References & Documentation

* Complete theoretical equations, data access links, and paleoclimate correction workflows: [summary.md](summary.md).
* Implementation schedule, technical milestones, and testing guidelines: [PLAN.md](PLAN.md).
