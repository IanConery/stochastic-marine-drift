# Stochastic Marine Drift (SMD): Master Engineering Implementation Plan

## 1. Project Overview & Objectives

Stochastic Marine Drift (SMD) is a scientific computing and machine learning framework designed to locate historical shipwrecks and submerged archaeological sites in the Caribbean Basin. 

By coupling forward stochastic Monte Carlo simulation with continuous inverse-time Lagrangian trajectory reconstruction (Physics-Informed Neural ODEs), the framework transforms qualitative archival records and open-source paleoceanographic reanalyses into calibrated 2D seafloor probability density maps and asymmetrical 3D debris field scatter polygons.

### Key Performance Indicators
* **Search Area Reduction (SAR) Ratio**: $\ge 85\text{–}95\%$ reduction compared to traditional circular dead-reckoning search radii.
* **Containment Probability Rate ($P_{\text{contain}}$)**: $\ge 90\%$ containment of ground-truth wreck sites within the 95% simulated probability density contour across the historical validation cohort.
* **Physical Generalization**: Zero reliance on arbitrary fixed foundering heuristics (e.g., "sinks after $N$ hours"); all phase transitions must be driven by dynamically learned environmental stress thresholds and bathymetric elevation bounds.

---

## 2. Mathematical Formulations & Component Specifications

### 2.1 Vessel Profile, 2D Mast/Rig Taxonomy & Leeway Equations
To prevent combinatorial explosion and severe sample sparsity ($N=1$ data scarcity for rare historical vessel designations), vessels are structured via a **2D Matrix of Mast Count ($0, 1, 2, 3+$) and Primary Rig Type (Square vs. Fore-and-Aft)** collapsing dozens of archival names into 5 physical Macro-Archetypes:
* `ONE_MAST_FORE_AFT`: Sloops, cutters (shallow draft $1.5\text{–}2.8\text{m}$, high crosswind leeway $CWL \approx 26^\circ$, frequent jibes).
* `TWO_MAST_FORE_AFT`: Schooners, ketches (draft $2.5\text{–}3.8\text{m}$, bimodal zigzag drift, moderate windage).
* `TWO_MAST_SQUARE`: Brigs, snows, brigantines (draft $3.0\text{–}4.2\text{m}$, downwind-dominated leeway $DWL \approx 3\%$).
* `THREE_MAST_SQUARE`: Galleons, frigates, naos, fluyts, ships of the line (deep draft $4.5\text{–}7.0\text{m}$, heavy downwind leeway $DWL \approx 4\text{–}5\%$, massive bare-pole windage).
* `MODERN_UNMASTED`: Powered industrial steel vessels (deep draft $8.0\text{–}14.0\text{m}$, current-dominated drift, hydrodynamic glide).

#### Continuous Volumetric Scaling
When displacement tonnage ($M_{\text{tons}}$) or armament is extracted from logbooks, baseline archetype parameters scale continuously:
$$\text{Scale Factor } s = \max\left(0.6, \, \min\left(1.8, \, \left(\frac{M_{\text{tons}}}{M_{\text{ref}}}\right)^{1/3}\right)\right)$$
$$\text{Draft: } T = T_{\text{base}} \times s, \quad \text{Freeboard: } F = F_{\text{base}} \times s^{0.8}, \quad a_{dw} = \frac{a_{\text{base}}}{s^{0.2}}$$

#### Leeway Dynamics Formulation
Leeway drift $\vec{L}$ is modeled using downwind ($DWL$) and crosswind ($CWL$) components relative to the 10-meter wind vector $\vec{U}_{10}$:
$$DWL = a_{dw} \|\vec{U}_{10}\| + b_{dw} + \epsilon_{dw}, \quad \epsilon_{dw} \sim \mathcal{N}(0, \sigma_{dw}^2)$$
$$CWL = a_{cw} \|\vec{U}_{10}\| + b_{cw} + \epsilon_{cw}, \quad \epsilon_{cw} \sim \mathcal{N}(0, \sigma_{cw}^2)$$
The instantaneous surface vessel velocity $\vec{V}_{\text{surface}}$ combines water current, leeway drift, and wave-induced Stokes drift:
$$\vec{V}_{\text{surface}}(t) = \vec{V}_{\text{current}}(t, \mathbf{x}) + \vec{L}(t, \vec{U}_{10}) + \vec{U}_{\text{Stokes}}(t, \mathbf{x})$$

Jibe switching (transitioning crosswind divergence sign $\pm CWL$) is evaluated at each discrete time step $\Delta t$ with Bernoulli probability:
$$P(\text{jibe}) = 1 - e^{-\lambda_{\text{jibe}} \Delta t}$$

### 2.2 Stokes Drift Formulation
Wave mass transport $\vec{U}_{\text{Stokes}}$ is derived from significant wave height $H_s$, peak wave period $T_p$, and dominant wave direction $\theta_w$:
$$\|\vec{U}_{\text{Stokes}}\| \approx \frac{\pi^2 H_s^2}{2 g T_p^3}$$
$$\vec{U}_{\text{Stokes}} = \|\vec{U}_{\text{Stokes}}\| \begin{bmatrix} \sin\theta_w \\ \cos\theta_w \end{bmatrix}$$

### 2.3 Bathymetric Collision & Grounding Filter
At each integration step $t$, the particle position $(x_t, y_t)$ is evaluated against the GEBCO digital elevation model $Z_{\text{bathymetry}}$:
$$\text{Collision Condition: } Z_{\text{bathymetry}}(x_t, y_t) \le Z_{\text{draft}} + \eta_{\text{tide}}(t) - \frac{H_s(t)}{2}$$
When satisfied:
1. Instantaneous velocity $\vec{V}(t) \to \mathbf{0}$.
2. Particle state flags transition to `GROUNDED_ON_REEF`.
3. Spatial coordinates are registered as a high-confidence primary impact site.

### 2.4 Dynamic Foundering Survival Function
Hull failure transition from surface leeway to 3D descent is evaluated via a Random Survival Forest learning hazard rate $\lambda(t | \mathbf{X}_t)$:
$$S(t | \mathbf{X}_t) = \exp\left( -\int_{0}^{t} \lambda_0(\tau) \exp(\beta^T \mathbf{X}_\tau) d\tau \right)$$
Where the dynamic feature vector $\mathbf{X}_t$ includes:
$$\mathbf{X}_t = \left[ H_s(t), \, \frac{H_s(t)}{g T_p(t)^2} \text{ (steepness)}, \, \int_0^t \|\vec{U}_{10}(\tau)\|^2 d\tau, \, \text{buoyancy\_factor}, \, \text{cargo\_mass\_ratio} \right]^T$$

### 2.5 Continuous Inverse Trajectory via Physics-Informed Neural ODE (PINN-ODE)
Continuous reverse-time trajectory reconstruction from seafloor coordinates $\mathbf{x}_{\text{seafloor}}$ to Surface Point of Sink (SPS) is solved as:
$$\frac{d\mathbf{x}}{dt} = f_{\theta}(\mathbf{x}(t), t, \mathbf{u}(t)), \quad t \in [T_{\text{foundered}}, 0]$$
Trained with adjoint sensitivity methods minimizing:
$$\mathcal{L}(\theta) = \mathcal{L}_{\text{obs}} + \lambda_1 \mathcal{L}_{\text{Navier-Stokes}} + \lambda_2 \max\left(0, \|\frac{d\mathbf{x}}{dt}\| - V_{\text{max\_vessel}}\right)^2$$
Ensuring the inverse trajectory strictly adheres to fluid continuity and cannot exceed physical hull velocity limits ($V_{\text{max\_vessel}} \le 6\text{ knots}$).

### 2.6 Epistemic Uncertainty & Bayesian Prior Seeding
Historical text evaluated by the LLM agent assigns a credibility tier $k \in \{1, 2, 3\}$. The initial spatial position prior $\mathbf{x}_0$ is sampled from a bivariate Gaussian:
$$\mathbf{x}_0 \sim \mathcal{N}\left(\mathbf{\mu}_{\text{LKP}}, \, \mathbf{\Sigma}_0\right), \quad \mathbf{\Sigma}_0 = \begin{bmatrix} \sigma_{\text{pos}, k}^2 & 0 \\ 0 & \sigma_{\text{pos}, k}^2 \end{bmatrix}$$
$$t_0 \sim \mathcal{N}(T_{\text{recorded}}, \, \sigma_{t, k}^2)$$

### 2.7 Asymmetric Debris Scatter Polygon
The seafloor deposition footprint is modeled as two coupled components:
1. **Primary Ballast Mound**: Bivariate normal distribution centered at initial bottom touchdown $\mathbf{x}_{\text{touchdown}}$ with tight covariance $\mathbf{\Sigma}_{\text{ballast}}$ ($50\text{m} \times 20\text{m}$).
2. **Peripheral Scatter Trail**: Elongated Gaussian oriented along the mean deep current vector $\vec{V}_{\text{deep}}$:
$$f_{\text{scatter}}(\mathbf{x}) = \alpha \mathcal{N}(\mathbf{x}_{\text{touchdown}}, \mathbf{\Sigma}_{\text{ballast}}) + (1-\alpha) \mathcal{N}(\mathbf{x}_{\text{touchdown}} + \Delta \mathbf{x}_{\text{drift}}, \mathbf{\Sigma}_{\text{debris}})$$

---

## 3. Phased Implementation Roadmap

```
+-------------------------------------------------------------------------------+
| Phase 1: Environment, Dependencies & Vessel Taxonomy Architecture            |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
| Phase 2: Decoupled Physics Engines (Wind, Current, Stokes Drift, Hurricanes)  |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
| Phase 3: Bathymetric Collision (Reef Filter) & Debris Field Dispersion        |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
| Phase 4: Archival NLP Agent & Epistemic Uncertainty Prior Calibrator         |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
| Phase 5: Forward Monte Carlo Drift Engine & Crew State Machine                |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
| Phase 6: Continuous Inverse Backtracker (Neural ODE & Discrete Lagrangian)   |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
| Phase 7: Dynamic Foundering Survival Classifier (RSF / Cox Proportional)     |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
| Phase 8: Spatio-Temporal Graph Neural Network (Convoy Breakdown Model)        |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
| Phase 9: End-to-End Validation & Historical Ground-Truth Benchmarking         |
+-------------------------------------------------------------------------------+
```

---

### Phase 1: Environment, Dependencies & Vessel Taxonomy Architecture
* **Goal**: Establish the core dependency tree, folder structure, and the object-oriented vessel classification system.
* **Tasks**:
  1. Generate `requirements.txt` and `environment.yml` with pinned versions (`opendrift`, `oceanparcels`, `torch`, `torchdyn`, `torch_geometric`, `xarray`, `netCDF4`, `rasterio`, `scikit-learn`, `lifelines`).
  2. Implement `src/profiles/vessel_class.py`: Define `VesselProfile` frozen dataclass with `mast_count`, `RigType`, and `MacroArchetype` enums, alongside physical bounds validation.
  3. Implement `src/profiles/presets.py`: Populate canonical Macro-Archetypes (`1_MAST_FORE_AFT`, `2_MAST_FORE_AFT`, `2_MAST_SQUARE`, `3_MAST_SQUARE`, `MODERN_UNMASTED`) and specific historical named vessels (`SPANISH_GALLEON_1620`, etc.).
  4. Implement `profile_from_masts_and_rig`: Dynamic factory mapping combinations of mast count and rig type to base archetypes with continuous volumetric scaling ($Draft \propto Tonnage^{1/3}$).
  5. Write unit tests in `tests/test_profiles.py` verifying bounds, archetype derivation, and tonnage scaling.
* **Deliverable**: Tested, importable `src/profiles` package with all unit tests passing.

### Phase 2: Decoupled Environmental Physics Modules
* **Goal**: Build independent numerical modules for atmospheric, current, wave, and storm forcing.
* **Tasks**:
  1. `src/modules/wind.py`: NetCDF ERA5 $U_{10}, V_{10}$ reader; stochastic Weibull noise perturbation generator; island coastal diurnal land-sea breeze offset calculation.
  2. `src/modules/current.py`: Copernicus multi-layer current interpolation; surface Ekman layer transport solver.
  3. `src/modules/stokes.py`: Empirical Stokes drift mass transport solver from wave spectra ($H_s, T_p, \theta_w$).
  4. `src/modules/extreme_event.py`: NOAA IBTrACS NetCDF/CSV parser; parametric Holland hurricane wind and pressure vortex field synthesizer.
  5. `src/modules/deep_water.py`: Terminal descent velocity ($w_{\text{term}}$) solver balancing buoyancy against drag; horizontal glide vector integration.
* **Deliverable**: Complete `src/modules/` with unit tests verifying energy conservation and velocity limits.

### Phase 3: Bathymetric Collision & Debris Field Dispersion
* **Goal**: Implement physical boundary constraints (grounding) and post-settlement spatial modeling.
* **Tasks**:
  1. `src/modules/collision.py`: Bilinear interpolation of GEBCO 15-arc-second bathymetry grids; dynamic depth evaluation against draft, tidal stage, and wave troughs; grounding state triggers.
  2. `src/models/debris_field.py`: Bivariate Gaussian scatter generator calculating major/minor dispersal axes based on deep current shear and descent duration.
* **Deliverable**: Grounding filter and debris dispersal polygon generators with GeoJSON export capabilities.

### Phase 4: Archival NLP & Epistemic Uncertainty Prior Calibrator
* **Goal**: Transform messy unstructured historical text into structured parameters and calibrated statistical priors.
* **Tasks**:
  1. `src/agents/parser.py`: LLM schema extractor prioritizing universal primary features: `mast_count` ($1, 2, 3+$) and `rig_type` (`square` vs `fore_and_aft`), bypassing brittle historical nomenclature to map directly to physical archetypes. Extracts continuous attributes (`cargo_mass`, `tonnage`, `buoyancy_factor`, `armament_tonnage`).
  2. `src/agents/epistemic.py`: Evaluator assessing source provenance (official captain's log vs. companion ship sighting vs. retrospective colonial inquiry), outputting calibrated spatial covariance $\mathbf{\Sigma}_0$ and temporal variance $\sigma_t^2$.
* **Deliverable**: Robust text-to-prior pipeline returning Pydantic-validated feature records.

### Phase 5: Forward Monte Carlo Drift Engine
* **Goal**: Assemble surface physics, leeway dynamics, and the crew state machine into a high-performance ensemble simulator.
* **Tasks**:
  1. `src/models/forward_drift.py`: OpenDrift-based particle simulation runner; integration of decoupled modules (wind, current, Stokes, extreme storm).
  2. Implement crew behavioral state machine: transitions between `SAILING_DAMAGED`, `HEAVED_TO`, `JETTISONING_CARGO`, `DERELICT_DEAD_DRIFT`, `GROUNDED`, and `FOUNDERED`.
  3. Stochastic particle integration loop generating forward trajectory histories and terminal spatial distributions.
* **Deliverable**: Functional forward Monte Carlo drift script generating probability density rasters.

### Phase 6: Continuous Inverse Backtracker (Neural ODE & Particle Filter)
* **Goal**: Reverse time steps from seafloor wreck locations back to Surface Point of Sink with minimal numerical diffusion.
* **Tasks**:
  1. `src/models/backtrack.py`: Discrete Lagrangian reverse particle filter with backward ascent math and Bayesian speed gating ($V_{max} \le 6\text{ knots}$).
  2. `src/models/neural_ode.py`: Continuous PyTorch/torchdyn Neural ODE optimizing backward trajectory paths using adjoint sensitivity with Navier-Stokes physics-informed regularization.
* **Deliverable**: Working bi-directional backtracker producing tight, non-diffuse reverse search corridors.

### Phase 7: Dynamic Foundering Survival Classifier
* **Goal**: Train a supervised survival model on backtracked historical wrecks to replace hardcoded sinking heuristics.
* **Tasks**:
  1. Generate synthetic training dataset: run backtracker on 50+ known Caribbean wrecks; intersect computed SPS coordinates with ERA5 wave/wind grids at loss timestamps.
  2. Implement **Hierarchical Bayesian Partial Pooling**: To eliminate $N=1$ overfitting on rare vessels, each vessel type $j$ belongs to a macro-archetype $k$ ($\theta_j \sim \mathcal{N}(\mu_k, \sigma_k^2)$). Rare classes shrink toward the macro-archetype mean rather than fitting noisy isolated parameters.
  3. `src/models/foundering.py`: Train Random Survival Forest and Cox Proportional Hazards models on cumulative wave energy, steepness, and hull fatigue integrals.
  4. Integrate the survival inference model directly into `forward_drift.py` to trigger the foundering phase dynamically.
* **Deliverable**: Trained survival model serialization (`foundering_rsf.pkl`) and dynamic foundering evaluator.

### Phase 8: Spatio-Temporal Graph Neural Network (Convoy Breakdown Model)
* **Goal**: Model fleet network cohesion and predict straggler separation events during extreme weather.
* **Tasks**:
  1. `data/networks/caribbean_convoy_routes.geojson`: Map historical departure ports (Cartagena, Portobelo, Veracruz, Havana), roadsteads, and choke points (Mona Passage, Florida Straits).
  2. `src/models/convoy_gnn.py`: PyTorch Geometric spatio-temporal graph convolutional network; dynamic edge weighting based on seasonal headwinds, currents, and hurricane radii; node/edge detachment prediction.
* **Deliverable**: Trained GNN predicting convoy vessel scatter probabilities under historical storm conditions.

### Phase 9: End-to-End Validation & Historical Ground-Truth Benchmarking
* **Goal**: Benchmark the complete SMD pipeline against documented historical Caribbean shipwrecks.
* **Tasks**:
  1. Benchmark against *Nuestra Senora de Atocha* (1622) and *Santa Margarita* (1622) in the Florida Keys.
  2. Benchmark against the *1715 Spanish Plate Fleet* catastrophic reef grounding event.
  3. Benchmark against *HMS Looe* (1744) barrier reef loss.
  4. Benchmark against *SS El Faro* (2015) deep-water foundering and hydrodynamic glide.
  5. Compute quantitative evaluation metrics:
     - Containment Probability Rate ($P_{\text{contain}}$ at 50%, 80%, 95%).
     - Search Area Reduction (SAR) ratio against naive circular LKP baselines.
     - Centroid distance error to primary ballast mound.
* **Deliverable**: Comprehensive validation notebook (`notebooks/05_validation_benchmark_atocha.ipynb`) and benchmark metrics report.

---

## 4. Verification & Testing Strategy

| Test Suite | File Path | Focus & Assertions |
| :--- | :--- | :--- |
| **Vessel Profiles** | `tests/test_profiles.py` | Validates parameter ranges, frozen immutability, leeway coefficient bounds, and preset definitions. |
| **Stokes Drift Math** | `tests/test_stokes.py` | Tests zero wave height condition ($\|\vec{U}_{\text{Stokes}}\| = 0$), dimensional correctness, and direction alignment. |
| **Grounding Filter** | `tests/test_collision.py` | Asserts particles halt when $Z_{\text{bathymetry}} \le Z_{\text{draft}}$, verifying transition to `GROUNDED_ON_REEF`. |
| **Inverse Adjoint Physics** | `tests/test_neural_ode.py` | Asserts Neural ODE reverse trajectories do not exceed maximum velocity bounds ($V \le V_{max}$). |
| **Survival Predictor** | `tests/test_foundering.py` | Confirms survival probability decreases monotonically with increasing cumulative wave stress. |
| **End-to-End Pipeline** | `tests/test_pipeline.py` | Runs a 10-particle test simulation from LKP to seafloor deposition, verifying schema consistency. |

---

## 5. Directory Scaffolding & Initial Code Generation

### Planned Directory Creation:
```bash
mkdir -p data/raw data/processed data/networks
mkdir -p src/profiles src/modules src/agents src/models src/utils
mkdir -p notebooks tests
```

### Initial Core Implementation Files:
1. `requirements.txt`: Master dependency manifest.
2. `src/profiles/__init__.py`: Package export.
3. `src/profiles/vessel_class.py`: Core `VesselProfile` dataclass.
4. `src/profiles/presets.py`: Standard historical and modern vessel parameter sets.
5. `tests/test_profiles.py`: Validation tests for vessel profiles.
