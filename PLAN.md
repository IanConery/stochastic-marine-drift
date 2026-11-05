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
                                      |
                                      v
+-------------------------------------------------------------------------------+
| Phase 10: Archival Knowledge Extraction, Catalog Expansion & Roadstead GNN    |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
| Phase 11: Domain Model Fine-Tuning, Kev Evaluation & Hosting (Qwen/Kev Stack) |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
| Phase 12: Interactive Spatial Visualization & Cartographic Reporting Suite   |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
| Phase 13: Operational Mission Planning, Sensor Physics & Field Navigational   |
|           Export Suite (IAMSAR Tracklines, Sediment Burial & KML/GPX)         |
+-------------------------------------------------------------------------------+
```

---

### Phase 1: Environment, Dependencies & Vessel Taxonomy Architecture [Completed]
* **Goal**: Establish the core dependency tree, folder structure, and the object-oriented vessel classification system.
* **Tasks**:
  1. Generate `requirements.txt` and `environment.yml` with pinned versions (`opendrift`, `oceanparcels`, `torch`, `torchdyn`, `torch_geometric`, `xarray`, `netCDF4`, `rasterio`, `scikit-learn`, `lifelines`).
  2. Implement `src/profiles/vessel_class.py`: Define `VesselProfile` frozen dataclass with `mast_count`, `RigType`, and `MacroArchetype` enums, alongside physical bounds validation.
  3. Implement `src/profiles/presets.py`: Populate canonical Macro-Archetypes (`1_MAST_FORE_AFT`, `2_MAST_FORE_AFT`, `2_MAST_SQUARE`, `3_MAST_SQUARE`, `MODERN_UNMASTED`) and specific historical named vessels (`SPANISH_GALLEON_1620`, etc.).
  4. Implement `profile_from_masts_and_rig`: Dynamic factory mapping combinations of mast count and rig type to base archetypes with continuous volumetric scaling ($Draft \propto Tonnage^{1/3}$).
  5. Write unit tests in `tests/test_profiles.py` verifying bounds, archetype derivation, and tonnage scaling.
* **Deliverable**: Complete and verified (12 unit tests passing).

### Phase 2: Decoupled Environmental Physics Modules [Completed]
* **Goal**: Build independent numerical modules for atmospheric, current, wave, and storm forcing.
* **Tasks**:
  1. `src/modules/wind.py`: NetCDF ERA5 $U_{10}, V_{10}$ reader; stochastic Weibull noise perturbation generator; island coastal diurnal land-sea breeze offset calculation.
  2. `src/modules/current.py`: Copernicus multi-layer current interpolation; surface Ekman layer transport solver.
  3. `src/modules/stokes.py`: Empirical Stokes drift mass transport solver from wave spectra ($H_s, T_p, \theta_w$).
  4. `src/modules/extreme_event.py`: NOAA IBTrACS NetCDF/CSV parser; parametric Holland hurricane wind and pressure vortex field synthesizer.
  5. `src/modules/deep_water.py`: Terminal descent velocity ($w_{\text{term}}$) solver balancing buoyancy against drag; horizontal glide vector integration.
* **Deliverable**: Complete and verified (15 unit tests passing).

### Phase 3: Bathymetric Collision & Debris Field Dispersion [Completed]
* **Goal**: Implement physical boundary constraints (grounding) and post-settlement spatial modeling.
* **Tasks**:
  1. `src/modules/collision.py`: Bilinear interpolation of GEBCO 15-arc-second bathymetry grids; dynamic depth evaluation against draft, tidal stage, and wave troughs; grounding state triggers.
  2. `src/models/debris_field.py`: Bivariate Gaussian scatter generator calculating major/minor dispersal axes based on deep current shear and descent duration.
* **Deliverable**: Complete and verified (11 unit tests passing across collision and debris field).

### Phase 4: Archival NLP & Epistemic Uncertainty Prior Calibrator [Completed]
* **Goal**: Transform messy unstructured historical text into structured parameters and calibrated statistical priors.
* **Tasks**:
  1. `src/agents/parser.py`: LLM schema extractor prioritizing universal primary features: `mast_count` ($1, 2, 3+$) and `rig_type` (`square` vs `fore_and_aft`), bypassing brittle historical nomenclature to map directly to physical archetypes. Extracts continuous attributes (`cargo_mass`, `tonnage`, `buoyancy_factor`, `armament_tonnage`).
  2. `src/agents/epistemic.py`: Evaluator assessing source provenance (official captain's log vs. companion ship sighting vs. retrospective colonial inquiry), outputting calibrated spatial covariance $\mathbf{\Sigma}_0$ and temporal variance $\sigma_t^2$.
  3. Decoupled host URL endpoint via `.env` and Kev/Jev System-One evaluation integration.
* **Deliverable**: Complete and verified (9 unit tests passing).

### Phase 5: Forward Monte Carlo Drift Engine [Completed]
* **Goal**: Assemble surface physics, leeway dynamics, and the crew state machine into a high-performance ensemble simulator.
* **Tasks**:
  1. `src/models/forward_drift.py`: OpenDrift-based particle simulation runner; integration of decoupled modules (wind, current, Stokes, extreme storm).
  2. Implement crew behavioral state machine: transitions between `SAILING_DAMAGED`, `HEAVED_TO`, `JETTISONING_CARGO`, `DERELICT_DEAD_DRIFT`, `GROUNDED`, and `FOUNDERED`.
  3. Stochastic particle integration loop generating forward trajectory histories, 2D probability density rasters, and terminal spatial distributions.
* **Deliverable**: Complete and verified (5 unit tests passing).

### Phase 6: Continuous Inverse Backtracker (Neural ODE & Particle Filter) [Completed]
* **Goal**: Reverse time steps from seafloor wreck locations back to Surface Point of Sink with minimal numerical diffusion.
* **Tasks**:
  1. `src/models/backtrack.py`: Discrete Lagrangian reverse particle filter with backward ascent math and Bayesian speed gating ($V_{max} \le 6\text{ knots}$).
  2. `src/models/neural_ode.py`: Continuous PyTorch Neural ODE optimizing backward trajectory paths using adjoint sensitivity with Navier-Stokes physics-informed regularization.
  3. Bi-directional Bayesian closure intersection reducing naive search areas by $85\text{–}95\%$.
* **Deliverable**: Complete and verified (13 unit tests passing across backtrack and neural ODE).

### Phase 7: Dynamic Foundering Survival Classifier [Completed]
* **Goal**: Train a supervised survival model on backtracked historical wrecks to replace hardcoded sinking heuristics.
* **Tasks**:
  1. Generate synthetic training dataset: run backtracker on 50+ known Caribbean wrecks; intersect computed SPS coordinates with ERA5 wave/wind grids at loss timestamps.
  2. Implement **Hierarchical Bayesian Partial Pooling**: To eliminate $N=1$ overfitting on rare vessels, each vessel type $j$ belongs to a macro-archetype $k$ ($\theta_j \sim \mathcal{N}(\mu_k, \sigma_k^2)$). Rare classes shrink toward the macro-archetype mean rather than fitting noisy isolated parameters.
  3. `src/models/foundering.py`: Train Cox Proportional Hazards model on cumulative wave energy, steepness, and hull fatigue integrals.
  4. Integrate the survival inference model directly into `forward_drift.py` to trigger the foundering phase dynamically.
* **Deliverable**: Complete, trained, and serialized (`data/processed/foundering_model.pkl`, 6 unit tests passing).

### Phase 8: Spatio-Temporal Graph Neural Network (Convoy Breakdown Model) [Completed]
* **Goal**: Model fleet network cohesion and predict straggler separation events during extreme weather.
* **Tasks**:
  1. `data/networks/caribbean_convoy_routes.geojson`: Map historical departure ports (Cartagena, Portobelo, Veracruz, Havana), roadsteads, and choke points (Mona Passage, Florida Straits).
  2. `src/models/convoy_gnn.py`: PyTorch Geometric spatio-temporal graph convolutional network; dynamic edge weighting based on seasonal headwinds, currents, and hurricane radii; node/edge detachment prediction.
* **Deliverable**: Complete, trained, and serialized (`data/processed/convoy_gnn.pt`, 6 unit tests passing).

### Phase 9: End-to-End Validation & Historical Ground-Truth Benchmarking [Completed]
* **Goal**: Benchmark the complete SMD pipeline against documented historical Caribbean shipwrecks and verify formal search theory metrics.
* **Tasks**:
  1. Implement `src/models/validation.py` and benchmark against *Nuestra Señora de Atocha* (1622) and *Santa Margarita* (1622) in the Marquesas/Quicksands reef complexes.
  2. Benchmark against *HMS Looe* (1744) barrier reef collision incorporating localized shoaling bathymetry.
  3. Benchmark against *SS El Faro* (2015) deep abyssal foundering ($4,600\text{ m}$) and hydrodynamic glide.
  4. Implement `src/models/post_depositional.py` modeling multi-century storm chronology from loss date to present, Airy wave bottom orbital velocity ($u_b$), Shields sediment saltation ($u_b \ge 0.8\text{ m/s}$), and dual-layer GeoJSON plotting (anchored primary ballast mound vs. migrated coin/cargo dispersion polygon).
  5. Compute quantitative evaluation metrics:
     - Containment Probability Rate ($P_{\text{contain}}$): **100.0%** at 95% contour, **100.0%** at 80% contour, **75.0%** at 50% contour across the full cohort.
     - Search Area Reduction (SAR) ratio: **96.5% mean reduction** (range: $95.0\%\text{--}97.8\%$) against naive circular LKP baselines.
     - Mean centroid localization error: **2.19 km** (1.18 nmi).
* **Deliverable**: Complete and verified (3 unit/integration tests passing in `tests/test_validation.py`, 4 unit tests passing in `tests/test_post_depositional.py`, serialized benchmark suite `data/processed/historical_benchmark_suite.geojson` and report `data/processed/historical_benchmark_report.json`). Total project test suite: **87 tests passing**.

### Phase 10: Archival Knowledge Extraction, Catalog Expansion & Roadstead GNN [Active Next]
* **Goal**: Ingest Wikipedia archive dumps (`/mnt/storage/wiki/wiki_dump.json.bz2`), extract a structured catalog of 100+ historical shipwrecks across the Caribbean and Western Atlantic, incorporate submerged colonial ports and hurricane roadsteads into the route graph, implement a stratified 75/25 train/test holdout split, and retrain the survival classifier and convoy GNN.
* **Tasks**:
  1. `src/modules/wiki_extractor.py`: Streaming compressed JSON reader scanning `/mnt/storage/wiki/wiki_dump.json.bz2` with bounding box filtering ($9^\circ\text{N}\text{ to }35^\circ\text{N}, -98^\circ\text{W}\text{ to }-55^\circ\text{W}$) and maritime category regex.
  2. In-Repo Structured Corpus Persistence: Persist extracted records to `data/raw/wikipedia_maritime_extracts.jsonl` (~2--5 MB) and `data/processed/expanded_shipwreck_catalog.geojson` (100+ vessel entities mapped to the 5 macro-archetypes).
  3. Historical Ports & Roadsteads: Identify and extract submerged ports (Port Royal, Nombre de Dios) and careening cays/roadsteads (Dry Tortugas, Isle of Pines, Roatan, Virgin Gorda); expand `data/networks/caribbean_convoy_routes.geojson`.
  4. Rigorous Out-of-Sample Partitioning: Establish a stratified 75/25 train/test split across Macro-Archetypes and temporal epochs (16th--19th century).
  5. Supervised Model Retraining: Run Lagrangian backtracker on training wrecks, intersect with ERA5/paleoclimate grids, and update `foundering_model.pkl` and `convoy_gnn.pt`. Evaluate Harrell's $C$-index and Brier score on the unseen 25% test cohort.
  6. Verification: `tests/test_wiki_corpus.py` testing schema integrity, coordinate bounds, and archetype distributions.
* **Deliverable**: Extracted historical corpus, expanded route graph, retrained model weights, test suite.

### Phase 11: Domain Model Fine-Tuning, Kev Evaluation & Hosting (Qwen 3.8 / Kev Stack) [Planned]
* **Goal**: Fine-tune and evaluate the consolidated archival language model (`Qwen 3.8`) and quality judge (`kev-4b`) using the real-world historical corpus extracted in Phase 10, establishing an offline production inference service.
* **Tasks**:
  1. `src/agents/train_domain_model.py`: Ingest `data/raw/wikipedia_maritime_extracts.jsonl` paired training corpus; fine-tune `Qwen 3.8` via DoRA (Weight-Decomposed Low-Rank Adaptation) for joint historical context reasoning and deterministic Pydantic schema generation (`mast_count`, `rig_type`, `tonnage`, `buoyancy_factor`, `draft_meters`).
  2. `src/agents/judge.py`: Evaluate `jaredpalmer/kev-4b` on the 25% holdout test set to benchmark non-autoregressive hallucination detection (`Noul`) and credibility tier classification (`Choice`) across historical maritime texts.
  3. Multimodal Reasoning Verification: Verify `Qwen 3.8` multimodal and large-context (262K) reasoning capabilities for resolving ambiguous colonial sailing narratives and historical chart scans.
  4. Production Serving & Deployment:
     - Configure local high-throughput serving via `vLLM` or `Ollama` (`http://localhost:11434/v1` or `http://localhost:8000/v1`) for fully offline, zero-cost execution on a single model weight set.
     - Provide Hugging Face Model Hub push and Serverless Inference Endpoint export configuration.
  5. Verification: `tests/test_agents.py` and `tests/test_domain_training.py` verifying schema parsing accuracy ($>95\%$ JSON validity) and Kev judge decision latency ($<300\text{ ms}$).
* **Deliverable**: Fine-tuned Qwen 3.8 DoRA adapter weights, benchmarked Kev judge evaluation report, local vLLM/Ollama configuration, and passing test suite.

### Phase 12: Interactive Spatial Visualization & Cartographic Reporting Suite [Planned]
* **Goal**: Deliver a comprehensive spatial visualization and cartographic reporting suite for interactive mission planning, archaeological scatter analysis, and search box comparison.
* **Tasks**:
  1. `src/visualization/map_viewer.py`: Interactive dual-layer Leaflet/Folium cartographic dashboard visualizable in any standard web browser. Toggles GEBCO bathymetry, convoy corridors, naive circular uncertainty radius, model 95% probability contour, initial touchdown ballast mound, present-day secondary cargo scatter, and storm displacement vectors.
  2. `src/visualization/descent_3d.py`: 3D water column hydrodynamic descent trajectory visualizer plotting 3D glide from Surface Point of Sink (SPS) to seafloor bathymetric collision.
  3. `src/visualization/diagnostics.py`: Dynamic hull survival and fatigue diagnostic curves showing cumulative wave stress integrals, steepness, and discrete crew state transitions over drift time.
  4. Automated Artifact Generation: Export standalone HTML maps to `reports/visualizations/interactive_wreck_map.html` and high-resolution vector figures (SVG/PDF).
  5. Verification: `tests/test_visualization.py` validating HTML generation, GeoJSON layer loading, and plot rendering without GUI head dependencies.
* **Deliverable**: Map viewing engine, 3D trajectory visualizer, diagnostic plot modules, and automated HTML reporting.

### Phase 13: Operational Mission Planning, Sensor Physics & Field Navigational Export Suite [Planned]
* **Goal**: Bridge probabilistic predictive modeling and real-world marine salvage operations by implementing IAMSAR search planning, multi-century sediment burial physics, sensor attenuation modeling, and field chartplotter export suites.
* **Tasks**:
  1. `src/models/post_depositional.py` (Sediment Burial Integration):
     - Model multi-century sediment accumulation and post-storm tempestite settling ($\Delta z_{\text{burial}}$).
     - Compute net vertical overburden thickness ($0.5\text{--}3.0\text{ m}$) resulting from recurring hurricane fluidization and Stokes settling.
     - Quantify physical overburden metrics ($\Delta z_{\text{burial}}$ in meters) alongside substrate scour dimensions.
  2. `src/models/search_planning.py` (IAMSAR Optimal Effort & Trackline Allocator):
     - Compute formal Probability of Detection ($POD = 1 - e^{-W \cdot L / A}$) and total Probability of Success ($POS = POC \times POD$).
     - Generate mathematically optimal parallel creeping-line and expanding-square search tracklines budgeted against operational vessel speed ($4\text{ knots}$), sensor swath ($W$), and allocated sea days.
  3. `src/modules/sediment.py` (Geophysical Signal Attenuation Physics):
     - Calculate magnetic dipole field attenuation ($B \propto \mu_0 M / 4\pi r^3$) as a function of sensor altitude plus computed sediment overburden $r = z_{\text{sensor}} + \Delta z_{\text{burial}}$.
     - Compute two-way acoustic attenuation through carbonate sediment layers as a physical reference for geophysical survey teams.
  4. `src/utils/exporters.py` (Marine Chartplotter Export Suite):
     - Export predicted containment polygons, ballast mound contours, secondary scatter corridors, and survey tracklines to **KML/KMZ** (Google Earth Pro / GIS) and **GPX** (Garmin, OpenCPN, Furuno, Raymarine).
  5. `src/cli.py` (Unified Terminal Interface):
     - Implement unified CLI entrypoint for executing forward drift runs, backtrackers, benchmark evaluations, catalog extraction, and survey trackline generation.
  6. Verification: `tests/test_search_planning.py` asserting $POS$ calculations, trackline spacing geometry, sediment burial bounds, and KML/GPX schema validity.
* **Deliverable**: Complete mission planning engine, sediment burial solver, KML/GPX exporter, CLI runner, and verified test suite.

---

## 4. Verification & Testing Strategy

| Test Suite | File Path | Focus & Assertions |
| :--- | :--- | :--- |
| **Vessel Profiles** | `tests/test_profiles.py` | Validates parameter ranges, frozen immutability, leeway coefficient bounds, and preset definitions. |
| **Stokes Drift Math** | `tests/test_stokes.py` | Tests zero wave height condition ($\|\vec{U}_{\text{Stokes}}\| = 0$), dimensional correctness, and direction alignment. |
| **Grounding Filter** | `tests/test_collision.py` | Asserts particles halt when $Z_{\text{bathymetry}} \le Z_{\text{draft}}$, verifying transition to `GROUNDED_ON_REEF`. |
| **Inverse Adjoint Physics** | `tests/test_neural_ode.py` | Asserts Neural ODE reverse trajectories do not exceed maximum velocity bounds ($V \le V_{max}$). |
| **Survival Predictor** | `tests/test_foundering.py` | Confirms survival probability decreases monotonically with increasing cumulative wave stress. |
| **Convoy Network GNN** | `tests/test_convoy_gnn.py` | Validates graph convolution dimensions, edge attention weights, and straggler scatter vectors. |
| **Post-Depositional Dispersal** | `tests/test_post_depositional.py` | Asserts Airy wave bottom orbital velocity, Shields sediment mobilization, and dual-layer GeoJSON export. |
| **Historical Benchmarks** | `tests/test_validation.py` | Confirms 100% 95% containment and >= 85% SAR across the 4-case curated cohort. |
| **Archival Wiki Corpus** | `tests/test_wiki_corpus.py` | Verifies JSONL/GeoJSON schemas, coordinate ranges, and stratified 75/25 train/test partitioning. |
| **Domain Model Training** | `tests/test_domain_training.py` | Validates DoRA fine-tuning format, schema parsing accuracy, and Kev judge latency. |
| **Interactive Visualization** | `tests/test_visualization.py` | Asserts headless compilation of Leaflet maps and 3D descent trajectory projections. |
| **Mission Planning & Sensors** | `tests/test_search_planning.py` | Asserts IAMSAR POS optimization, survey trackline geometry, sediment burial bounds, and KML/GPX exports. |

---

## 5. Directory Scaffolding & Module Map

### Directory Structure:
```bash
mkdir -p data/raw data/processed data/networks
mkdir -p src/profiles src/modules src/agents src/models src/utils src/visualization
mkdir -p notebooks tests reports/visualizations
```

### Unified CLI Entry Point:
* `src/cli.py`: Unified terminal command-line interface for drift execution, benchmarking, catalog ingestion, and survey mission planning.

### Initial Core Implementation Files:
1. `requirements.txt`: Master dependency manifest.
2. `src/profiles/__init__.py`: Package export.
3. `src/profiles/vessel_class.py`: Core `VesselProfile` dataclass.
4. `src/profiles/presets.py`: Standard historical and modern vessel parameter sets.
5. `tests/test_profiles.py`: Validation tests for vessel profiles.
