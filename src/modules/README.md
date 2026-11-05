# Decoupled Environmental Physics Modules (`src/modules/`)

This package implements independent numerical solvers for atmospheric, hydrodynamic, wave, storm, bathymetric, and deep-water forcing mechanisms. Environmental forces operate on differing spatial and temporal scales; decoupling them avoids monolithic equation errors and permits customized stochastic variance per layer.

---

## 1. Physics Modules Summary

### A. Stokes Wave Drift (`stokes.py`)
Wave mass transport pushes vessels independently of ocean currents. In severe sea states, Stokes drift accounts for $30\%\text{–}50\%$ of surface displacement:
$$U_s(0) = \frac{\pi^3 H_s^2}{2 g T_p^3}$$
Vertical depth attenuation:
$$U_s(z) = U_s(0) \exp(2 k z), \quad k = \frac{4\pi^2}{g T_p^2}, \quad z \le 0$$

### B. Atmospheric Turbulence & Coastal Land-Sea Breeze (`wind.py`)
* **Weibull Stochastic Gust Turbulence**: Samples normalized boundary layer wind fluctuations preserving $\mathbb{E}[\text{multiplier}] = 1.0$:
  $$\xi \sim \text{Weibull}(k_{\text{shape}}), \quad \text{normalized by } \Gamma(1 + 1/k)$$
* **Island Coastal Diurnal Breeze**: Simulates daytime onshore heating and nighttime offshore catabatic drainage winds near high-relief Caribbean islands (Cuba, Hispaniola, Jamaica, Puerto Rico):
  $$\Delta \vec{U}_{\text{breeze}}(d, t) = V_{\text{max}} \cdot \sin\left(\frac{2\pi (t_{\text{local}} - 9)}{24}\right) \exp\left(-\frac{d}{d_0}\right) \hat{n}_{\text{coast}}$$

### C. Ocean Currents & Ekman Shear (`current.py`)
* **Ekman Boundary Layer**: Surface wind stress $\tau = \rho_a C_d W_{10}^2$ deflects surface drift $+45^\circ$ to the right of the wind in the Northern Hemisphere (Coriolis parameter $f = 2\Omega\sin\phi > 0$):
  $$V_0 = \frac{\tau}{\rho_w \sqrt{|f| A_z}}$$
* **Depth Interpolation**: Smooth cosine transition between Copernicus surface currents and deep reference layers.

### D. Extreme Events: Holland Cyclone Vortex (`extreme_event.py`)
Synthesizes cyclonic wind and barometric pressure fields from NOAA IBTrACS tropical cyclone tracks (Holland, 1980):
$$P(r) = P_c + (P_n - P_c) \exp\left(-\left(\frac{R_{max}}{r}\right)^B\right)$$
$$V_{\text{grad}}(r) = \sqrt{\frac{B}{\rho_a} \left(\frac{R_{max}}{r}\right)^B \Delta P \exp\left(-\left(\frac{R_{max}}{r}\right)^B\right) + \left(\frac{r f}{2}\right)^2} - \frac{r f}{2}$$
Incorporates boundary layer cross-isobar inflow angles ($15^\circ\text{–}25^\circ$) and forward storm translation asymmetry (dangerous right-front quadrant).

### E. 3D Water Column Descent & Ascent (`deep_water.py`)
* **Terminal Descent Velocity**: Balances negative submerged weight against quadratic hydrodynamic drag:
  $$w_{\text{term}} = \sqrt{\frac{2 (\rho_{\text{hull}} - \rho_{\text{water}}) V_{\text{disp}} g}{\rho_{\text{water}} C_d A_{\text{proj}}}}$$
* **Bi-Directional Trajectories**: 
  - Forward 3D descent integrates multi-depth horizontal current deflections and hydrodynamic glide ($L/D$).
  - Inverse 3D ascent inverts velocity signs ($-\vec{V}_{\text{deep}}$) from seafloor wreck coordinates up to the exact Surface Point of Sink (SPS).

### F. Bathymetric Collision & Reef Grounding (`collision.py`)
Evaluates instantaneous water depth under the keel against the GEBCO digital elevation model:
$$D_{\text{eff}}(t, \mathbf{x}) = Z_{\text{water\_depth}}(\mathbf{x}) + \eta_{\text{tide}}(t) - \frac{H_s(t)}{2}$$
$$\text{Grounding Trigger: } D_{\text{eff}}(t, \mathbf{x}) \le Z_{\text{draft}}$$
When effective depth drops below keel draft, particle motion terminates immediately and registers a `GROUNDED_ON_REEF` event.

### G. Archival Knowledge Extraction Engine (`wiki_extractor.py`)
High-speed streaming parser extracting historical maritime wreck accounts and colonial roadsteads from compressed archives (`/mnt/storage/wiki/wiki_dump.json.bz2`):
* Filters articles within geographic bounds ($9^\circ\text{N}\text{--}35^\circ\text{N}$, $98^\circ\text{W}\text{--}55^\circ\text{W}$).
* Structures narrative loss accounts, dates, infoboxes, and coordinates.
* Generates an expanded catalog (100+ wrecks) with a stratified 75/25 train/test holdout partition.
