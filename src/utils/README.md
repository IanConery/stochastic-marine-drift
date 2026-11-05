# Geospatial, Geodesic & Paleoceanographic Utilities (`src/utils/`)

This package provides core mathematical routines for geodesic conversions, spatial metrics, paleoclimate adjustments, and search theory evaluation across the Stochastic Marine Drift framework.

---

## 1. Modules Summary

### A. Geodesic & Coordinate Transformations (`geo.py`)
Standardizes conversions between global WGS84 geographic coordinates (latitude, longitude in decimal degrees) and local metric Cartesian coordinates $(x, y)$ in meters:

* **Cartesian Offsets**:
  $$\Delta \phi = \frac{dy}{\text{METERS\_PER\_DEGREE\_LAT}}$$
  $$\Delta \lambda = \frac{dx}{\text{METERS\_PER\_DEGREE\_LAT} \cdot \cos(\phi)}$$
* **Great-Circle Distance (Haversine)**:
  $$a = \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)$$
  $$d = 2 R_{\text{earth}} \text{atan2}(\sqrt{a}, \sqrt{1-a})$$
* **Compass Azimuth Bearing**:
  $$\theta = \text{atan2}(\sin(\Delta \lambda)\cos(\phi_2), \, \cos(\phi_1)\sin(\phi_2) - \sin(\phi_1)\cos(\phi_2)\cos(\Delta \lambda))$$

---

## 2. Search Theory & Verification Metrics (`metrics.py`)
Quantifies framework accuracy against confirmed archaeological ground-truth coordinates:

1. **Containment Probability Rate ($P_{\text{contain}}$)**:
   Measures whether confirmed seafloor wreck coordinates fall within simulated Monte Carlo probability contours (50%, 80%, and 95%).
2. **Search Area Reduction (SAR) Ratio**:
   Measures efficiency gains over traditional circular dead-reckoning search boxes:
   $$\text{SAR} = \left(1 - \frac{\text{Area}_{\text{model\_95\%}}}{\text{Area}_{\text{naive\_LKP}}}\right) \times 100\%$$
   *Target*: $\ge 85\%\text{–}95\%$ reduction (achieved cohort mean: $96.5\%$).
3. **Centroid Localization Error**:
   Great-circle distance from the Maximum A Posteriori (MAP) probability peak to the documented primary ballast mound touchdown site.

---

## 3. Paleoclimate Seawater Correction (`paleo.py`)
* Adjusts modern Copernicus/ERA5 baselines to the Little Ice Age (1600–1850) using Caribbean coral geochemical proxy records (NOAA NCEI Paleo).
* Applies thermal offset ($\Delta T \approx -2^\circ\text{C}$), correcting seawater density $\rho_w$ and kinematic viscosity $\nu$.
