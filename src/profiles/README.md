# Vessel Profiles & Leeway Taxonomy (`src/profiles/`)

This package defines the hydrodynamic, aerodynamic, and physical properties governing how vessels drift on the surface, respond to wave stress, and descend through the water column.

---

## 1. The 2D Classification Matrix (Mast Count x Rig Type)

To eliminate the $N=1$ sample sparsity problem common in historical archives—where dozens of obscure vessel names (*nao*, *urca*, *pinnace*, *snow*, *ketch*, *bilander*) have few documented wrecks—vessels are grouped into **5 physical Macro-Archetypes** defined by **standing mast count** and **primary sail plan orientation**:

```
                              PRIMARY RIG ORIENTATION
                     Square-Rigged          Fore-and-Aft Rigged
                 +----------------------+--------------------------+
          1 Mast | (Rare historical)    | ONE_MAST_FORE_AFT        |  <-- Sloop, Cutter
                 |                      | (High CWL, agile, shallow)|
                 +----------------------+--------------------------+
         2 Masts | TWO_MAST_SQUARE      | TWO_MAST_FORE_AFT        |  <-- Schooner, Ketch
MAST             | (Brig, Snow)         | (Bimodal zigzag drift)   |
COUNT            +----------------------+--------------------------+
        3+ Masts | THREE_MAST_SQUARE    | (Rare historical)        |  <-- Galleon, Frigate, Nao
                 | (High DWL, deep keel)|                          |
                 +----------------------+--------------------------+
         0 Masts | MODERN_UNMASTED (Powered steel, hydrodynamic glide)
                 +-------------------------------------------------+
```

### Macro-Archetype Properties

| Macro-Archetype | Typical Historical Designations | Draft Range | Downwind Leeway ($a_{dw}$) | Crosswind Divergence ($\theta_{cw}$) | Typical Ballast Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `1_MAST_FORE_AFT` | Bermuda Sloop, Cutter, Jamaican Sloop | $1.5\text{–}2.8\text{ m}$ | $0.020$ ($2.0\%$) | $26.0^\circ$ | $0.20$ |
| `2_MAST_FORE_AFT` | Schooner, Ketch, Bilander | $2.5\text{–}3.8\text{ m}$ | $0.025$ ($2.5\%$) | $20.0^\circ$ | $0.25$ |
| `2_MAST_SQUARE` | Brig, Snow, Brigantine | $3.0\text{–}4.2\text{ m}$ | $0.032$ ($3.2\%$) | $6.0^\circ$ | $0.28$ |
| `3_MAST_SQUARE` | Spanish Galleon, Frigate, Nao, Fluyt | $4.5\text{–}7.0\text{ m}$ | $0.040$ ($4.0\%$) | $2.0^\circ$ | $0.33$ |
| `MODERN_UNMASTED` | Container Ship, Bulk Carrier, Tanker | $8.0\text{–}14.0\text{ m}$| $0.018$ ($1.8\%$) | $10.0^\circ$ | $0.15$ |

---

## 2. Continuous Volumetric Scaling

When displacement tonnage ($M_{\text{tons}}$) is extracted from archival manifests, parameters scale continuously rather than relying on static discrete lookup tables:

$$\text{Scale Factor } s = \max\left(0.6, \, \min\left(1.8, \, \left(\frac{M_{\text{tons}}}{M_{\text{ref}}}\right)^{1/3}\right)\right)$$
$$\text{Draft: } T = T_{\text{base}} \times s$$
$$\text{Freeboard: } F = F_{\text{base}} \times s^{0.8}$$
$$\text{Downwind Leeway: } a_{dw} = \frac{a_{\text{base}}}{s^{0.2}}$$

This ensures an 800-ton galleon and a 400-ton galleon behave with physically distinct drafts, windage values, and descent profiles while sharing the same underlying aerodynamic archetype.

---

## 3. Core Files

* [`vessel_class.py`](vessel_class.py): Contains the immutable `VesselProfile` dataclass, parameter validation checks, `RigType` enum, and `MacroArchetype` enum.
* [`presets.py`](presets.py): Pre-calibrated vessel instances (`SPANISH_GALLEON_1620`, `BERMUDA_SLOOP_1710`, etc.) and the `profile_from_masts_and_rig(...)` factory function.
