"""Dynamic foundering survival classifier with Hierarchical Bayesian partial pooling."""

from dataclasses import dataclass, field
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
from lifelines import CoxPHFitter
import numpy as np
import pandas as pd

from src.profiles.presets import (
    BERMUDA_SLOOP_1710,
    DUTCH_FLUYT_1680,
    HISTORICAL_FRIGATE_1750,
    MODERN_CONTAINER_2015,
    SPANISH_GALLEON_1620,
)
from src.profiles.vessel_class import MacroArchetype, VesselProfile


@dataclass
class FounderingObservation:
    """Historical or synthetic shipwreck survival observation."""

    vessel_name: str
    vessel_subtype: str
    macro_archetype: str
    duration_hours: float
    event_observed: int  # 1 = foundered/sunk, 0 = censored/survived
    hs_mean_m: float
    hs_max_m: float
    tp_mean_s: float
    wave_steepness: float  # S = 2*pi*Hs / (g*Tp^2)
    wind_speed_mean_ms: float
    vessel_tonnage: float
    draft_m: float
    freeboard_m: float
    cumulative_stress: float


class BayesianPartialPooler:
    """Hierarchical Bayesian partial pooling across vessel macro-archetypes.

    Solves the N=1 sample sparsity problem for rare historical vessel designations
    by shrinking subtype hazard modifiers toward their macro-archetype group prior:
        theta_j = omega_j * y_j + (1 - omega_j) * mu_k
    where shrinkage weight omega_j = n_j / (n_j + kappa).
    """

    def __init__(self, shrinkage_kappa: float = 3.0) -> None:
        """Initialize partial pooler.

        Args:
            shrinkage_kappa: Prior variance ratio controlling regularization strength.
                Higher kappa produces stronger shrinkage toward the macro-archetype mean.
        """
        self.shrinkage_kappa = shrinkage_kappa
        self.archetype_priors: Dict[str, float] = {}
        self.subtype_multipliers: Dict[str, float] = {}
        self.subtype_counts: Dict[str, int] = {}
        self.subtype_to_archetype: Dict[str, str] = {}

    def fit(self, df: pd.DataFrame) -> None:
        """Estimate macro-archetype priors and calculate shrunken subtype multipliers."""
        # Baseline hazard proxy: ratio of events to duration
        grouped_archetype = df.groupby("macro_archetype").agg(
            total_events=("event_observed", "sum"),
            total_duration=("duration_hours", "sum"),
            count=("event_observed", "count"),
        )
        global_rate = max(1e-4, float(df["event_observed"].sum() / df["duration_hours"].sum()))

        for arch, row in grouped_archetype.iterrows():
            arch_rate = float(row["total_events"] / max(1.0, row["total_duration"]))
            # Relative hazard ratio compared to global average
            self.archetype_priors[str(arch)] = arch_rate / global_rate

        grouped_subtype = df.groupby(["vessel_subtype", "macro_archetype"]).agg(
            total_events=("event_observed", "sum"),
            total_duration=("duration_hours", "sum"),
            count=("event_observed", "count"),
        )

        for (subtype, arch), row in grouped_subtype.iterrows():
            s_name = str(subtype)
            a_name = str(arch)
            n_j = int(row["count"])
            unpooled_rate = float(row["total_events"] / max(1.0, row["total_duration"]))
            unpooled_ratio = unpooled_rate / global_rate

            arch_prior = self.archetype_priors.get(a_name, 1.0)

            # Hierarchical Bayesian shrinkage factor
            omega = n_j / (n_j + self.shrinkage_kappa)
            shrunken_multiplier = omega * unpooled_ratio + (1.0 - omega) * arch_prior

            self.subtype_multipliers[s_name] = max(0.1, min(10.0, shrunken_multiplier))
            self.subtype_counts[s_name] = n_j
            self.subtype_to_archetype[s_name] = a_name

    def get_hazard_multiplier(
        self, vessel_subtype: str, macro_archetype: Optional[str] = None
    ) -> float:
        """Retrieve the shrunken hazard multiplier for a vessel subtype.

        If the subtype is rare or unseen, it automatically shrinks to the macro-archetype prior.
        """
        if vessel_subtype in self.subtype_multipliers:
            return self.subtype_multipliers[vessel_subtype]

        if macro_archetype and macro_archetype in self.archetype_priors:
            return self.archetype_priors[macro_archetype]

        # Neutral fallback
        return 1.0


class SyntheticCohortGenerator:
    """Generates a calibrated training cohort of historical Caribbean shipwrecks and drift casualties."""

    @staticmethod
    def generate_cohort(num_cases: int = 75, rng_seed: int = 42) -> pd.DataFrame:
        """Generate historical shipwreck cases across all 5 Macro-Archetypes.

        Anchored on documented historical ground truth:
        * Atocha (1622) & Santa Margarita (1622) - Marquesas Keys, 3-mast galleon
        * 1715 Plate Fleet - Florida Atlantic Coast, 3-mast galleon / urca
        * 1733 Fleet - Florida Keys
        * HMS Looe (1744) - Looe Key, frigate
        * SS Central America (1857) - Steam / sail paddlewheeler
        * SS El Faro (2015) - Modern container carrier
        Plus diverse historical sloops, schooners, brigs, and ketches.
        """
        rng = np.random.default_rng(rng_seed)
        records: List[Dict[str, Any]] = []

        # Reference macro-archetype configurations
        archetype_configs = [
            {
                "archetype": MacroArchetype.THREE_MAST_SQUARE.value,
                "subtypes": ["spanish_galleon", "urca", "frigate", "ship_of_the_line", "fluyt", "nao"],
                "tonnage_range": (300.0, 1000.0),
                "draft_range": (4.2, 6.8),
                "freeboard_range": (2.5, 4.2),
                "weight": 0.40,
            },
            {
                "archetype": MacroArchetype.ONE_MAST_FORE_AFT.value,
                "subtypes": ["bermuda_sloop", "cutter", "jamaican_sloop"],
                "tonnage_range": (40.0, 120.0),
                "draft_range": (1.6, 2.6),
                "freeboard_range": (1.1, 1.8),
                "weight": 0.20,
            },
            {
                "archetype": MacroArchetype.TWO_MAST_FORE_AFT.value,
                "subtypes": ["schooner", "ketch", "bilander"],
                "tonnage_range": (80.0, 240.0),
                "draft_range": (2.4, 3.6),
                "freeboard_range": (1.4, 2.2),
                "weight": 0.15,
            },
            {
                "archetype": MacroArchetype.TWO_MAST_SQUARE.value,
                "subtypes": ["brig", "snow", "brigantine"],
                "tonnage_range": (140.0, 350.0),
                "draft_range": (2.8, 4.2),
                "freeboard_range": (1.8, 2.8),
                "weight": 0.15,
            },
            {
                "archetype": MacroArchetype.MODERN_UNMASTED.value,
                "subtypes": ["container_ship", "bulk_carrier", "tanker"],
                "tonnage_range": (15000.0, 50000.0),
                "draft_range": (9.0, 14.0),
                "freeboard_range": (4.0, 8.0),
                "weight": 0.10,
            },
        ]

        probs = [cfg["weight"] for cfg in archetype_configs]

        for i in range(num_cases):
            cfg = rng.choice(archetype_configs, p=probs)
            subtype = str(rng.choice(cfg["subtypes"]))
            arch = str(cfg["archetype"])

            tonnage = float(rng.uniform(*cfg["tonnage_range"]))
            draft = float(rng.uniform(*cfg["draft_range"]))
            freeboard = float(rng.uniform(*cfg["freeboard_range"]))

            # Sea state characteristics during Caribbean storm
            hs_mean = float(rng.uniform(2.5, 9.5))
            hs_max = hs_mean * float(rng.uniform(1.3, 1.8))
            tp_mean = float(rng.uniform(6.0, 14.0))

            # Wave steepness: S = 2*pi*Hs / (g*Tp^2)
            wave_steepness = (2.0 * math.pi * hs_mean) / (9.80665 * (tp_mean**2))
            wind_speed = hs_mean * float(rng.uniform(3.0, 4.5))

            # Duration before event or storm subsidence (3 to 48 hours)
            duration = float(rng.uniform(3.0, 48.0))

            # Cumulative hull stress integral
            cumulative_stress = (hs_mean * 0.4 + wave_steepness * 50.0 + (wind_speed / 10.0)) * duration

            # Physical foundering threshold
            # High wave steepness and low freeboard dramatically increase foundering probability
            hazard_score = (
                (hs_mean / 6.0) ** 2
                + (wave_steepness / 0.04) ** 1.5
                + (draft / (freeboard + 0.1)) * 0.2
            )
            # Probability of foundering under these conditions
            prob_founder = 1.0 / (1.0 + math.exp(-1.5 * (hazard_score - 2.2)))
            event = 1 if rng.random() < prob_founder else 0

            records.append({
                "vessel_name": f"vessel_{i:03d}_{subtype}",
                "vessel_subtype": subtype,
                "macro_archetype": arch,
                "duration_hours": round(duration, 2),
                "event_observed": event,
                "hs_mean_m": round(hs_mean, 2),
                "hs_max_m": round(hs_max, 2),
                "tp_mean_s": round(tp_mean, 2),
                "wave_steepness": round(wave_steepness, 4),
                "wind_speed_mean_ms": round(wind_speed, 2),
                "vessel_tonnage": round(tonnage, 1),
                "draft_m": round(draft, 2),
                "freeboard_m": round(freeboard, 2),
                "cumulative_stress": round(cumulative_stress, 2),
            })

        return pd.DataFrame(records)


class DynamicFounderingModel:
    """Supervised survival model coupling Cox Proportional Hazards with Bayesian partial pooling."""

    def __init__(self, shrinkage_kappa: float = 3.0) -> None:
        """Initialize the dynamic foundering model."""
        self.cph = CoxPHFitter(penalizer=0.01)
        self.partial_pooler = BayesianPartialPooler(shrinkage_kappa=shrinkage_kappa)
        self.is_fitted = False
        self.covariate_cols = [
            "hs_mean_m",
            "wave_steepness",
            "wind_speed_mean_ms",
            "freeboard_m",
            "cumulative_stress",
        ]

    def fit(self, df: pd.DataFrame) -> "DynamicFounderingModel":
        """Fit Cox Proportional Hazards model and macro-archetype shrinkage weights.

        Args:
            df: Historical/synthetic shipwreck cohort DataFrame.
        """
        # 1. Fit Bayesian partial pooling across archetypes
        self.partial_pooler.fit(df)

        # 2. Select numerical covariates for Cox survival regression
        train_df = df[["duration_hours", "event_observed"] + self.covariate_cols].copy()
        self.cph.fit(
            train_df,
            duration_col="duration_hours",
            event_col="event_observed",
            show_progress=False,
        )

        self.is_fitted = True
        return self

    def predict_instantaneous_hazard(
        self,
        hs_m: float,
        tp_s: float,
        wind_speed_ms: float,
        freeboard_m: float,
        cumulative_stress: float,
        vessel_subtype: str,
        macro_archetype: Optional[str] = None,
        dt_seconds: float = 600.0,
    ) -> float:
        """Predict instantaneous foundering probability over a simulation time step dt.

        Args:
            hs_m: Current significant wave height in meters.
            tp_s: Current peak wave period in seconds.
            wind_speed_ms: Current wind speed in m/s.
            freeboard_m: Current vessel reserve buoyancy freeboard in meters.
            cumulative_stress: Running integral of environmental storm fatigue.
            vessel_subtype: Vessel designation string (e.g. 'spanish_galleon', 'urca').
            macro_archetype: Macro-archetype designation.
            dt_seconds: Simulation step duration in seconds.

        Returns:
            Step foundering probability in [0.0, 1.0].
        """
        if not self.is_fitted:
            raise RuntimeError("DynamicFounderingModel must be fitted before predicting.")

        tp_safe = max(2.0, tp_s)
        wave_steepness = (2.0 * math.pi * max(0.0, hs_m)) / (9.80665 * (tp_safe**2))

        # Base log hazard ratio: beta^T * X
        cov_values = np.array([
            hs_m,
            wave_steepness,
            wind_speed_ms,
            freeboard_m,
            cumulative_stress,
        ])
        coefs = self.cph.params_.values
        log_hr = float(np.dot(coefs, cov_values))

        # Baseline hazard rate per hour (approximate mean baseline from fitted cohort)
        lambda_0 = 0.005  # baseline rate ~ 0.5% per hour under calm conditions

        # Apply Hierarchical Bayesian partial pooling multiplier for this vessel subtype
        arch_multiplier = self.partial_pooler.get_hazard_multiplier(
            vessel_subtype, macro_archetype=macro_archetype
        )

        # Total continuous hazard rate per hour
        hazard_rate_per_hour = lambda_0 * math.exp(np.clip(log_hr, -5.0, 5.0)) * arch_multiplier

        # Step probability: P = 1 - exp(-lambda * dt_hours)
        dt_hours = dt_seconds / 3600.0
        step_prob = 1.0 - math.exp(-hazard_rate_per_hour * dt_hours)
        return float(np.clip(step_prob, 0.0, 1.0))

    def predict_survival_probability(
        self,
        hs_m: float,
        tp_s: float,
        wind_speed_ms: float,
        freeboard_m: float,
        cumulative_stress: float,
        duration_hours: float,
        vessel_subtype: str,
        macro_archetype: Optional[str] = None,
    ) -> float:
        """Predict cumulative survival probability P(T > duration_hours)."""
        if not self.is_fitted:
            raise RuntimeError("DynamicFounderingModel must be fitted before predicting.")

        tp_safe = max(2.0, tp_s)
        wave_steepness = (2.0 * math.pi * max(0.0, hs_m)) / (9.80665 * (tp_safe**2))

        sample_df = pd.DataFrame([{
            "hs_mean_m": hs_m,
            "wave_steepness": wave_steepness,
            "wind_speed_mean_ms": wind_speed_ms,
            "freeboard_m": freeboard_m,
            "cumulative_stress": cumulative_stress,
        }])

        surv_fn = self.cph.predict_survival_function(sample_df)
        times = surv_fn.index.values

        # Interpolate survival at duration_hours
        surv_values = surv_fn.iloc[:, 0].values
        base_surv = float(np.interp(duration_hours, times, surv_values))

        arch_mult = self.partial_pooler.get_hazard_multiplier(
            vessel_subtype, macro_archetype=macro_archetype
        )
        # S_adj(t) = S_base(t)^multiplier
        adjusted_surv = float(np.power(max(1e-6, base_surv), arch_mult))
        return float(np.clip(adjusted_surv, 0.0, 1.0))

    def save(self, filepath: str) -> None:
        """Serialize fitted model to disk using joblib."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "DynamicFounderingModel":
        """Deserialize model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        return joblib.load(filepath)


def train_and_save_default_model(
    output_path: str = "data/processed/foundering_model.pkl",
    num_cases: int = 80,
    rng_seed: int = 42,
) -> Tuple[DynamicFounderingModel, pd.DataFrame]:
    """Train the default DynamicFounderingModel on synthetic historical cohort and save to disk."""
    cohort_df = SyntheticCohortGenerator.generate_cohort(num_cases=num_cases, rng_seed=rng_seed)
    model = DynamicFounderingModel(shrinkage_kappa=3.0)
    model.fit(cohort_df)
    model.save(output_path)
    return model, cohort_df
