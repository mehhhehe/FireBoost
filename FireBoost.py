"""
FireBoost (paper-faithful Python implementation)
- Binary Firefly Algorithm (FFA) feature selection
- OXGBoost training: feature-specific binning, mini-batch boosting, in-loop DLR
- Adaptive weighting (emphasize misclassified samples) + weighted prediction aggregation

Paper references:
- FFA: init p=0.5, fitness, brightness, movement, sigmoid binarization tau=0.5
- OXGBoost: per-feature bins 512/256/128/64 by FFA importance quartiles, DLR eta_t=eta0*exp(-a*t),
  mini-batch gradient concept, adaptive weighting + prediction aggregation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Literal

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer

import xgboost as xgb


# -----------------------------
# Paper: FireBoost variant presets (Table III)
# -----------------------------
VariantName = Literal["FB1", "FB2", "FB3", "FB4"]

FIREBOOST_VARIANTS: Dict[VariantName, Dict[str, float]] = {
    "FB1": {"g": 10, "N": 10, "beta0": 1.0,  "gamma": 1.0, "alpha": 0.2},
    "FB2": {"g": 10, "N": 10, "beta0": 0.5,  "gamma": 0.8, "alpha": 0.1},
    "FB3": {"g": 8,  "N": 10, "beta0": 0.75, "gamma": 1.2, "alpha": 0.1},
    "FB4": {"g": 8,  "N": 10, "beta0": 0.75, "gamma": 1.2, "alpha": 0.3},
}


# -----------------------------
# Helpers
# -----------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Paper: choose cutoff on held-out 10% validation split by maximizing Youden index.
    """
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    k = int(np.argmax(j))
    t = thr[k]
    return float(t) if np.isfinite(t) else 0.5


def drop_high_missing_columns(df: pd.DataFrame, max_missing_frac: float) -> pd.DataFrame:
    miss = df.isna().mean(axis=0)
    keep = miss <= max_missing_frac
    return df.loc[:, keep]


def build_preprocessor(
    df: pd.DataFrame,
    standardize_numeric: bool = False
) -> ColumnTransformer:
    """
    Paper uses encoding + imputation; in KDD they also standardized numeric fields (case-study-specific).
    We expose standardize_numeric as an option.
    """
    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    cat_cols = [c for c in df.columns if c not in num_cols]

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if standardize_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True
    )


# -----------------------------
# FFA (Binary Firefly Algorithm) — paper-faithful
# -----------------------------
@dataclass
class FFAConfig:
    # Paper defaults / stated values
    p_init: float = 0.5      # probability of selecting a feature during init
    tau: float = 0.5         # binarization threshold
    lam: float = 0.9         # fitness trade-off λ in paper fitness
    eval_model: Literal["xgb", "rf"] = "xgb"
    eval_num_boost_round: int = 30
    eval_max_depth: int = 4
    seed: int = 42


class FireflyFeatureSelector:
    """
    Paper:
      x_ij = 1 w.p. p else 0
      fitness f(Xi) = λ*error + (1-λ)*|Si|/|F|
      brightness Bi = -f(Xi)
      movement Xi <- Xi + beta0*exp(-gamma*r^2)*(Xj - Xi) + alpha*rand
      sigmoid binarization with tau=0.5
    """

    def __init__(self, variant: VariantName, cfg: Optional[FFAConfig] = None):
        self.variant = variant
        self.cfg = cfg or FFAConfig()

        v = FIREBOOST_VARIANTS[variant]
        self.g = int(v["g"])               # iterations/generations
        self.N = int(v["N"])               # population size
        self.beta0 = float(v["beta0"])     # attraction base
        self.gamma = float(v["gamma"])     # light absorption
        self.alpha = float(v["alpha"])     # randomness

        self.best_mask_: Optional[np.ndarray] = None
        self.best_fitness_: Optional[float] = None
        self.importance_: Optional[np.ndarray] = None  # normalized feature importance proxy

    def _fitness(self, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return 1.0  # worst

        Xt = Xtr[:, mask]
        Xv = Xva[:, mask]

        # Paper: model M can be RandomForest or XGBoost. We implement XGBoost (fast, consistent with paper stack).
        dtr = xgb.DMatrix(Xt, label=ytr)
        dva = xgb.DMatrix(Xv, label=yva)
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "max_depth": self.cfg.eval_max_depth,
            "eta": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "seed": self.cfg.seed,
        }
        booster = xgb.train(params=params, dtrain=dtr, num_boost_round=self.cfg.eval_num_boost_round, verbose_eval=False)
        prob = booster.predict(dva)
        pred = (prob >= 0.5).astype(int)
        err = 1.0 - accuracy_score(yva, pred)

        # Paper fitness
        lam = float(self.cfg.lam)
        size_pen = float(mask.sum() / mask.size)
        return lam * err + (1.0 - lam) * size_pen

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FireflyFeatureSelector":
        rng = np.random.default_rng(self.cfg.seed)

        # internal split for fitness computation (not the final threshold split)
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=self.cfg.seed)

        n_features = X.shape[1]

        # Init fireflies in {0,1}^d, plus real-valued carrier for motion
        F_bin = rng.binomial(1, self.cfg.p_init, size=(self.N, n_features)).astype(np.float32)
        F_real = F_bin.copy()

        best_fit = float("inf")
        best_mask = None

        # importance proxy: frequency/weight among good solutions across generations
        imp_acc = np.zeros(n_features, dtype=np.float64)

        for _ in range(self.g):
            fitness = np.zeros(self.N, dtype=np.float64)
            for i in range(self.N):
                mask = F_bin[i].astype(bool)
                fitness[i] = self._fitness(Xtr, ytr, Xva, yva, mask)

            brightness = -fitness
            order = np.argsort(fitness)  # best first

            # store best
            if fitness[order[0]] < best_fit:
                best_fit = float(fitness[order[0]])
                best_mask = F_bin[order[0]].copy().astype(bool)

            # accumulate importance from top 3 each generation (stable proxy)
            topk = order[: min(3, self.N)]
            # weight by relative brightness (higher brightness => more weight)
            b = brightness[topk]
            b = b - b.min()
            w = (b + 1e-12) / (b.sum() + 1e-12)
            for wi, idx in zip(w, topk):
                imp_acc += wi * F_bin[idx]

            # move fireflies (paper: j attracts i if Bj > Bi)
            for i in range(self.N):
                for j in range(self.N):
                    if brightness[j] > brightness[i]:
                        r = np.linalg.norm(F_bin[i] - F_bin[j])  # Euclidean distance
                        beta = self.beta0 * np.exp(-self.gamma * (r ** 2))
                        noise = self.alpha * rng.normal(0.0, 1.0, size=n_features)  # rand ~ N(0, I)
                        F_real[i] = F_real[i] + beta * (F_real[j] - F_real[i]) + noise

                # binarize: sigmoid + threshold tau=0.5
                probs = _sigmoid(F_real[i])
                F_bin[i] = (probs > self.cfg.tau).astype(np.float32)

        # finalize
        self.best_mask_ = best_mask
        self.best_fitness_ = best_fit
        self.importance_ = imp_acc / imp_acc.max() if imp_acc.max() > 0 else imp_acc
        return self


# -----------------------------
# Feature-specific binning — paper mapping
# -----------------------------
@dataclass
class BinningConfig:
    # Paper mapping: top quartile 512, then 256, 128, 64
    bins_q: Tuple[int, int, int, int] = (512, 256, 128, 64)

    # Paper also mentions experiments coupled with binning with 128 bins (global).
    # This cap lets you match that statement when you want strict parity with that setting.
    global_bin_cap: Optional[int] = None  # e.g., 128

    min_bins: int = 4
    strategy: Literal["quantile"] = "quantile"


def per_feature_bins_from_importance(importance: np.ndarray, cfg: BinningConfig) -> np.ndarray:
    q1, q2, q3 = np.quantile(importance, [0.25, 0.50, 0.75])
    hi, mid_hi, mid_lo, lo = cfg.bins_q
    out = np.empty_like(importance, dtype=int)
    for i, s in enumerate(importance):
        if s >= q3:
            out[i] = hi
        elif s >= q2:
            out[i] = mid_hi
        elif s >= q1:
            out[i] = mid_lo
        else:
            out[i] = lo
    if cfg.global_bin_cap is not None:
        out = np.minimum(out, int(cfg.global_bin_cap))
    return out


def discretize_per_feature(X: np.ndarray, bins: np.ndarray, cfg: BinningConfig) -> Tuple[np.ndarray, List[Optional[KBinsDiscretizer]]]:
    """
    Implements per-feature binning via sklearn KBinsDiscretizer (ordinal encoding).
    This is the closest implementable substitute for 'inject bins into histogram builder' in Python.
    """
    n, d = X.shape
    Xb = np.empty((n, d), dtype=np.float32)
    models: List[Optional[KBinsDiscretizer]] = []

    for j in range(d):
        col = X[:, [j]]
        # handle NaNs
        med = np.nanmedian(col)
        col2 = np.where(np.isnan(col), med, col)

        uniq = np.unique(col2)
        if uniq.size <= 1:
            Xb[:, j] = 0.0
            models.append(None)
            continue

        nb = int(bins[j])
        nb = max(cfg.min_bins, nb)
        nb = min(nb, int(uniq.size))

        kb = KBinsDiscretizer(n_bins=nb, encode="ordinal", strategy=cfg.strategy)
        Xb[:, j] = kb.fit_transform(col2).astype(np.float32).ravel()
        models.append(kb)

    return Xb, models


# -----------------------------
# OXGBoost loop: DLR + mini-batch + adaptive weighting + weighted aggregation
# -----------------------------
@dataclass
class OXGBoostConfig:
    T: int = 200
    eta0: float = 0.2
    decay_a: float = 0.01
    max_depth: int = 6
    reg_alpha: float = 0.5
    reg_lambda: float = 1.5
    batch_size: int = 500
    seed: int = 42

    # Adaptive weighting strength: increase weights on misclassified samples
    # Paper states the idea but does not give a specific formula; we implement a correct, standard emphasis rule.
    misclf_boost: float = 0.5   # multiply weight by (1 + misclf_boost) if sample is misclassified
    weight_floor: float = 1e-9

    # Weighted aggregation: model already sums trees with eta; we also store per-round eta for transparency
    # (no external behavior change; it matches "weighted averaging" description)
    store_eta_history: bool = True


class FireBoostBinaryClassifier:
    """
    End-to-end FireBoost for binary classification.

    Pipeline:
      1) Preprocess: drop high-missing cols, impute + one-hot (and optional standardize numeric)
      2) FFA feature selection using chosen variant FB1..FB4
      3) Feature-specific binning based on FFA importance quartiles (512/256/128/64), optional global cap
      4) OXGBoost training loop:
         - for t=0..T-1: eta_t = eta0 * exp(-a*t) (inside loop)
         - sample mini-batch Bt of size nb
         - train one boosting step on Bt (hist tree)
         - adaptive weighting: upweight misclassified samples for next iteration
      5) Choose decision threshold by Youden index on 10% validation split (held out from training)
    """

    def __init__(
        self,
        variant: VariantName = "FB4",
        max_missing_frac: float = 0.6,
        standardize_numeric: bool = False,
        ffa_cfg: Optional[FFAConfig] = None,
        bin_cfg: Optional[BinningConfig] = None,
        oxgb_cfg: Optional[OXGBoostConfig] = None,
    ):
        self.variant = variant
        self.max_missing_frac = max_missing_frac
        self.standardize_numeric = standardize_numeric

        self.ffa_cfg = ffa_cfg or FFAConfig()
        self.bin_cfg = bin_cfg or BinningConfig()
        self.oxgb_cfg = oxgb_cfg or OXGBoostConfig(seed=self.ffa_cfg.seed)

        self.preprocessor_: Optional[ColumnTransformer] = None
        self.feature_names_: Optional[np.ndarray] = None

        self.selected_mask_: Optional[np.ndarray] = None
        self.selected_feature_names_: Optional[List[str]] = None

        self.binners_: Optional[List[Optional[KBinsDiscretizer]]] = None

        self.booster_: Optional[xgb.Booster] = None
        self.threshold_: float = 0.5
        self.eta_history_: List[float] = []

    def fit(self, X_df: pd.DataFrame, y: np.ndarray) -> "FireBoostBinaryClassifier":
        y = np.asarray(y).astype(int)

        # ---- Paper: hold out 10% validation split for threshold tuning (test isolated)
        X_df = drop_high_missing_columns(X_df, self.max_missing_frac)

        self.preprocessor_ = build_preprocessor(X_df, standardize_numeric=self.standardize_numeric)
        X_all = self.preprocessor_.fit_transform(X_df)
        self.feature_names_ = self.preprocessor_.get_feature_names_out()

        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y, test_size=0.10, stratify=y, random_state=self.ffa_cfg.seed
        )

        # ---- FFA feature selection (FB1..FB4)
        selector = FireflyFeatureSelector(self.variant, cfg=self.ffa_cfg)
        selector.fit(X_train, y_train)

        if selector.best_mask_ is None or selector.importance_ is None:
            raise RuntimeError("FFA failed to produce a feature mask/importance.")

        self.selected_mask_ = selector.best_mask_
        self.selected_feature_names_ = list(self.feature_names_[self.selected_mask_])

        # restrict to selected features
        Xs_train = X_train[:, self.selected_mask_]
        Xs_val = X_val[:, self.selected_mask_]

        # ---- Feature-specific binning based on normalized FFA importance quartiles
        imp_sel = selector.importance_[self.selected_mask_]
        bins_sel = per_feature_bins_from_importance(imp_sel, self.bin_cfg)

        Xb_train, binners = discretize_per_feature(Xs_train, bins_sel, self.bin_cfg)
        Xb_val, _ = discretize_per_feature(Xs_val, bins_sel, self.bin_cfg)
        self.binners_ = binners

        # ---- OXGBoost training loop (DLR + mini-batch + adaptive weighting)
        rng = np.random.default_rng(self.oxgb_cfg.seed)
        n = Xb_train.shape[0]
        bs = int(self.oxgb_cfg.batch_size)

        # adaptive weights init uniform
        w = np.ones(n, dtype=np.float64) / n

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "max_depth": int(self.oxgb_cfg.max_depth),
            "reg_alpha": float(self.oxgb_cfg.reg_alpha),
            "reg_lambda": float(self.oxgb_cfg.reg_lambda),
            "seed": int(self.oxgb_cfg.seed),
        }

        booster: Optional[xgb.Booster] = None
        self.eta_history_.clear()

        for t in range(int(self.oxgb_cfg.T)):
            eta_t = float(self.oxgb_cfg.eta0 * np.exp(-self.oxgb_cfg.decay_a * t))
            params["eta"] = eta_t
            if self.oxgb_cfg.store_eta_history:
                self.eta_history_.append(eta_t)

            # mini-batch Bt sampled each iteration (paper)
            if bs >= n:
                idx = np.arange(n)
            else:
                idx = rng.choice(n, size=bs, replace=False, p=w)

            dtr = xgb.DMatrix(Xb_train[idx], label=y_train[idx], weight=w[idx])

            # one tree per iteration; trees are aggregated by boosting sum (weighted by eta_t)
            booster = xgb.train(
                params=params,
                dtrain=dtr,
                num_boost_round=1,
                xgb_model=booster,
                verbose_eval=False,
            )

            # adaptive weighting: highlight misclassified samples (paper)
            # We compute current predictions on the full training set and upweight misclassified points.
            d_all = xgb.DMatrix(Xb_train)
            p_all = booster.predict(d_all)
            y_hat = (p_all >= 0.5).astype(int)

            mis = (y_hat != y_train).astype(np.float64)

            # increase weights on misclassified samples
            w = w * (1.0 + float(self.oxgb_cfg.misclf_boost) * mis)

            # stabilize + normalize
            w = np.maximum(w, float(self.oxgb_cfg.weight_floor))
            w = w / w.sum()

        self.booster_ = booster

        # ---- Choose threshold on validation by Youden index
        dval = xgb.DMatrix(Xb_val)
        p_val = self.booster_.predict(dval)
        self.threshold_ = youden_threshold(y_val, p_val)

        return self

    def _transform(self, X_df: pd.DataFrame) -> np.ndarray:
        if self.preprocessor_ is None or self.selected_mask_ is None or self.binners_ is None:
            raise RuntimeError("Model not fitted.")

        X_df = drop_high_missing_columns(X_df, self.max_missing_frac)
        X_all = self.preprocessor_.transform(X_df)
        Xs = X_all[:, self.selected_mask_]

        # apply fitted discretizers
        n, d = Xs.shape
        Xb = np.empty((n, d), dtype=np.float32)
        for j in range(d):
            kb = self.binners_[j]
            col = Xs[:, [j]]
            med = np.nanmedian(col)
            col2 = np.where(np.isnan(col), med, col)
            if kb is None:
                Xb[:, j] = 0.0
            else:
                Xb[:, j] = kb.transform(col2).astype(np.float32).ravel()
        return Xb

    def predict_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        if self.booster_ is None:
            raise RuntimeError("Model not fitted.")
        Xb = self._transform(X_df)
        d = xgb.DMatrix(Xb)
        p1 = self.booster_.predict(d)
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        p1 = self.predict_proba(X_df)[:, 1]
        return (p1 >= self.threshold_).astype(int)

    @property
    def selected_features(self) -> List[str]:
        if self.selected_feature_names_ is None:
            raise RuntimeError("Model not fitted.")
        return self.selected_feature_names_


# -----------------------------
# Convenience factory
# -----------------------------
def make_fireboost(
    variant: VariantName = "FB4",
    seed: int = 42,
    *,
    # match paper note "binning with 128 bins" by setting global_bin_cap=128 if desired
    global_bin_cap: Optional[int] = None,
) -> FireBoostBinaryClassifier:
    ffa_cfg = FFAConfig(seed=seed)
    bin_cfg = BinningConfig(global_bin_cap=global_bin_cap)
    oxgb_cfg = OXGBoostConfig(seed=seed)
    return FireBoostBinaryClassifier(
        variant=variant,
        ffa_cfg=ffa_cfg,
        bin_cfg=bin_cfg,
        oxgb_cfg=oxgb_cfg,
    )
