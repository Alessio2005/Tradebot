"""FeatureOrthogonalizer — PCA + Spearman dedup for feature matrices.

Migrated from legacy ta_features.py.
Orthogonalises correlated feature blocks so that CatBoost sees
near-independent inputs. Designed for CPCV: per-fold fit avoids leakage.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE: bool = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    PCA = None  # type: ignore[assignment,misc]
    StandardScaler = None  # type: ignore[assignment,misc]

try:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from scipy.stats import spearmanr as _spearmanr
    _SCIPY_AVAILABLE: bool = True
    spearmanr_func = _spearmanr
except Exception:
    _SCIPY_AVAILABLE = False
    fcluster = linkage = squareform = spearmanr_func = None  # type: ignore[assignment]

class FeatureOrthogonalizer:
    """PCA-orthogonalisatie binnen een feature-cluster (micro / meso / macro).

    Probleem — substitutie-effect in CatBoost:
        Gecorreleerde features (bijv. meerdere ATR-varianten, meerdere
        FFD-orders) splitsen de Feature Importance. CatBoost kiest willekeurig
        één van de near-duplicaten, wat CFI instabiel maakt en het model doet
        "verwarren" bij gecorreleerde inputs.

    Oplossing:
        PCA per cluster → orthogonale hoofdcomponenten als input.
        De componenten zijn per definitie ongecorreleerd (r² = 0), hebben
        maximale informatiedichtheid, en verwijderen de substitutie-redundantie.

    Leakage-preventie (standaard):
        PCA wordt EENMALIG gecalibreerd op de eerste ``calib_frac`` × n bars
        (default 60%) en daarna gecacht.  Alle volgende aanroepen (ook live)
        hergebruiken exact dezelfde transformatie.

    H2-WAARSCHUWING — CPCV lekkage:
        ``fit_transform()`` transformeert de volledige dataset (100%) met een
        PCA die op 60% is gefit.  Als CPCV test-folds vroege periodes bevatten
        die in die 60% vallen, zijn de PCA-features van die test-bars niet
        volledig leekvrij.  Gevolg: backtestprestaties licht overschat voor
        vroege test-folds.

        Gebruik ``fit_on_train_indices(X, train_idx, ...)`` per CPCV-fold om
        de lekkage volledig te elimineren.  De PCA wordt dan per fold gefit
        op uitsluitend de trainingsdata van die fold.

    Gebruik (standaard, pre-CPCV):
        orth = FeatureOrthogonalizer(n_components=0.95)
        X_orth, names_orth = orth.fit_transform(X, feature_names, prefix="micro_pc")
        # ... sla orth op via joblib ...
        X_live = orth.transform(X_live_row)

    Gebruik (per CPCV-fold, geen lekkage):
        for train_idx, val_idx, _ in cv.split(...):
            X_fold, names = orth.fit_on_train_indices(X_raw, train_idx, names_raw)
            x_tr, x_val = X_fold[train_idx], X_fold[val_idx]

    Args:
        n_components  : variantie-verklaringsdrempel voor PCA (default 0.95).
        min_components: minimum aantal te behouden componenten (default 2).
        calib_frac    : fractie van de data waarop PCA wordt gefit (default 0.60).
        max_calib_bars: absoluut maximum kalibratie-bars (default 5000).
    """

    # AUDIT-FIX (N20 — per-asset Spearman dedup threshold):
    #   0.25 te agressief voor SOL (features minder gecorreleerd, meer idiosyncratisch
    #   door validator-uitval, MEV-cycles). BTC heeft sterk gecorreleerde ATR-varianten
    #   → agressiever dedup gewenst.  Per-asset aanbeveling:
    #     BTC: 0.10 (|ρ|≥0.90), ETH: 0.15 (|ρ|≥0.85), SOL: 0.20 (|ρ|≥0.80)
    #   Geef threshold expliciet via cfg of FeatureOrthogonalizer constructor.
    ASSET_SPEARMAN_THRESHOLDS: dict[str, float] = {  # noqa: RUF012
        "BTCUSDT": 0.10,
        "BTCBUSD": 0.10,
        "ETHUSDT": 0.15,
        "ETHBUSD": 0.15,
        "SOLUSDT": 0.20,
        "SOLBUSD": 0.20,
    }

    def __init__(
        self,
        n_components: float = 0.95,
        min_components: int = 2,
        calib_frac: float = 0.60,
        max_calib_bars: int = 5_000,
        spearman_prefilter: bool = True,
        spearman_cluster_threshold: float = 0.25,
        symbol: str | None = None,
        # AUDIT-FIX (N20): als symbol is opgegeven, wordt spearman_cluster_threshold
        # automatisch overschreven met de asset-specifieke waarde uit
        # ASSET_SPEARMAN_THRESHOLDS, tenzij de caller al een expliciete waarde
        # (≠ default 0.25) heeft doorgegeven.
    ) -> None:
        self.n_components   = n_components
        self.min_components = min_components
        self.calib_frac     = calib_frac
        self.max_calib_bars = max_calib_bars

        # N20: automatisch per-asset threshold als symbol is opgegeven
        if symbol is not None and spearman_cluster_threshold == 0.25:
            spearman_cluster_threshold = self.ASSET_SPEARMAN_THRESHOLDS.get(
                symbol,
                # Prefix-match fallback (BTC, ETH, SOL)
                next(
                    (v for k, v in self.ASSET_SPEARMAN_THRESHOLDS.items() if symbol.startswith(k[:3])),
                    0.25,
                ),
            )
        self.symbol: str | None = symbol

        # PRE-PCA FIX: hiërarchische Spearman-cluster filtering.
        # Waarom: PCA op zwaar-gecorreleerde ruis produceert **orthogonale
        # ruis** (component 2 wordt gedomineerd door een kleine eigenwaarde
        # van near-duplicate feature-paren). We filteren eerst redundantie
        # weg door features te clusteren op hun Spearman-rank correlatie en
        # per cluster één representant te behouden (degene met de hoogste
        # variantie, dus de meest informatiedichte).
        #
        # spearman_cluster_threshold = 1 − ρ als `distance`.
        #   0.25 ⇒ clustert features met |ρ| ≥ 0.75 samen (agressief dedup).
        #   AUDIT-FIX (N20): per-asset waarde nu auto-bepaald (zie boven).
        self.spearman_prefilter:        bool  = spearman_prefilter
        self.spearman_cluster_threshold: float = float(spearman_cluster_threshold)

        # Interne state (gevuld door fit_transform)
        self._var_mask: np.ndarray | None    = None   # bool mask: niet-constante features
        self._dedup_mask: np.ndarray | None   = None   # bool mask na Spearman dedup
        self._scaler:   object | None         = None   # StandardScaler
        self._pca:      object | None         = None   # sklearn PCA
        self._output_names: list[str]            = []
        self._is_fitted: bool                    = False

    # ------------------------------------------------------------------
    @staticmethod
    def _spearman_dedup_mask(
        X_calib: np.ndarray,
        feature_names: list[str],
        distance_threshold: float,
    ) -> tuple[np.ndarray, list[str]]:
        """Hiërarchische cluster-dedup op Spearman rank correlatie.

        Bouwt de afstandsmatrix (1 − |ρ_Spearman|), clustert met complete
        linkage en kiest per cluster de feature met maximale std als
        representant (informatie-rijkste). Retourneert:

          * bool-mask (shape == n_features) — True voor te behouden features
          * nieuwe naamlijst voor de behouden features

        Fallback: als scipy niet beschikbaar is of de berekening numeriek
        faalt, retourneert een all-True mask (identity; no-op).
        """
        p = X_calib.shape[1]
        keep_all = np.ones(p, dtype=bool)

        if p <= 2 or not _SCIPY_AVAILABLE or spearmanr_func is None or linkage is None or fcluster is None:
            return keep_all, list(feature_names)

        try:
            rho_result = spearmanr_func(X_calib, axis=0, nan_policy="omit")
            # spearmanr retourneert een SpearmanrResult (namedtuple-like)
            # met .correlation en .pvalue attributen, of in oudere versies
            # een tuple. Gebruik getattr om beide te ondersteunen.
            rho = getattr(rho_result, "correlation", None)
            if rho is None:
                rho = rho_result[0]
            rho_arr = np.asarray(rho, dtype=np.float64)
            if rho_arr.ndim == 0:  # p == 2 edge-case
                rho_arr = np.array([[1.0, float(rho_arr)], [float(rho_arr), 1.0]])
            # NaN-safety: vervang door 0 (geen correlatie).
            rho_arr = np.nan_to_num(rho_arr, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(rho_arr, 1.0)
            dist = 1.0 - np.abs(rho_arr)
            # Zorg voor symmetrie + niet-negatieve afstanden (numeriek).
            dist = 0.5 * (dist + dist.T)
            dist = np.clip(dist, 0.0, 2.0)
            # squareform eist hol-diagonale afstand + 0 op de diagonaal.
            np.fill_diagonal(dist, 0.0)
            if squareform is None:
                return keep_all, list(feature_names)
            cond = squareform(dist, checks=False)
            Z = linkage(cond, method="complete")
            cluster_ids = fcluster(
                Z, t=distance_threshold, criterion="distance"
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Spearman-dedup faalde (%s) — PCA runt zonder pre-filter.", exc
            )
            return keep_all, list(feature_names)

        # Per cluster: kies feature met maximale std (meest informatief).
        stds = np.std(X_calib, axis=0)
        mask = np.zeros(p, dtype=bool)
        for c in np.unique(cluster_ids):
            members = np.where(cluster_ids == c)[0]
            if members.size == 1:
                mask[members[0]] = True
                continue
            best = members[int(np.argmax(stds[members]))]
            mask[best] = True

        if not mask.any():
            return keep_all, list(feature_names)

        kept_names = [nm for nm, keep in zip(feature_names, mask) if keep]
        logger.info(
            "Spearman dedup: %d → %d features (threshold=%.2f).",
            p, int(mask.sum()), distance_threshold,
        )
        return mask, kept_names

    # ------------------------------------------------------------------
    def fit_transform(
        self,
        X: np.ndarray,
        feature_names: list[str],
        prefix: str = "pc",
    ) -> tuple[np.ndarray, list[str]]:
        """Fit PCA op de kalibratie-subset en transformeer de volledige reeks.

        Stap 1: verwijder constante features (std ≈ 0 over kalibratie-subset).
        Stap 2: StandardScaler op kalibratie-subset (verplicht voor PCA).
        Stap 3: PCA tot ``n_components`` verklaarde variantie.
        Stap 4: transformeer de volledige reeks X met de gefitte PCA.

        Args:
            X            : (n_bars, n_feats) float32/float64 feature matrix.
            feature_names: lijst van feature-namen (len == n_feats).
            prefix       : naamprefix voor de PCA-componenten.

        Returns:
            Tuple (X_orth, new_names):
                X_orth     — (n_bars, n_components) float32, orthogonale matrix.
                new_names  — lijst van namen ["micro_pc0", "micro_pc1", ...].
        """
        # CHIEF AUDIT 2026-05-23 (P-7): fit_transform fits PCA on the first
        # ``calib_frac × n`` bars and then transforms the FULL dataset.
        # Inside a CPCV context this leaks distributional information from
        # test folds whose bars overlap the calibration slice.  Use
        # ``fit_on_train_indices(X, train_idx, ...)`` per fold to eliminate
        # this leakage.  The legacy path is preserved for backward-compat.
        warnings.warn(
            "FeatureOrthogonalizer.fit_transform fits on full (calibration) "
            "data — use fit_on_train_indices() in CPCV context to avoid "
            "distributional leakage.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not _SKLEARN_AVAILABLE:
            logger.warning(
                "FeatureOrthogonalizer: sklearn niet beschikbaar → no-op."
            )
            self._is_fitted = True
            self._output_names = list(feature_names)
            return X.astype(np.float32), list(feature_names)

        n, p = X.shape
        if p < 2:
            logger.info(
                "FeatureOrthogonalizer: te weinig features (%d) → no-op.", p
            )
            self._is_fitted = True
            self._output_names = list(feature_names)
            return X.astype(np.float32), list(feature_names)

        # ── Kalibratie-subset ─────────────────────────────────────────
        calib_end = max(
            min(int(n * self.calib_frac), self.max_calib_bars),
            self.min_components + 1,
            10,
        )
        X_calib = X[:calib_end].astype(np.float64)

        # ── Stap 1: verwijder constante features ──────────────────────
        std_calib: np.ndarray = np.std(X_calib, axis=0)
        var_mask: np.ndarray  = std_calib > 1e-9

        if not var_mask.any():
            logger.warning(
                "FeatureOrthogonalizer [%s]: alle features zijn constant → no-op.",
                prefix,
            )
            self._is_fitted = True
            self._output_names = list(feature_names)
            return X.astype(np.float32), list(feature_names)

        X_valid       = X[:, var_mask].astype(np.float64)
        X_calib_valid = X_calib[:, var_mask]
        names_valid   = [nm for nm, keep in zip(feature_names, var_mask) if keep]

        # ── Stap 1b: Spearman-rank dedup vóór PCA (voorkomt "orthogonale ruis") ──
        if self.spearman_prefilter:
            dedup_mask, names_valid = self._spearman_dedup_mask(
                X_calib_valid, names_valid, self.spearman_cluster_threshold
            )
        else:
            dedup_mask = np.ones(X_calib_valid.shape[1], dtype=bool)

        X_valid       = X_valid[:, dedup_mask]
        X_calib_valid = X_calib_valid[:, dedup_mask]

        # ── Stap 2: StandardScaler (fit op kalibratie-subset) ─────────
        if StandardScaler is None:  # pragma: no cover
            self._is_fitted = True
            self._output_names = list(feature_names)
            return X.astype(np.float32), list(feature_names)
        scaler = StandardScaler()
        scaler.fit(X_calib_valid)
        X_scaled       = scaler.transform(X_valid)
        X_calib_scaled = X_scaled[:calib_end]

        # ── Stap 3: PCA (fit op kalibratie-subset) ────────────────────
        # L3-FIX: Reorganiseer PCA-initialisatie zodat de dubbele fit alleen
        # in de edge-case optreedt en altijd expliciet gedocumenteerd is.
        # Achtergrond: de variantie-drempel (float zoals 0.95) selecteert pas
        # na de fit het werkelijke componentenaantal — er is geen manier om dit
        # vooraf exact te weten zonder te fitten. Als de drempel te weinig
        # componenten geeft (<min_components), is één extra fit onvermijdelijk.
        # In de normale situatie (voldoende variance) volstaat één fit.
        if PCA is None:  # pragma: no cover
            self._is_fitted = True
            self._output_names = list(feature_names)
            return X.astype(np.float32), list(feature_names)
        n_comp_request = min(
            max(self.min_components, 1),
            X_calib_valid.shape[1],
        )
        if X_calib_valid.shape[1] > self.min_components:
            pca = PCA(n_components=float(self.n_components))
        else:
            pca = PCA(n_components=n_comp_request)

        pca.fit(X_calib_scaled)

        # Edge-case: variantiedrempel selecteerde minder dan min_components.
        # Herfit met hard minimum (tweede fit is hier onvermijdelijk omdat
        # het werkelijke componentenaantal pas na de eerste fit bekend is).
        if pca.n_components_ < self.min_components:                 # type: ignore[attr-defined]
            n_hard = min(self.min_components, X_calib_valid.shape[1])
            pca = PCA(n_components=n_hard)
            pca.fit(X_calib_scaled)

        actual_comps = pca.n_components_                            # type: ignore[attr-defined]
        _ = actual_comps  # diagnostiek; gebruikt in log hieronder

        # ── Stap 4: transformeer volledige reeks ──────────────────────
        # H2-WAARSCHUWING: De PCA is hier gefit op de eerste calib_frac×n bars
        # en transformeert daarna de VOLLEDIGE reeks (100%).
        # Als CPCV test-folds vroege periodes bevatten die in de kalibratie-
        # subset zaten, zijn de PCA-features van die test-bars licht lekvrij:
        # de principal components zijn impliciet gecalibreerd op die test-data.
        # Gevolg: backtestprestaties licht overschat voor vroege test-folds.
        #
        # Volledige oplossing: gebruik fit_on_train_indices() per CPCV-fold
        # in de trainingsloop (zie methode hieronder).  Dit is architectureel
        # zwaarder maar elimineert de lekkage volledig.
        X_orth: np.ndarray = pca.transform(X_scaled).astype(np.float32)  # type: ignore[attr-defined]
        new_names = [f"{prefix}{i}" for i in range(X_orth.shape[1])]

        # Bewaar state voor productie-inferentie
        self._var_mask      = var_mask
        self._dedup_mask    = dedup_mask
        self._scaler        = scaler
        self._pca           = pca
        self._output_names  = new_names
        self._is_fitted     = True

        explained = float(
            sum(pca.explained_variance_ratio_)  # type: ignore[attr-defined]
        ) * 100.0
        logger.info(
            "FeatureOrthogonalizer [%s]: %d → %d componenten "
            "(%.1f%% variantie verklaard, calib=%d bars).",
            prefix,
            int(var_mask.sum()),
            X_orth.shape[1],
            explained,
            calib_end,
        )

        return X_orth, new_names

    # ------------------------------------------------------------------
    def fit_on_train_indices(
        self,
        X: np.ndarray,
        train_idx: np.ndarray,
        feature_names: list[str],
        prefix: str = "pc",
    ) -> tuple[np.ndarray, list[str]]:
        """H2-FIX: Fit PCA uitsluitend op de opgegeven train-indices en transformeer
        daarna de VOLLEDIGE matrix X.

        Dit is de correcte aanpak voor CPCV: per fold wordt de PCA gefit op de
        trainingsdata van díe fold, waarna de test-data van die fold wordt
        getransformeerd met de fold-specifieke PCA.  Hiermee is er geen lekkage
        van PCA-variantiestructuur naar test-bars.

        Typisch gebruik in een CPCV-loop::

            for fold_idx, (train_idx, val_idx, _) in enumerate(cv.split(...)):
                X_micro_fold, names = orth_micro.fit_on_train_indices(
                    X_micro_raw, train_idx, micro_names, prefix="micro_pc"
                )
                x1_tr_s = X_micro_fold[train_idx]
                x1_val_s = X_micro_fold[val_idx]

        Args:
            X           : volledige (n_bars, n_feats) feature-matrix.
            train_idx   : integer array met de bar-indices van de training-fold.
            feature_names: lijst van feature-namen (len == n_feats).
            prefix      : naamprefix voor de PCA-componenten.

        Returns:
            Tuple (X_orth_full, new_names):
                X_orth_full — (n_bars, n_components) float32, getransformeerde matrix.
                new_names   — lijst van namen ["{prefix}0", "{prefix}1", ...].
        """
        if not _SKLEARN_AVAILABLE:
            logger.warning("FeatureOrthogonalizer: sklearn niet beschikbaar -> no-op.")
            self._is_fitted = True
            self._output_names = list(feature_names)
            return X.astype(np.float32), list(feature_names)

        _, p = X.shape
        if p < 2 or len(train_idx) < 2:
            self._is_fitted = True
            self._output_names = list(feature_names)
            return X.astype(np.float32), list(feature_names)

        X_train = X[train_idx].astype(np.float64)

        # Stap 1: verwijder constante features over de train-subset
        std_train: np.ndarray = np.std(X_train, axis=0)
        var_mask: np.ndarray  = std_train > 1e-9

        if not var_mask.any():
            self._is_fitted = True
            self._output_names = list(feature_names)
            return X.astype(np.float32), list(feature_names)

        X_train_valid = X_train[:, var_mask]
        X_all_valid   = X[:, var_mask].astype(np.float64)
        names_valid = [nm for nm, keep in zip(feature_names, var_mask) if keep]

        # Stap 1b: Spearman rank dedup op de TRAIN-subset (geen lekkage).
        if self.spearman_prefilter:
            dedup_mask, names_valid = self._spearman_dedup_mask(
                X_train_valid, names_valid, self.spearman_cluster_threshold
            )
        else:
            dedup_mask = np.ones(X_train_valid.shape[1], dtype=bool)

        X_train_valid = X_train_valid[:, dedup_mask]
        X_all_valid   = X_all_valid[:, dedup_mask]

        # Stap 2: StandardScaler gefit op train-subset
        if StandardScaler is None or PCA is None:  # pragma: no cover
            self._is_fitted = True
            self._output_names = list(feature_names)
            return X.astype(np.float32), list(feature_names)
        scaler = StandardScaler()
        scaler.fit(X_train_valid)
        X_train_scaled = scaler.transform(X_train_valid)
        X_all_scaled   = scaler.transform(X_all_valid)

        # Stap 3: PCA gefit op train-subset (geen lekkage naar test-bars)
        n_comp_req = min(max(self.min_components, 1), X_train_valid.shape[1])
        if X_train_valid.shape[1] > self.min_components:
            pca = PCA(n_components=float(self.n_components))
        else:
            pca = PCA(n_components=n_comp_req)

        pca.fit(X_train_scaled)

        if pca.n_components_ < self.min_components:                 # type: ignore[attr-defined]
            n_hard = min(self.min_components, X_train_valid.shape[1])
            pca = PCA(n_components=n_hard)
            pca.fit(X_train_scaled)

        # Stap 4: transformeer ALLE bars (ook test-bars) met de fold-PCA
        X_orth_full: np.ndarray = pca.transform(X_all_scaled).astype(np.float32)  # type: ignore[attr-defined]
        new_names = [f"{prefix}{i}" for i in range(X_orth_full.shape[1])]

        # Sla state op voor productie-inferentie (overschrijft vorige fold-state)
        self._var_mask     = var_mask
        self._dedup_mask   = dedup_mask
        self._scaler       = scaler
        self._pca          = pca
        self._output_names = new_names
        self._is_fitted    = True

        explained = float(sum(pca.explained_variance_ratio_)) * 100.0  # type: ignore[attr-defined]
        logger.debug(
            "FeatureOrthogonalizer.fit_on_train_indices [%s]: "
            "%d train-bars → %d componenten (%.1f%% variantie).",
            prefix, len(train_idx), X_orth_full.shape[1], explained,
        )

        return X_orth_full, new_names

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Pas de gefitte PCA toe op nieuwe data (inference/productie).

        Args:
            X: (n_bars, n_feats) float array — dezelfde feature-volgorde
               als gebruikt bij fit_transform().

        Returns:
            (n_bars, n_components) float32 array.

        Raises:
            RuntimeError: als fit_transform() nog niet aangeroepen is.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "FeatureOrthogonalizer.transform() aangeroepen vóór fit_transform(). "
                "Roep eerst fit_transform() aan of laad een opgeslagen instantie."
            )

        if not _SKLEARN_AVAILABLE or self._pca is None or self._scaler is None:
            # No-op modus (sklearn niet beschikbaar of niet echt gefit)
            return X.astype(np.float32)

        if self._var_mask is None:
            return X.astype(np.float32)

        X_valid  = X[:, self._var_mask].astype(np.float64)
        if self._dedup_mask is not None and self._dedup_mask.shape[0] == X_valid.shape[1]:
            X_valid = X_valid[:, self._dedup_mask]
        X_scaled = self._scaler.transform(X_valid)            # type: ignore[union-attr]
        X_orth   = self._pca.transform(X_scaled)              # type: ignore[union-attr]
        return X_orth.astype(np.float32)

    # ------------------------------------------------------------------
    @property
    def output_names(self) -> list[str]:
        """Lijst van feature-namen na orthogonalisatie."""
        return self._output_names

    @property
    def is_fitted(self) -> bool:
        """True als fit_transform() succesvol aangeroepen is."""
        return self._is_fitted

    def __repr__(self) -> str:
        n_out = len(self._output_names) if self._output_names else 0
        return (
            f"FeatureOrthogonalizer("
            f"n_components={self.n_components}, "
            f"fitted={self._is_fitted}, "
            f"output_dims={n_out})"
        )
