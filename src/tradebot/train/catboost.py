# src/tradebot/train/catboost.py
"""CatBoost agents for regime-based binary classification.

Migrated from agent.py.  Online scaling lives in _scalers.py to keep
each file under the 800-LOC CI gate.
"""
from __future__ import annotations

import collections
import logging
from pathlib import Path
from typing import Any, cast

import catboost as cb
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

# Phase 0: scikit-learn is gepind op >=1.6, waarin sklearn.frozen bestaat. De
# try/except zette _FrozenEstimator op None, waarna de Platt-kalibratie
# stilzwijgend op een NIET-bevroren estimator werd gefit - een data-lek richting
# de kalibratieset.
from sklearn.frozen import FrozenEstimator as _FrozenEstimator

# Phase 0 stap 5: dit bestand importeerde acht symbolen uit het niet-bestaande
# pakket `quant_architect`. De except-tak verving elk symbool door `Any`. Meting
# wijst uit dat GEEN van die symbolen in dit bestand ooit werd gebruikt: het
# importblok was volledig dode code die alleen de illusie van een
# blueprint-integratie wekte. De symbolen leven in tradebot.train.{reward,
# schema_guard,stack,thompson} en worden daar geimporteerd waar ze echt nodig
# zijn (zie train/ensemble.py).
from ._scalers import RollingRobustScaler, get_git_hash, numba_rolling_robust_scale  # noqa: F401

logger = logging.getLogger(__name__)

class RegimeCatAgent:
    """Wrapper rondom CatBoost voor binaire classificatie op marktregimes."""

    # AUDIT-FIX (N19 — model version metadata):
    #   Versie "2.0" = K3-fix actief (use_pca_for_catboost toggleable).
    #   Opgeslagen modellen van vóór K3-fix hebben feature_schema="pca" maar
    #   nieuwe inference verwacht "raw" → stille slechte voorspellingen.
    #   validate_schema() blokkeert mismatches vóór predict().
    ARTIFACT_VERSION: str = "2.0"

    def __init__(
        self,
        cfg: dict,
        model_params: dict | None = None,
        model_version: str = "cat_binary_v4",
        symbol: str | None = None,
        feature_schema: str = "raw",
        # AUDIT-FIX (N19): "raw" = K3-fix (CatBoost op ruwe features),
        # "pca" = pre-K3 gedrag (CatBoost op PCA-componenten).
        # Sla ALTIJD op bij training; check bij laden.
        feature_names_trained: list[str] | None = None,
        # De exacte feature-namen in training-volgorde.  Wordt gecontroleerd
        # bij inference om stille feature-name-mismatch te detecteren.
        # CHIEF AUDIT 2026-05-23 (P-7): expliciete side ("LONG"/"SHORT").
        # Wordt door ContextualBanditEnsemble doorgegeven aan
        # counterfactual-reward logica. ``None`` = legacy detectie via
        # min_conf_long-threshold (deprecated).
        side: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.model_version = model_version
        self.git_hash = get_git_hash()
        self.is_trained = False
        self.symbol = symbol
        self.side: str | None = side

        # N19: model schema metadata
        self.feature_schema: str = feature_schema
        self.feature_names_trained: list[str] | None = feature_names_trained
        self.artifact_version: str = self.ARTIFACT_VERSION

        self.min_conf_long = 0.35
        self.min_conf_short = 0.35

        self.scaler_micro: RollingRobustScaler | None = None
        self.scaler_meso: RollingRobustScaler | None = None
        self.scaler_macro: RollingRobustScaler | None = None
        self.feature_order: dict[str, list[str]] | None = None

        if self.cfg.get('machine', {}).get('saved_models_dir'):
            self._load_resources(Path(self.cfg['machine']['saved_models_dir']))

        default_params = {
            'iterations': 1000, 'learning_rate': 0.01, 'depth': 6,
            'l2_leaf_reg': 5.0, 'random_strength': 1.0, 'bagging_temperature': 0.8,
            'rsm': 0.5, 'od_type': 'Iter', 'od_wait': 50, 'border_count': 128,
            'loss_function': 'Logloss', 'eval_metric': 'Logloss', 'task_type': 'CPU',
            'allow_writing_files': False,
            # L4-FIX: meta-labeling geeft typisch ~30-40% wins (1) vs ~60-70% losses (0).
            # Zonder class-gewichten voorspelt CatBoost altijd "0" (hoge accuracy maar
            # nul alpha). 'Balanced' herschaalt zodat elke klasse gelijk bijdraagt aan
            # de Logloss-gradiënt: class_w_k = (n / n_classes) / n_k.
            'auto_class_weights': 'Balanced',
        }

        if model_params:
            clean_params = model_params.copy()
            self.min_conf_long = clean_params.pop('min_conf_long', 0.35)
            self.min_conf_short = clean_params.pop('min_conf_short', 0.35)
            for key in ['min_conf', 'min_confidence', 'horizon']:
                clean_params.pop(key, None)

            default_params.update(clean_params)

        self.params = default_params
        self.model: cb.CatBoostClassifier | CalibratedClassifierCV | None = None

    def _load_resources(self, base_dir: Path) -> None:
        suffix = f"_{self.symbol}" if self.symbol else ""
        fo_path = base_dir / f"feature_order{suffix}.pkl"

        if fo_path.exists():
            self.feature_order = joblib.load(fo_path)

        def load_scaler(name: str) -> RollingRobustScaler | None:
            prod_path = base_dir / f"{name}{suffix}_prod.pkl"
            std_path = base_dir / f"{name}{suffix}.pkl"
            path_to_use = prod_path if prod_path.exists() else (std_path if std_path.exists() else None)

            if not path_to_use:
                return None

            obj = joblib.load(path_to_use)
            result = obj['scaler'] if isinstance(obj, dict) and 'scaler' in obj else obj
            return cast(RollingRobustScaler | None, result)

        self.scaler_micro = load_scaler("scaler_micro")
        self.scaler_meso = load_scaler("scaler_meso")
        self.scaler_macro = load_scaler("scaler_macro")

    def stamp_schema_into_stack(self, stack: Any) -> None:
        """BLUEPRINT-FIX (V.8): druk de getrainde schema-fingerprint in de stack.

        De orchestrator (Cowork live-loop / portfolio_backtest) roept dit aan
        direct na ``RegimeCatAgent`` van disk te hebben geladen. Daarna kan
        :meth:`FeatureSchemaGuard.check` per bar een hash-vergelijking doen
        tegen de live-pipeline; bij mismatch raised het een
        :class:`SchemaMismatchError` zodat trading kan worden gepauzeerd.

        Args:
            stack: ``QuantArchitectStack`` (of compat) met ``schema_guard``-attr.

        Returns:
            None. No-op wanneer geen feature_names_trained beschikbaar is of
            als de stack geen ``schema_guard`` heeft.
        """
        if self.feature_names_trained is None or not self.feature_names_trained:
            logger.debug(
                "stamp_schema_into_stack: feature_names_trained ontbreekt — "
                "schema-guard niet gestempeld voor [%s].",
                self.symbol,
            )
            return
        guard = getattr(stack, "schema_guard", None)
        if guard is None:
            return
        fp = guard.stamp(self.feature_names_trained)
        logger.info(
            "FeatureSchemaGuard gestempeld voor [%s]: sha=%s… (%d features).",
            self.symbol, str(fp.sha256)[:16], int(fp.feature_count),
        )

    def validate_schema(self, runtime_feature_schema: str, runtime_feature_names: list[str] | None = None) -> None:
        """Valideer compatibiliteit van opgeslagen model-schema met huidige config.

        AUDIT-FIX (N19): roep aan vóór elke predict()-aanroep nadat een model
        van disk is geladen.  Blokkeert stille PCA-vs-raw mismatches die
        optreden wanneer een pre-K3 model (schema="pca") wordt geladen terwijl
        use_pca_for_catboost=False actief is.

        Args:
            runtime_feature_schema: "raw" of "pca" op basis van huidige config.
            runtime_feature_names: feature-namen die de huidige pipeline levert.

        Raises:
            ValueError: bij schema-mismatch of feature-naam-mismatch.
        """
        if self.feature_schema != runtime_feature_schema:
            raise ValueError(
                f"N19 ModelSchemaMismatch [{self.symbol}]: "
                f"model opgeslagen met feature_schema='{self.feature_schema}' "
                f"maar runtime verwacht '{runtime_feature_schema}'. "
                f"Hertrain het model met de huidige config of zet "
                f"use_pca_for_catboost={'True' if runtime_feature_schema == 'pca' else 'False'}."
            )
        if (
            runtime_feature_names is not None
            and self.feature_names_trained is not None
            and runtime_feature_names != self.feature_names_trained
        ):
            missing = set(self.feature_names_trained) - set(runtime_feature_names)
            extra   = set(runtime_feature_names) - set(self.feature_names_trained)
            raise ValueError(
                f"N19 FeatureNameMismatch [{self.symbol}]: "
                f"model verwacht {len(self.feature_names_trained)} features; "
                f"runtime levert {len(runtime_feature_names)}. "
                f"Ontbrekend: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}. "
                f"Extra: {sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}. "
                f"Hertrain of update feature_pipeline config."
            )

    def _align_input_custom(self, x_raw: np.ndarray, target_features: list[str], timeframe: str) -> np.ndarray:
        if x_raw is None or x_raw.size == 0:
            n_rows = x_raw.shape[0] if (x_raw is not None and x_raw.ndim >= 1) else 1
            return np.zeros((n_rows, len(target_features) if target_features else 0), dtype=np.float32)

        if not target_features:
            return np.zeros((x_raw.shape[0], 0), dtype=np.float32)

        if x_raw.shape[1] == len(target_features):
            # M2-FIX: Shape-overeenkomst garandeert NIET dat de volgorde overeenkomt.
            # Als twee pipeline-versies toevallig even veel features produceren maar
            # in een andere volgorde, geeft het model stille, foute voorspellingen.
            # De aanroeper (prepare_inference_data) herordent via target_features-indices;
            # een debug-log hier maakt toekomstige pipeline-updates traceerbaarder.
            logger.debug(
                "_align_input_custom [%s]: %d features — shape overeenkomst, "
                "volgorde ongecontroleerd. Controleer feature_order na modelupdates.",
                timeframe, x_raw.shape[1],
            )
            return x_raw

        # M3-FIX: RuntimeError in plaats van stille nul-matrix.
        # All-zeros invoer geeft altijd een specifieke kans terug (afhankelijk van het
        # trainingsgemiddelde bij zeros) zonder dat er een alarm klinkt — onacceptabel
        # in productie. Een harde fout maakt het probleem onmiddellijk zichtbaar.
        raise RuntimeError(
            f"Shape mismatch in [{timeframe}]: verwacht {len(target_features)} features, "
            f"ontvangen {x_raw.shape[1]}. Controleer de feature-pipeline na een modelupdate."
        )

    def prepare_inference_data(
        self,
        x_micro: Any,
        x_meso: Any,
        x_macro: Any,
        base_feature_map: dict | None = None,
        is_pre_scaled: bool = False
    ) -> np.ndarray:

        def process_part(x: Any, timeframe: str, scaler: RollingRobustScaler | None) -> np.ndarray | None:
            if x is None:
                return None

            x_arr = np.asarray(x.values if hasattr(x, 'values') else x)
            if x_arr.ndim == 3:
                x_arr = x_arr[:, -1, :]

            target_features = base_feature_map.get(timeframe, []) if base_feature_map else (
                self.feature_order.get(timeframe, []) if self.feature_order else []
            )

            if not target_features:
                return np.zeros((x_arr.shape[0], 0), dtype=np.float32)

            x_aligned = self._align_input_custom(x_arr, target_features, timeframe)

            if x_aligned is None or x_aligned.size == 0:
                return np.zeros((x_arr.shape[0], 0), dtype=np.float32)

            if scaler is not None and not is_pre_scaled:
                x_aligned = scaler.transform(x_aligned)

            local_features = self.feature_order.get(timeframe, []) if self.feature_order else []
            if not local_features:
                return x_aligned

            indices = [target_features.index(f) for f in local_features if f in target_features]
            if not indices:
                return np.zeros((x_aligned.shape[0], 0), dtype=np.float32)

            return x_aligned[:, indices]

        p_micro = process_part(x_micro, "micro", self.scaler_micro)
        p_meso = process_part(x_meso, "meso", self.scaler_meso)
        p_macro = process_part(x_macro, "macro", self.scaler_macro)

        parts = [p for p in [p_micro, p_meso, p_macro] if p is not None and p.ndim == 2 and p.shape[1] > 0]
        n_rows = x_micro.shape[0] if x_micro is not None else 1

        return np.hstack(parts) if parts else np.zeros((n_rows, 0))

    def predict_binary(
        self,
        x_micro: Any,
        x_meso: Any = None,
        x_macro: Any = None,
        base_feature_map: dict | None = None,
        is_pre_scaled: bool = False
    ) -> dict[str, float]:

        if self.model is None:
            logger.warning("predict_binary aangeroepen maar model is None.")
            return {"prob_win": 0.0, "prob_loss": 1.0}

        X = self.prepare_inference_data(x_micro, x_meso, x_macro, base_feature_map, is_pre_scaled)
        if X is None or X.size == 0 or X.shape[1] == 0:
            return {"prob_win": 0.0, "prob_loss": 1.0}

        probs = self.model.predict_proba(X)[0]
        # M4-FIX: Als CatBoost slechts één klasse heeft gezien (edge-case bij
        # extreme klasse-onbalans in een CPCV-fold), retourneert predict_proba
        # één waarde. De oude code zette prob_loss = 0.0, waardoor de kansen
        # niet optelden tot 1.0. Correcte afhandeling: prob_loss = 1 - prob_win.
        if len(probs) == 1:
            prob_win  = float(probs[0])
            prob_loss = 1.0 - prob_win
        else:
            prob_win  = float(probs[1])
            prob_loss = float(probs[0])

        return {"prob_win": prob_win, "prob_loss": prob_loss}

    def train(
        self,
        x_micro: Any,
        y: Any,
        x_meso: Any = None,
        x_macro: Any = None,
        eval_set: Any = None,
        sample_weight: Any = None,
        is_pre_scaled: bool = False,
        calib_x_micro: Any = None,
        calib_x_meso: Any = None,
        calib_x_macro: Any = None,
        calib_y: Any = None,
        calib_w: Any = None
    ) -> RegimeCatAgent:

        if np.unique(y).size < 2:
            return self

        def _get_x(mic: Any, mes: Any, mac: Any) -> np.ndarray:
            if is_pre_scaled:
                parts = [np.asarray(p) for p in (mic, mes, mac) if p is not None and len(p) > 0]
                parts = [p.reshape(-1, 1) if p.ndim == 1 else p for p in parts]
                return np.hstack(parts) if parts else np.zeros((len(mic), 0))
            return self.prepare_inference_data(mic, mes, mac)

        X = _get_x(x_micro, x_meso, x_macro)
        # DTYPE-FIX: cast naar float32 vlak vóór cb.Pool. De rest van de
        # pijplijn (Bandit, Kelly, Platt) werkt op float64; CatBoost zelf
        # traint sneller op float32. copy=False: no-op als al float32.
        train_pool = cb.Pool(X.astype(np.float32, copy=False), label=y, weight=sample_weight)

        # PLATT-FALLBACK-FIX (Item 2): de fallback in het calib-pad gebruikte
        # voorheen een base_model met `auto_class_weights='Balanced'`. Een
        # Balanced model spuugt kansen uit rond 0.50 — een gecalibreerd model
        # rond de ware prior. Wisselt de fold tussen feasible en fallback,
        # dan wisselt de schaal van prob_win tussen folds en breekt de vaste
        # drempel `min_conf`. Wanneer er calib-data is geleverd, willen we
        # ALTIJD op de natuurlijke (scheve) verdeling trainen zodat:
        #   - feasible pad: Platt herijkt naar de ware prior;
        #   - fallback pad: kansen liggen al op de prior (geen Platt nodig).
        # Het no-calib pad (else hieronder) houdt 'Balanced' aan zoals voorheen.
        params_no_balanced = {
            k: v for k, v in self.params.items() if k != "auto_class_weights"
        }
        base_model = cb.CatBoostClassifier(**self.params)

        if calib_x_micro is not None and calib_y is not None and np.unique(calib_y).size >= 2:
            cX = _get_x(calib_x_micro, calib_x_meso, calib_x_macro)
            calib_y_arr = np.asarray(calib_y)

            # LEAKAGE-FIX (3-way split): Gebruik NOOIT dezelfde set voor zowel
            # early stopping als Platt Scaling.
            # Probleem: Als CatBoost stopt op basis van de error op set X, heeft
            # het impliciet de boomdiepte en het iteratieaantal gekozen om die
            # specifieke set te passen. De sigmoid-fitter ziet dan een te optimistische
            # kansverdeling — de kalibratie wordt overmoedig (te dicht bij 0/1).
            # Fix: Splits de calib-data chronologisch:
            #   • Eerste 70% → early stopping (ES)
            #   • Laatste 30% → Platt scaling (nooit gezien door CatBoost)
            n_calib = len(calib_y_arr)
            n_es = max(10, int(n_calib * 0.70))

            cX_es   = cX[:n_es]
            cy_es   = calib_y_arr[:n_es]
            cw_es   = (np.asarray(calib_w)[:n_es] if calib_w is not None else None)

            cX_platt  = cX[n_es:]
            cy_platt  = calib_y_arr[n_es:]

            # Minimumcheck: sigmoid-fit vereist ≥2 klassen en voldoende samples
            platt_feasible = (
                len(cy_platt) >= 10
                and np.unique(cy_platt).size >= 2
                and np.unique(cy_es).size >= 2
            )

            if platt_feasible:
                # CLASS-WEIGHTS-FIX: auto_class_weights='Balanced' dwingt CatBoost om
                # vaker 'win' te voorspellen (herschaling van gradiënten). Direct daarna
                # probeert Platt Scaling de output terug te mappen naar ware kansen —
                # maar Platt ziet dan al een kunstmatig gebalanceerde distributie, niet
                # de echte scheve verdeling. De twee methodes werken tegen elkaar.
                # Oplossing: laat het model de WARE (scheve) distributie leren wanneer
                # Platt beschikbaar is. Platt herkalib. dan naar de echte klassen-ratio.
                # min_conf compenseert de lagere gemiddelde prob_win (zie conf_config).
                base_model = cb.CatBoostClassifier(**params_no_balanced)
                # DTYPE-FIX: float32 alleen bij cb.Pool; Platt (sklearn) krijgt float64.
                eval_pool = cb.Pool(cX_es.astype(np.float32, copy=False), label=cy_es, weight=cw_es)
                base_model.fit(
                    train_pool, eval_set=eval_pool,
                    early_stopping_rounds=50, verbose=False, use_best_model=True,
                )
                _est = _FrozenEstimator(base_model) if _FrozenEstimator is not None else cast(Any, base_model)
                calibrator = CalibratedClassifierCV(
                    estimator=_est, cv="prefit", method="sigmoid"
                )
                # Geen sample_weight: kalibratie moet de NATUURLIJKE klassen-
                # distributie zien, niet de gebalanceerde trainingsgewichten.
                calibrator.fit(cX_platt, cy_platt)
                self.model = calibrator
                logger.info(
                    "Platt scaling succesvol toegepast — kansen gecalibreerd "
                    "op afgescheiden Platt-set (%d samples, %.1f%% wins). "
                    "ES-set: %d samples.",
                    len(cy_platt),
                    float(np.mean(cy_platt)) * 100.0,
                    len(cy_es),
                )
            else:
                # Niet genoeg samples voor een schone 3-way split.
                # Val terug op early stopping op de volledige calib-set zonder Platt.
                # Liever geen kalibratie dan een kalibratie op te weinig data.
                #
                # PLATT-FALLBACK-FIX (Item 2): train óók in het fallback-pad
                # zonder 'Balanced' zodat de uitgespuugde kansen op de ware
                # prior-schaal liggen — identiek aan het feasible-pad. Anders
                # produceert deze fold prob_win rond 0.50, wat de vaste drempel
                # (min_conf=0.35) op bijna alles laat triggeren en de CPCV
                # pad-reconstructie destabiliseert.
                logger.warning(
                    "Calib-set te klein voor 3-way split (n_calib=%d, n_platt=%d) — "
                    "early stopping op volledige calib-set zonder Platt; trainen "
                    "op natuurlijke prior (geen 'Balanced').",
                    n_calib, len(cy_platt),
                )
                cw_full = np.asarray(calib_w) if calib_w is not None else None
                # DTYPE-FIX: float32 alleen bij cb.Pool.
                eval_pool = cb.Pool(cX.astype(np.float32, copy=False), label=calib_y_arr, weight=cw_full)
                fallback_model = cb.CatBoostClassifier(**params_no_balanced)
                fallback_model.fit(
                    train_pool, eval_set=eval_pool,
                    early_stopping_rounds=50, verbose=False, use_best_model=True,
                )
                self.model = fallback_model
        else:
            base_model.fit(train_pool, verbose=False)
            self.model = base_model

        self.is_trained = True
        return self


class RegimeEnsembleCat:
    """Simpele ensembling wrapper over meerdere getrainde modellen."""

    def __init__(
        self,
        models: list[RegimeCatAgent],
        meta_feature_map: dict | None = None,
        meta_scalers: dict | None = None,
        prob_window: int = 500,
        dynamic_threshold_pct: float = 80.0,
    ):
        self.models = models
        self.meta_feature_map = meta_feature_map
        self.meta_scalers = meta_scalers
        # Rolling percentiel-drempel (THRESHOLD-FIX): track de verdeling van prob_win
        # over een rollend venster. Drempel = Xde percentiel → model handelt altijd
        # slechts de sterkste (100-X)% van de signalen, ongeacht regime.
        # Statische drempel (bijv. 0.35) slaagt in Regime A maar filtert bijna alles
        # in Regime B waar de gemiddelde prob_win verschuift.
        self._prob_history: collections.deque[float] = collections.deque(maxlen=prob_window)
        self._dynamic_threshold_pct = dynamic_threshold_pct

    def update_live_scalers(self, x_micro: Any, x_meso: Any = None, x_macro: Any = None) -> None:
        if not self.models or not self.meta_scalers or not self.meta_feature_map:
            return

        ref_model = self.models[0]
        for tf, raw_data in [("micro", x_micro), ("meso", x_meso), ("macro", x_macro)]:
            scaler = self.meta_scalers.get(tf)
            if raw_data is not None and scaler is not None and hasattr(scaler, 'update'):
                target_feats = self.meta_feature_map.get(tf, [])
                raw_arr = np.asarray(raw_data)

                if raw_arr.ndim == 1:
                    raw_arr = raw_arr.reshape(1, -1)

                aligned = ref_model._align_input_custom(raw_arr, target_feats, tf)
                if aligned is not None and aligned.size > 0:
                    scaler.update(aligned)

    def recalibrate_all_scalers(self) -> None:
        if not self.meta_scalers:
            return

        for tf in ["micro", "meso", "macro"]:
            scaler = self.meta_scalers.get(tf)
            if scaler is not None and hasattr(scaler, 'recalibrate'):
                scaler.recalibrate()

    def predict_greybox_strategy(self, x_micro: Any, x_meso: Any, x_macro: Any, is_pre_scaled: bool = False, **kwargs) -> dict[str, Any]:
        # SHARED-SCALER-FIX (Item 10): central scaling pre-pass. Zie
        # ContextualBanditEnsemble._pre_scale_once voor uitleg. Wanneer de
        # caller raw data aanlevert, scalen we hier centraal en geven we
        # is_pre_scaled=True door aan elke agent.
        if (
            not is_pre_scaled
            and self.meta_scalers is not None
            and self.meta_feature_map is not None
            and self.models
        ):
            ref_model = self.models[0]

            def _scale(raw: Any, tf: str) -> Any:
                if raw is None:
                    return None
                arr = np.asarray(raw.values if hasattr(raw, "values") else raw)
                if arr.size == 0:
                    return raw
                if arr.ndim == 3:
                    arr = arr[:, -1, :]
                elif arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                target_feats = self.meta_feature_map.get(tf, []) if self.meta_feature_map else []
                if not target_feats:
                    return arr
                # Phase 0: `except RuntimeError: return raw` gaf de
                # ONUITGELIJNDE, ONGESCHAALDE feature-matrix terug. Het model
                # kreeg dan kolommen in een andere volgorde/schaal dan waarop het
                # is getraind - de klassieke stille schema-drift.
                aligned = ref_model._align_input_custom(arr, target_feats, tf)
                if aligned is None or aligned.size == 0:
                    return aligned
                scaler = self.meta_scalers.get(tf) if self.meta_scalers else None
                if scaler is not None and hasattr(scaler, "transform"):
                    aligned = scaler.transform(aligned)
                return aligned

            x_micro_s = _scale(x_micro, "micro")
            x_meso_s = _scale(x_meso, "meso")
            x_macro_s = _scale(x_macro, "macro")
            agent_pre_scaled = True
        else:
            x_micro_s, x_meso_s, x_macro_s = x_micro, x_meso, x_macro
            agent_pre_scaled = is_pre_scaled

        win_probs = np.array([
            m.predict_binary(
                x_micro_s, x_meso_s, x_macro_s,
                base_feature_map=self.meta_feature_map,
                is_pre_scaled=agent_pre_scaled,
            )["prob_win"]
            for m in self.models
        ])

        avg_win_prob = float(np.mean(win_probs))
        uncertainty = float(np.std(win_probs))

        ref_model = self.models[0]
        # SIDE-FIX: min_conf_long < 1.0 → Long-model; == 1.0 → Short-model.
        is_long_model = ref_model.min_conf_long < 1.0

        # THRESHOLD-FIX: rollende percentiel-drempel ipv statisch absoluut getal.
        # Bereken drempel op history VOOR append (geen lookahead in eigen distributie).
        floor_threshold = float(kwargs.get('min_confidence', 0.33))
        if len(self._prob_history) >= 50:
            dynamic_threshold = float(np.percentile(list(self._prob_history), self._dynamic_threshold_pct))
            effective_threshold = max(floor_threshold, dynamic_threshold)
        else:
            effective_threshold = floor_threshold  # warm-up periode: val terug op vloer

        # Voeg huidige prob toe NADAT drempel bepaald is
        self._prob_history.append(avg_win_prob)

        max_uncertainty = float(kwargs.get('max_uncertainty', 0.15))
        is_base_signal = (avg_win_prob >= effective_threshold) and (uncertainty <= max_uncertainty)

        return {
            "signal": 1 if is_base_signal else 0,
            "prob_win": avg_win_prob,
            "uncertainty": uncertainty,
            "threshold_used": effective_threshold,
            "side_detected": "LONG" if is_long_model else "SHORT",
        }


class _RewardClipper:
    """Stub placeholder — implementation pending."""

    pass
