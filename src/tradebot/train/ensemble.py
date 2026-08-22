# src/tradebot/train/ensemble.py
"""Contextual Bandit Ensemble for dynamic model-weight allocation.

Migrated from agent.py (root-level legacy).

Classes
-------
_RewardClipper
    Per-arm EMA-based Huber-equivalent reward clipping (±k·sigma).
_RandomFourierFeatures
    RFF approximation of RBF kernel (Rahimi & Recht 2007) for KernelUCB.
ContextualBanditEnsemble
    LinUCB / Thompson Sampling ensemble with:
    - Sherman-Morrison online inverse updates
    - Cholesky recompute every N steps (numerical stability)
    - Vol-aware discount factor gamma
    - Delayed feedback proxy (intraday MTM)
    - Asymmetric counterfactual rewards (long vs. short Triple Barrier)
    - EntropyGate + LedoitWolfThompson + NetAlphaReward blueprint stack
"""
from __future__ import annotations

import collections
import logging
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

# Chief-Architect blueprint integratie (soft-import).
try:
    from quant_architect import (
        EntropyGate,
        EntropyGateDecision,
        FeatureSchemaGuard,
        LedoitWolfThompsonSampler,
        NetAlphaResult,
        NetAlphaReward,
        QuantArchitectStack,
        SchemaMismatchError,
    )
    _QUANT_ARCHITECT_AVAILABLE: bool = True
except ImportError:
    _QUANT_ARCHITECT_AVAILABLE = False
    EntropyGate = Any  # type: ignore[assignment, misc]
    EntropyGateDecision = Any  # type: ignore[assignment, misc]
    FeatureSchemaGuard = Any  # type: ignore[assignment, misc]
    LedoitWolfThompsonSampler = Any  # type: ignore[assignment, misc]
    NetAlphaResult = Any  # type: ignore[assignment, misc]
    NetAlphaReward = Any  # type: ignore[assignment, misc]
    QuantArchitectStack = Any  # type: ignore[assignment, misc]

    class SchemaMismatchError(RuntimeError):  # type: ignore[no-redef]
        pass

logger = logging.getLogger(__name__)

# Type alias for RegimeCatAgent — avoid circular import by using Any at runtime.
from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    from .catboost import RegimeCatAgent  # noqa: F401
from ._bandit_helpers import _RandomFourierFeatures, _RewardClipper  # noqa: E402

# LOC-EXCEPTION: ContextualBanditEnsemble is a tightly-coupled LinUCB algorithm.
# Splitting it would destroy its mathematical cohesion. CI exemption in Makefile.

class ContextualBanditEnsemble:
    """Ensemble mechanisme dat zich baseert op Contextual Bandits (LinUCB iteraties) voor model weging.

    CHIEF AUDIT 2026-05-23 (P-3): roep ``reset_proxy_state()`` aan direct
    na ``joblib.load()`` of bij de overgang naar een nieuwe periode/run.
    Zonder reset blijven oude proxy-reward EMA's (_proxy_r_mu/var) en de
    cumulatieve _y_proxy_shift staan en worden ze als prior gebruikt bij
    cold-start — dit veroorzaakt 5-15% prob-distortion in de eerste ~100
    bars van een nieuwe periode.
    """

    def __init__(
        self,
        models: list[Any],
        meta_feature_map: dict | None = None,
        meta_scalers: dict | None = None,
        context_dim: int = 0,
        gamma: float = 0.995,
        v: float = 1.0,
        reset_interval: int = 10000,
        prob_window: int = 500,
        dynamic_threshold_pct: float = 80.0,
        # --- KernelUCB / RFF ---
        kernel: Literal["linear", "rbf"] = "linear",
        rff_dim: int = 128,
        rff_sigma: float = 1.0,
        rff_seed: int = 0,
        # --- Numerieke stabiliteit ---
        cholesky_interval: int = 1000,
        # --- Dynamische γ (vol-aware forgetting) ---
        gamma_min: float = 0.95,
        gamma_max: float = 0.999,
        # --- Exploratie ---
        selection: Literal["thompson", "ucb"] = "thompson",
        ucb_alpha: float = 1.0,
        # --- Asymmetrische counterfactuals ---
        slippage_bps: float = 0.0,
        # --- Reward clipping (Huber-equivalent) ---
        huber_clip_sigma: float = 3.0,
        huber_alpha: float = 0.01,
        # --- Delayed feedback (intraday MTM proxy) ---
        proxy_decay: float = 0.1,
        # --- Macro context PCA (Item 4) ---
        orthogonalizer_macro: Any | None = None,
        # PCA-BANDIT-FIX (Item 4): True = bandit zelf PCA-transformeert de
        # macro-context voordat hij in B/B⁻¹ wordt gestopt. Default False
        # omdat de standaard-pipeline (FeaturePipeline.use_pca_orth=True)
        # al een PCA op de macro-laag toepast — een tweede PCA hier zou
        # dubbelwerk en numeriek-instabiel zijn. Zet op True wanneer
        # ``use_pca_orth=False`` upstream draait, zodat de bandit nog steeds
        # decorrelated context ziet.
        apply_macro_orth: bool = False,
        # ── BLUEPRINT-FIX (Chief-Architect III/V) ────────────────────────
        # ``stack`` injecteert de blueprint-componenten (LedoitWolfThompson,
        # NetAlphaReward, EntropyGate, FeatureSchemaGuard, KalmanImpactObserver)
        # in één keer. ``None`` ⇒ pre-blueprint gedrag (legacy MVN-sampling,
        # geen entropy-gate, geen schema-guard). De orchestrator hoort één
        # gedeelde instantie door te geven zodat eta_observer-state en
        # schema-fingerprint cross-component consistent zijn.
        stack: Any | None = None,
        # CHIEF AUDIT 2026-05-23 (P-7): expliciete side-attribute. Eerder
        # werd ``_is_long_ens`` afgeleid uit ``self.models[0].min_conf_long
        # < 1.0`` — fragile threshold-trick die breekt zodra long en short
        # ensembles dezelfde min_conf-config delen, of bij models waar
        # min_conf_long niet bestaat. ``side`` is "LONG" of "SHORT" en
        # heeft voorrang als gezet. ``None`` (default) = legacy detection
        # met deprecation warning.
        side: Literal["LONG", "SHORT"] | None = None,
    ) -> None:
        self.models: list[Any] = models
        self.n_arms: int = len(models) if models else 1
        # ── BLUEPRINT-FIX (III.5 / V.9): bewaar stack-pointer en
        # afgeleide ``Optional`` references zodat hot-path methods geen
        # `getattr` overhead nodig hebben. Bij ``stack=None`` blijven
        # alle references ``None`` en valt de bandit terug op het oude
        # MVN-pad — exact identiek aan pre-blueprint gedrag.
        self.stack: Any | None = stack
        self._lwts: LedoitWolfThompsonSampler | None = (
            stack.lw_thompson if stack is not None else None
        )
        self._entropy_gate: EntropyGate | None = (
            stack.entropy_gate if stack is not None else None
        )
        self._net_alpha: NetAlphaReward | None = (
            stack.net_alpha if stack is not None else None
        )
        self._schema_guard: FeatureSchemaGuard | None = (
            stack.schema_guard if stack is not None else None
        )
        # Rolling percentiel-drempel — zie RegimeEnsembleCat voor uitleg.
        self._prob_history: collections.deque[float] = collections.deque(maxlen=prob_window)
        self._dynamic_threshold_pct: float = float(dynamic_threshold_pct)
        self.meta_feature_map: dict | None = meta_feature_map
        self.meta_scalers: dict | None = meta_scalers

        # CHIEF AUDIT 2026-05-23 (P-7): expliciete side voor counterfactual-
        # reward toewijzing. Heeft voorrang op de legacy min_conf_long-trick.
        self.side: Literal["LONG", "SHORT"] | None = side

        self.context_dim: int = int(context_dim)
        self.gamma: float = float(gamma)
        self.v: float = float(v)
        self.reset_interval: int = int(reset_interval)
        self.trade_count: int = 0

        # KernelUCB / RFF state
        self.kernel: Literal["linear", "rbf"] = kernel
        self.rff_dim: int = int(rff_dim)
        self.rff_sigma: float = float(rff_sigma)
        self.rff_seed: int = int(rff_seed)
        self._rff: _RandomFourierFeatures | None = None

        # Cholesky exact-recompute interval
        self.cholesky_interval: int = max(0, int(cholesky_interval))
        self._sm_steps_since_recompute: int = 0

        # Vol-aware γ
        self.gamma_min: float = float(gamma_min)
        self.gamma_max: float = float(gamma_max)

        # Exploratie
        self.selection: Literal["thompson", "ucb"] = selection
        self.ucb_alpha: float = float(ucb_alpha)

        # Asymmetrische counterfactuals
        self.slippage_bps: float = float(slippage_bps)

        # Delayed feedback proxy gewicht
        self.proxy_decay: float = float(np.clip(proxy_decay, 0.0, 1.0))

        # PCA-BANDIT-FIX (Item 4): orthogonaliseer macro-context vóór de
        # bandit-matrices. Sherman-Morrison + Cholesky beschermen alleen
        # tegen rekenkundige instabiliteit; sterk gecorreleerde features
        # (bv. VIX & high-yield spread) maken B ill-conditioned, waardoor
        # Thompson Sampling astronomische varianties op specifieke assen
        # toekent en willekeurige model-wegingen produceert. Een gefitte
        # FeatureOrthogonalizer (PCA op de train-fold) is hier de juiste
        # decorrelatie. Wanneer ``None`` of ``apply_macro_orth=False`` wordt
        # het oude (pass-through) gedrag aangehouden.
        self.orthogonalizer_macro: Any | None = orthogonalizer_macro
        self.apply_macro_orth: bool = bool(apply_macro_orth)

        # Reward clipper (initieel n_arms; resize later indien dim-shift)
        self._reward_clipper: _RewardClipper = _RewardClipper(
            n_arms=self.n_arms,
            sigma_clip=float(huber_clip_sigma),
            alpha=float(huber_alpha),
        )

        self.B: list[np.ndarray] = []
        self.B_inv: list[np.ndarray] = []
        self.theta_hat: list[np.ndarray] = []
        self.y: list[np.ndarray] = []

        # PROXY-FIX (Item 5): houd cumulatieve y-shift bij voor in-flight
        # proxies. update_bandit_proxy() voegt hieraan toe (en aan y);
        # update_bandit() trekt het er weer af vóór het de echte rank-1
        # update toepast. Zo blijft B⁻¹ ongemoeid totdat de trade sluit.
        self._y_proxy_shift: list[np.ndarray] = []

        # AUDIT-FIX (Round 3 — tanh MTM proxy stabilisation):
        # Ruwe intraday-MTM proxy-rewards hebben een fat-tailed verdeling
        # (Solana validator-outages, ETH DeFi-cascades ± 5-10% op één bar).
        # Extreme proxy-waarden kunnen de θ-vector van de bandit destabiliseren
        # ongeacht de RewardClipper — want clip schalt lineair, niet saturerend.
        # Fix: normaliseer proxy_r via tanh((r - μ) / max(σ, ε)) vóór clip.
        #   • μ, σ bijgewerkt via online EMA (α=0.01, init op eerste observatie).
        #   • tanh ∈ (-1, +1) → effectief een zacht clip met behoud van teken en
        #     orde van magnitude.  Kleine rewards (<0.3σ) veranderen nauwelijks;
        #     3σ-outliers worden begrensd op tanh(3) ≈ 0.995.
        # _proxy_r_n        : aantal geobserveerde proxy-rewards (voor init)
        # _proxy_r_mu       : lopend gemiddelde proxy-reward (EMA)
        # _proxy_r_var      : lopende variantie proxy-reward (EMA online)
        self._proxy_r_n:   int   = 0
        self._proxy_r_mu:  float = 0.0
        self._proxy_r_var: float = 1e-6  # kleine init-variantie → σ ≈ 0.001

        # CHIEF AUDIT 2026-05-23 (P-6): teller voor singular-cov fallbacks
        # in Thompson sampling. Zonder logging is een ill-conditioned B_inv
        # een stille bron van "pure exploitation" (sampled = theta_hat),
        # wat exploration effectief uitzet voor de getroffen arm.
        self._singular_cov_count: int = 0

        if self.context_dim > 0:
            self._init_matrices()

    @property
    def _effective_dim(self) -> int:
        """Effectieve feature-dimensie: rff_dim bij RBF-kernel, anders context_dim."""
        if self.kernel == "rbf" and self.context_dim > 0:
            return self.rff_dim
        return self.context_dim

    def _ensure_rff(self) -> None:
        """Lazy init van RFF zodra context_dim bekend is."""
        if self.kernel != "rbf" or self.context_dim <= 0:
            return
        if self._rff is None or self._rff.in_dim != self.context_dim or self._rff.out_dim != self.rff_dim:
            self._rff = _RandomFourierFeatures(
                in_dim=self.context_dim,
                out_dim=self.rff_dim,
                sigma=self.rff_sigma,
                seed=self.rff_seed,
            )

    def _apply_kernel(self, context: np.ndarray) -> np.ndarray:
        """Map raw context naar feature-space (lineair of RBF via RFF)."""
        v = np.asarray(context, dtype=np.float64).flatten()
        if self.kernel == "rbf":
            self._ensure_rff()
            if self._rff is None:
                return v
            return self._rff.transform(v)
        return v

    def _orth_macro(self, raw_context: np.ndarray) -> np.ndarray:
        """PCA-BANDIT-FIX (Item 4): orthogonaliseer raw macro-context indien
        ``apply_macro_orth=True`` én een gefitte ``FeatureOrthogonalizer`` is
        meegegeven. Zonder beide is dit een identiteitsfunctie zodat oude
        opgeslagen ensembles backwards-compatible blijven werken.

        Args:
            raw_context: 1D macro-feature-vector (al ge-aligneerd en geschaald).

        Returns:
            1D vector in PCA-component-ruimte (of identiek aan input bij no-op).
        """
        v = np.asarray(raw_context, dtype=np.float64).flatten()
        if v.size == 0 or not self.apply_macro_orth:
            return v
        orth = self.orthogonalizer_macro
        if orth is None or not getattr(orth, "_is_fitted", False):
            return v
        try:
            transformed = orth.transform(v.reshape(1, -1))
            return np.asarray(transformed, dtype=np.float64).flatten()
        except Exception as exc:  # pragma: no cover
            # BANDIT-DIM-FIX: nooit de ruwe vector teruggeven als PCA actief is
            # maar faalt. De bandit-matrices (B, B_inv, theta_hat) zijn opgespannen
            # in PCA-ruimte (context_dim dimensies). Als hier de ruwe vector
            # (mogelijk 15 dimensies) wordt teruggegeven, crasht de volgende
            # Thompson Sampling iteratie op een X × B⁻¹ dimensie-mismatch.
            # Geef een nulvector van de juiste PCA-dimensie terug: de bandit
            # interpreteert dit als "geen macro context" voor deze bar maar
            # blijft stabiel draaien.
            logger.warning(
                "ContextualBanditEnsemble._orth_macro: PCA-transformatie faalde "
                "(%s). Nulvector (dim=%d) teruggegeven — bandit blijft stabiel.",
                exc, self.context_dim if self.context_dim > 0 else v.size,
            )
            if self.context_dim > 0 and self.apply_macro_orth:
                return np.zeros(self.context_dim, dtype=np.float64)
            return v

    def _init_matrices(self) -> None:
        """Initialiseert aparte state tracking matrices voor elke arm in het ensemble.

        Bij kernel='rbf' worden matrices opgespannen in RFF-feature ruimte (rff_dim),
        zodat KernelUCB-gedrag wordt benaderd zonder O(t^2) groei.
        """
        self._ensure_rff()
        d = self._effective_dim
        if d <= 0:
            return
        self.B = [np.eye(d, dtype=np.float64) for _ in range(self.n_arms)]
        self.B_inv = [np.eye(d, dtype=np.float64) for _ in range(self.n_arms)]
        self.theta_hat = [np.zeros(d, dtype=np.float64) for _ in range(self.n_arms)]
        self.y = [np.zeros(d, dtype=np.float64) for _ in range(self.n_arms)]
        # PROXY-FIX (Item 5): in-flight proxy-shifts per arm.
        self._y_proxy_shift = [np.zeros(d, dtype=np.float64) for _ in range(self.n_arms)]
        self._reward_clipper.resize(self.n_arms)
        self._sm_steps_since_recompute = 0

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

    def reset_proxy_state(self) -> None:
        """Reset proxy-reward state na joblib reload of voor nieuwe periode/run.

        CHIEF AUDIT 2026-05-23 (P-3): voorkomt dat oude proxy-rewards als
        prior worden gebruikt bij cold-start van een nieuwe periode (5-15%
        prob-distortion in eerste 100 bars). Reset:
          • _proxy_r_n  : observatie-teller voor EMA-bootstrap
          • _proxy_r_mu : lopende mean
          • _proxy_r_var: lopende variantie
          • _y_proxy_shift[k] = 0 voor alle armen (in-flight shift cleared)
        """
        self._proxy_r_n = 0
        self._proxy_r_mu = 0.0
        self._proxy_r_var = 1e-6
        if hasattr(self, "_y_proxy_shift"):
            for arm_k in range(len(self._y_proxy_shift)):
                self._y_proxy_shift[arm_k] = np.zeros_like(self._y_proxy_shift[arm_k])

    def _sample_theta_raw(
        self, theta_hat_k: np.ndarray, B_inv_k: np.ndarray
    ) -> np.ndarray:
        """Legacy MVN-trekking — fallback wanneer geen LedoitWolfThompson actief is.

        Behoudt het pre-blueprint gedrag exact zoals het was, zodat oude
        unit-tests die zonder ``stack`` draaien hun deterministische
        output (modulo ``np.random.seed``) blijven krijgen.
        """
        cov = (self.v ** 2) * np.asarray(B_inv_k, dtype=np.float64)
        cov = (cov + cov.T) / 2.0
        try:
            sampled = np.random.multivariate_normal(
                np.asarray(theta_hat_k, dtype=np.float64), cov
            )
        except np.linalg.LinAlgError:
            # CHIEF AUDIT 2026-05-23 (P-6): silent fallback naar pure
            # exploitation (sampled = theta_hat) verbergt een ill-conditioned
            # B_inv. We tellen de fallbacks en loggen periodiek (elke 10 hits)
            # zodat het in productielogs zichtbaar wordt — zonder de hot-path
            # te overspoelen met warnings.
            self._singular_cov_count = getattr(self, "_singular_cov_count", 0) + 1
            if self._singular_cov_count % 10 == 0:
                logger.warning(
                    "Thompson: %d singular-cov fallbacks (arm exploration disabled)",
                    self._singular_cov_count,
                )
            sampled = np.asarray(theta_hat_k, dtype=np.float64)
        return np.asarray(sampled, dtype=np.float64)

    # ------------------------------------------------------------------
    # BLUEPRINT-FIX (V.9): EntropyGate convenience-laag.
    # ------------------------------------------------------------------
    def gate_signal(self, prob_win: float) -> bool:
        """Beslis of een signaal door de EntropyGate komt.

        ``prob_win`` is de kans op de target-klasse (long ⇒ kans op
        positieve barrier-hit, short ⇒ kans op negatieve barrier-hit). De
        gate evalueert ``H([prob_loss, prob_win])`` t.o.v. een rolling
        percentiel-drempel en blokkeert top-N% entropie-bars (model is
        wezenlijk onbeslist).

        Returns:
            ``True`` als het signaal door mag (lage entropie ⇒ hoge edge),
            ``False`` als de gate NEUTRAL forceert. Wanneer er geen stack
            is geconfigureerd valt de methode terug op ``True`` (legacy
            no-op gedrag).
        """
        gate = self._entropy_gate
        if gate is None:
            return True
        p = float(np.clip(prob_win, 0.0, 1.0))
        decision: EntropyGateDecision = gate.evaluate([1.0 - p, p])
        if not decision.pass_through:
            logger.debug(
                "EntropyGate blokkeert signaal: H=%.4f > τ=%.4f (action=%s).",
                decision.entropy, decision.threshold, decision.action,
            )
        return bool(decision.pass_through)

    # ------------------------------------------------------------------
    # BLUEPRINT-FIX (III.6): NetAlphaReward convenience-wrapper.
    # ------------------------------------------------------------------
    def compute_net_alpha_reward(
        self,
        price_entry: float,
        price_exit: float,
        order_size: float,
        bar_volume: float,
        sigma_per_bar: float,
        side: int = 1,
    ) -> Any | None:
        """Bereken netto-alpha reward met de huidige (Kalman-tracked) η.

        Roept :meth:`NetAlphaReward.compute` aan met dezelfde η-waarde die
        :class:`KalmanImpactObserver` momenteel in de stack vasthoudt.
        Zo blijft de bandit-feedback consistent met de live impact-
        kalibratie. Returns ``None`` wanneer geen stack actief is — caller
        dient dan terug te vallen op handmatig samengestelde rewards.

        De caller geeft ``.reward`` van het resultaat door aan
        :meth:`update_bandit` (long én short kant). Voor een symmetrische
        round-trip is dezelfde reward voor beide kanten passend; voor
        asymmetrische triple-barrier (PT≠SL) moet de caller per kant een
        aparte ``compute_net_alpha_reward`` doen.
        """
        net_alpha = self._net_alpha
        if net_alpha is None or self.stack is None:
            return None
        # Sync η: live observer kan tussentijds geüpdatet zijn → trek de
        # actuele waarde door naar de reward-functie zodat we niet op een
        # stale calibratie afgaan.
        observed_eta = float(getattr(self.stack.eta_observer, "eta", net_alpha.eta))
        net_alpha.eta = observed_eta
        return net_alpha.compute(
            price_entry=float(price_entry),
            price_exit=float(price_exit),
            order_size=float(order_size),
            bar_volume=float(bar_volume),
            sigma_per_bar=float(sigma_per_bar),
            side=int(side),
        )

    # ------------------------------------------------------------------
    # BLUEPRINT-FIX (V.8): FeatureSchemaGuard hook.
    # ------------------------------------------------------------------
    def validate_feature_schema(self, live_feature_names: Sequence[str]) -> None:
        """Verifieer dat de live-pipeline matcht met het opgeslagen schema.

        Raises:
            SchemaMismatchError: bij hash-, count- of name-mismatch — de
                orchestrator MOET trading direct pauzeren.
        """
        guard = self._schema_guard
        if guard is None:
            return  # No-op zonder stack — legacy gedrag.
        guard.check(live_feature_names)

    def get_bandit_weights(self, context_vector: np.ndarray) -> np.ndarray:
        """Bereken arm-gewichten via Thompson Sampling of expliciete UCB.

        - thompson: trekt θ̃ ~ N(θ̂, v²·B⁻¹) per arm — variantie krijgt direct
          gewicht in de score, dus armen met hoge onzekerheid worden af-en-toe
          alsnog gekozen ook al is hun verwachting tijdelijk lager. Bij
          beschikbare ``stack.lw_thompson`` wordt de cov eerst Ledoit-Wolf
          shrunken zodat ill-conditioned B-matrices geen extreme draws
          meer veroorzaken (audit-item III.5).
        - ucb     : score = θ̂·x + α·√(xᵀ·B⁻¹·x). Bonus-term schaalt expliciet
          met de posterior-onzekerheid op die context-richting.
        """
        if context_vector is None or context_vector.size == 0 or self.context_dim == 0:
            return np.ones(self.n_arms) / self.n_arms

        x_feat = self._apply_kernel(context_vector)
        if x_feat.size == 0 or self._effective_dim == 0 or len(self.B_inv) != self.n_arms:
            return np.ones(self.n_arms) / self.n_arms

        scores = np.zeros(self.n_arms, dtype=np.float64)

        for k in range(self.n_arms):
            B_inv_k = self.B_inv[k]
            theta_hat_k = self.theta_hat[k]

            if self.selection == "ucb":
                quad = float(x_feat @ B_inv_k @ x_feat)
                bonus = self.ucb_alpha * float(np.sqrt(max(quad, 0.0)))
                scores[k] = float(np.dot(theta_hat_k, x_feat)) + bonus
            else:
                # Thompson Sampling — expliciet variantie laten meewegen.
                # BLUEPRINT-FIX (III.5): bij beschikbare LedoitWolfThompson-
                # sampler trekken we via shrunken cov i.p.v. de raw
                # multivariate_normal. Sterk gecorreleerde features (BTC/ETH)
                # maken B ill-conditioned → B⁻¹ heeft astronomische
                # varianties op individuele richtingen → MVN produceert
                # extreme exploratie-keuzes. LW-shrinkage corrigeert dit
                # door een convexe combinatie met een gestructureerd
                # target (default identity·avg_var).
                if self._lwts is not None:
                    try:
                        sampled_theta_lw, _delta = self._lwts.sample_theta(
                            theta_hat_k, B_inv_k
                        )
                        sampled_theta = np.asarray(sampled_theta_lw, dtype=np.float64)
                    except Exception as exc:  # pragma: no cover
                        logger.debug(
                            "LedoitWolfTS faalde (%s) — fallback raw MVN.", exc
                        )
                        sampled_theta = self._sample_theta_raw(theta_hat_k, B_inv_k)
                else:
                    sampled_theta = self._sample_theta_raw(theta_hat_k, B_inv_k)
                scores[k] = float(np.dot(sampled_theta, x_feat))

        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        exp_rewards = np.exp(scores - np.max(scores))
        sum_exp = float(np.sum(exp_rewards))
        return exp_rewards / sum_exp if sum_exp > 0 else np.ones(self.n_arms) / self.n_arms

    def _pre_scale_once(
        self,
        x_micro: Any,
        x_meso: Any,
        x_macro: Any,
    ) -> tuple:
        """SHARED-SCALER-FIX (Item 10): ge-aligneerd + ge-scaled per timeframe,
        ÉÉN keer per bar i.p.v. N keer per agent.

        Het oude pad liet elke agent zelf ``scaler.transform()`` aanroepen op
        een gedeelde scaler-instantie. Bij 15 modellen kostte dat 15 redundante
        normalisatie-calls per bar. Hier scalen we centraal via ``meta_scalers``
        en geven agents ``is_pre_scaled=True`` mee. Bijkomend voordeel: agents
        binnen de ensemble hoeven hun eigen ``scaler_*`` niet meer vast te
        houden — ze kunnen op ``None`` blijven, wat ook het pickle-payload
        kleiner maakt.

        Retourneert (x_micro_s, x_meso_s, x_macro_s) — zelfde shape als de
        invoer, maar geschaald en ge-aligneerd op ``meta_feature_map``.
        """
        if not self.models or self.meta_scalers is None or self.meta_feature_map is None:
            return x_micro, x_meso, x_macro

        ref_model = self.models[0]

        def _scale(raw: Any, timeframe: str) -> Any:
            if raw is None:
                return None
            arr = np.asarray(raw.values if hasattr(raw, "values") else raw)
            if arr.size == 0:
                return raw
            if arr.ndim == 3:
                arr = arr[:, -1, :]
            elif arr.ndim == 1:
                arr = arr.reshape(1, -1)
            target_feats = self.meta_feature_map.get(timeframe, []) if self.meta_feature_map else []
            if not target_feats:
                return arr
            try:
                aligned = ref_model._align_input_custom(arr, target_feats, timeframe)
            except RuntimeError:
                # Shape-mismatch — laat de agent het zelf afhandelen (oud pad).
                return raw
            if aligned is None or aligned.size == 0:
                return aligned
            scaler = self.meta_scalers.get(timeframe) if self.meta_scalers else None
            if scaler is not None and hasattr(scaler, "transform"):
                aligned = scaler.transform(aligned)
            return aligned

        return (
            _scale(x_micro, "micro"),
            _scale(x_meso, "meso"),
            _scale(x_macro, "macro"),
        )

    def predict_greybox_strategy(self, x_micro: Any, x_meso: Any, x_macro: Any, **kwargs) -> dict[str, Any]:
        is_pre_scaled = kwargs.get('is_pre_scaled', False)

        # SHARED-SCALER-FIX (Item 10): scaal centraal als de caller niet al
        # vooraf scaled. Agents krijgen daarna is_pre_scaled=True en doen
        # uitsluitend nog hun lokale feature-subset-selectie.
        if not is_pre_scaled and self.meta_scalers is not None:
            x_micro_s, x_meso_s, x_macro_s = self._pre_scale_once(x_micro, x_meso, x_macro)
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

        ref_model = self.models[0]
        target_macro = self.meta_feature_map.get("macro", []) if self.meta_feature_map else []

        if x_macro is not None and np.asarray(x_macro).size > 0 and len(target_macro) > 0:
            macro_arr = np.asarray(x_macro)
            if macro_arr.ndim == 1:
                macro_arr = macro_arr.reshape(1, -1)

            aligned_macro = ref_model._align_input_custom(macro_arr, target_macro, "macro")

            if aligned_macro is not None and aligned_macro.size > 0:
                if is_pre_scaled:
                    context_raw = aligned_macro.flatten()
                else:
                    macro_scaler = self.meta_scalers.get("macro") if self.meta_scalers else None
                    if macro_scaler is not None:
                        context_raw = macro_scaler.transform(aligned_macro).flatten()
                    else:
                        context_raw = aligned_macro.flatten()
                # PCA-BANDIT-FIX (Item 4): orthogonaliseer raw macro-context.
                context = self._orth_macro(context_raw)
            else:
                context = np.array([])
        elif x_macro is not None and self.context_dim > 0:
            # Legacy ensembles trained without meta_feature_map (e.g. train_cpcv.py)
            # pass x_macro as the raw macro feature matrix already aligned to context_dim.
            # Use it directly when its flattened length matches — this restores both
            # get_bandit_weights() arm selection and the macro_context_used stored for
            # the deferred update_bandit() call, eliminating the dim-mismatch warning.
            _macro_flat = np.asarray(x_macro, dtype=np.float64).flatten()
            context = (
                self._orth_macro(_macro_flat)
                if len(_macro_flat) == self.context_dim
                else np.array([])
            )
        else:
            context = np.array([])

        if self.context_dim == 0 and len(context) > 0:
            self.context_dim = len(context)
            self._init_matrices()

        if len(context) == self.context_dim and self.context_dim > 0:
            weights = self.get_bandit_weights(context)
            avg_win_prob = np.sum(win_probs * weights)
        else:
            avg_win_prob = np.mean(win_probs)

        uncertainty = float(np.std(win_probs))
        # SIDE-FIX: zie ook RegimeEnsembleCat — zelfde conventie.
        # min_conf_long < 1.0 → Long-model; min_conf_long == 1.0 → Short-model.
        is_long_model = ref_model.min_conf_long < 1.0

        # THRESHOLD-FIX: rollende percentiel-drempel (zie RegimeEnsembleCat).
        floor_threshold = float(kwargs.get('min_confidence', 0.33))
        if len(self._prob_history) >= 50:
            dynamic_threshold = float(np.percentile(list(self._prob_history), self._dynamic_threshold_pct))
            effective_threshold = max(floor_threshold, dynamic_threshold)
        else:
            effective_threshold = floor_threshold

        # Voeg toe NADAT drempel bepaald is
        self._prob_history.append(float(avg_win_prob))

        max_uncertainty = float(kwargs.get('max_uncertainty', 0.15))
        is_base_signal = (avg_win_prob >= effective_threshold) and (uncertainty <= max_uncertainty)

        # ── BLUEPRINT-FIX (V.9): EntropyGate vóór het signaal naar buiten gaat.
        # Wanneer de bandit-aggregaat onzekerheid (H over [loss,win]) boven
        # de rolling τ ligt, forceren we NEUTRAL. Dit is een EXTRA gate
        # bovenop ``effective_threshold``/``max_uncertainty`` — beide oude
        # filters blijven actief omdat ze andere signalen meten (drempel
        # = niveau, uncertainty = arm-spread, gate = voorspel-entropie).
        gate_passed = self.gate_signal(float(avg_win_prob)) if is_base_signal else True
        final_signal = 1 if (is_base_signal and gate_passed) else 0

        return {
            "signal": final_signal,
            "prob_win": float(avg_win_prob),
            "uncertainty": uncertainty,
            "threshold_used": effective_threshold,
            "side_detected": "LONG" if is_long_model else "SHORT",
            "raw_model_probs": win_probs.tolist(),
            "macro_context_used": context.tolist(),
            "entropy_gate_passed": bool(gate_passed),
        }

    def update_bandit(
        self,
        context: Any,
        raw_model_probs: Any,
        reward_long: float,
        reward_short: float,
        taken_side: Literal["LONG", "SHORT"] | None = None,
    ) -> None:
        """Prediction with Expert Advice: update ALLE armen met asymmetrische counterfactuals.

        De financiële markt is een full-information environment: als model k zegt
        "LONG" en de markt daalt, weten we exact wat de reward van dat model had
        moeten zijn (een verlies). Door alle n_arms armen tegelijk te updaten in plaats
        van slechts de gekozen arm, gebruiken we 100% van de beschikbare informatie.

        ASYMMETRIE-FIX (was: counterfactual_reward = -reward_long voor bearish modellen):
          Met Triple Barrier PT=2R en SL=1R zijn long- en short-returns NIET elkaars
          spiegelbeeld. Als de markt omhoog schiet en LONG PT=+2R raakt, had SHORT de
          SL geraakt: -1R (niet -2R). Door simpelweg het teken om te draaien werd het
          bearish model exponentieel zwaarder bestraft dan de markt werkelijk deed.

          Oplossing: geef beide werkelijke uitkomsten mee en wijs correct toe:
            - bullish model k  (prob_win[k] >= 0.5) → reward_long   (lang-kant resultaat)
            - bearish model k  (prob_win[k] <  0.5) → reward_short  (short-kant resultaat)

        Args:
            context        : context-vector (macro features) op het moment van de trade.
            raw_model_probs: array-like met prob_win voor elk van de n_arms modellen.
            reward_long    : gerealiseerd netto-rendement als LONG (na spread).
            reward_short   : gerealiseerd netto-rendement als SHORT (na spread).
        """
        context_arr = np.asarray(context, dtype=np.float64).flatten()

        # LAZY-INIT-FIX: als predict_greybox_strategy() voor het eerst werd aangeroepen
        # zonder macro-context (lege array), bleef context_dim op 0 en werden alle
        # update_bandit()-aanroepen daarna stil geskipped met een warning.
        if self.context_dim == 0 and len(context_arr) > 0:
            logger.info(
                "update_bandit: context_dim was 0 — initialiseer matrices met dim=%d "
                "(eerste keer dat een geldig context-vector arriveert).",
                len(context_arr),
            )
            self.context_dim = len(context_arr)
            self._init_matrices()

        if self.context_dim == 0 or len(context_arr) != self.context_dim:
            logger.warning(
                "update_bandit: context dim mismatch "
                "(%d vs %d). Update overgeslagen.",
                len(context_arr), self.context_dim,
            )
            return

        probs = np.asarray(raw_model_probs, dtype=np.float64).flatten()

        # Zorg dat we precies n_arms kansen hebben; vul op of knip af indien nodig.
        if len(probs) != self.n_arms:
            logger.warning(
                "update_bandit: %d probs ontvangen maar n_arms=%d. "
                "Ontbrekende armen krijgen prob_win=0.5 (neutraal).",
                len(probs), self.n_arms,
            )
            padded = np.full(self.n_arms, 0.5, dtype=np.float64)
            padded[: min(len(probs), self.n_arms)] = probs[: self.n_arms]
            probs = padded

        # Map naar feature-space (lineair of RFF/RBF) — bandit werkt vanaf hier
        # in _effective_dim ruimte, niet in raw context_dim.
        x_feat = self._apply_kernel(context_arr)
        if x_feat.size == 0 or self._effective_dim == 0:
            logger.warning("update_bandit: kernel-mapping leverde lege feature-vector op.")
            return

        # SLIPPAGE-FIX: counterfactual reward op de NIET-uitgevoerde kant overschat
        # de PnL omdat geen marktimpact/spread werd betaald. Trek slippage_bps af van
        # de kant die niet feitelijk werd gehandeld. taken_side=None → behavior als
        # voorheen (geen extra correctie, gebruiker geeft al netto-rendementen).
        slip = self.slippage_bps * 1e-4
        adj_long = float(reward_long)
        adj_short = float(reward_short)
        if slip > 0.0 and taken_side is not None:
            if taken_side == "LONG":
                adj_short = adj_short - slip if adj_short > 0 else adj_short + slip * 0.5
            elif taken_side == "SHORT":
                adj_long = adj_long - slip if adj_long > 0 else adj_long + slip * 0.5

        # PROXY-FIX (Item 5): voordat de echte rank-1 update wordt toegepast,
        # eerst alle in-flight proxy y-shifts terugdraaien zodat de bandit
        # niet "dubbel telt" (proxy + werkelijk rendement). Hierna staat
        # ``self.y`` weer op de pre-proxy waarde en kan de gerealiseerde
        # update normaal worden toegepast (incl. SM-update op B⁻¹).
        if (
            len(self._y_proxy_shift) == self.n_arms
            and all(s.shape == self.y[k].shape for k, s in enumerate(self._y_proxy_shift))
        ):
            for arm_k in range(self.n_arms):
                if np.any(self._y_proxy_shift[arm_k] != 0.0):
                    self.y[arm_k] = self.y[arm_k] - self._y_proxy_shift[arm_k]
                    self._y_proxy_shift[arm_k] = np.zeros_like(self._y_proxy_shift[arm_k])
        else:
            # Dim mismatch (na re-init) → reset gewoon de tracking-buffers.
            self._y_proxy_shift = [np.zeros_like(self.y[k]) for k in range(self.n_arms)]

        # Counterfactual update voor ELKE arm — juiste uitkomst per kant.
        # SIDE-FIX: het ensemble moet weten of het een Long- of Short-ensemble is.
        # Voor Long-ensemble: prob_win_k >= 0.5 → model verwacht winst via LONG → reward_long.
        # Voor Short-ensemble: prob_win_k >= 0.5 → model verwacht winst via SHORT → reward_short.
        # Zonder deze fix ontving het Short-ensemble de reward_long als beloning wanneer het
        # met overtuiging een short-signaal gaf — training optimaliseerde op de verkeerde kant.
        # CHIEF AUDIT 2026-05-23 (P-7): expliciete side krijgt voorrang op
        # de fragile min_conf_long-threshold-detectie. Legacy-pad blijft
        # voor backwards-compat met oude opgeslagen ensembles, met warning.
        if self.side is not None:
            _is_long_ens = (self.side == "LONG")
        else:
            logger.warning(
                "ContextualBanditEnsemble.side niet gezet — terugval op legacy "
                "min_conf_long-detectie. Set ``side=\"LONG\"/\"SHORT\"`` bij "
                "constructie om dit pad te verlaten (deprecated)."
            )
            _is_long_ens = self.models[0].min_conf_long < 1.0 if self.models else True
        for arm_k in range(self.n_arms):
            prob_win_k = probs[arm_k]
            if _is_long_ens:
                # Bullish arm → long-rendement; bearish arm → short-rendement.
                counterfactual_reward = adj_long if prob_win_k >= 0.5 else adj_short
            else:
                # Short-ensemble: hoge prob_win_k betekent "verwacht SHORT-winst" → reward_short.
                counterfactual_reward = adj_short if prob_win_k >= 0.5 else adj_long

            # HUBER-FIX: clip op ±k·σ rond running EMA-gemiddelde. Voorkomt dat een
            # flash-crash reward de θ-vector explodeert.
            r_clipped = self._reward_clipper.clip(arm_k, counterfactual_reward)
            self._update_arm(arm_k, x_feat, r_clipped)

        # Één trade = één bar; teller stijgt hier (niet per arm).
        self.trade_count += 1
        self._sm_steps_since_recompute += 1

        # Periodieke exacte Cholesky-recompute (vervangt pure SM error-accumulatie)
        # + legacy reset_interval fallback.
        self._maybe_cholesky_recompute()
        self._maybe_resymmetrize()

    def _update_arm(self, arm_index: int, x: np.ndarray, reward: float) -> None:
        """Low-level Sherman-Morrison update voor een specifieke arm.

        trade_count wordt NIET hier verhoogd omdat update_bandit() deze methode
        n_arms keer aanroept per bar. De teller wordt extern bijgehouden via
        increment_trade_count() zodat de reset-check klopt.
        """
        if arm_index < 0 or arm_index >= self.n_arms:
            logger.error(f"Invalid arm_index {arm_index}.")
            return

        x_vec = np.asarray(x, dtype=np.float64).reshape(-1, 1)

        self.B[arm_index] = self.gamma * self.B[arm_index] + x_vec @ x_vec.T
        self.y[arm_index] = self.gamma * self.y[arm_index] + reward * x_vec.flatten()

        A_inv = self.B_inv[arm_index] / self.gamma
        denom = 1.0 + float(x_vec.T @ A_inv @ x_vec)
        # Bescherm tegen near-singular SM update (denom → 0 ⇒ B_inv explodeert).
        if abs(denom) < 1e-12:
            denom = 1e-12 if denom >= 0 else -1e-12
        num = A_inv @ x_vec @ x_vec.T @ A_inv

        self.B_inv[arm_index] = A_inv - (num / denom)
        self.B_inv[arm_index] = (self.B_inv[arm_index] + self.B_inv[arm_index].T) / 2.0
        self.theta_hat[arm_index] = (self.B_inv[arm_index] @ self.y[arm_index]).flatten()

    def _maybe_resymmetrize(self) -> None:
        """Herbereken B_inv exact via linalg.inv voor alle armen elke reset_interval bars.

        Wordt éénmaal per update_bandit()-aanroep gecheckt (= één keer per bar),
        ongeacht hoeveel armen er zijn bijgewerkt.
        """
        if self.reset_interval <= 0 or self.trade_count % self.reset_interval != 0:
            return

        d = self._effective_dim
        if d <= 0:
            return

        for k in range(self.n_arms):
            self.B[k] = (self.B[k] + self.B[k].T) / 2.0
            try:
                self.B_inv[k] = np.linalg.inv(self.B[k])
            except np.linalg.LinAlgError:
                regularization = 1e-4 * np.eye(d)
                self.B[k] += regularization
                self.B_inv[k] = np.linalg.inv(self.B[k])
            self.B_inv[k] = (self.B_inv[k] + self.B_inv[k].T) / 2.0
            self.theta_hat[k] = (self.B_inv[k] @ self.y[k]).flatten()

    def _maybe_cholesky_recompute(self) -> None:
        """Exacte herberekening B_inv via Cholesky-decompositie elke cholesky_interval steps.

        Pure rank-1 Sherman-Morrison stapelt floating-point fouten op; na duizenden
        bars wordt B_inv asymmetrisch / niet-positief-definiet en breekt Thompson
        Sampling. Cholesky (B = L·Lᵀ) + dubbele triangulaire-solve geeft een
        numeriek stabiele exacte inverse — typisch 2-3× nauwkeuriger dan np.linalg.inv
        op slecht-geconditioneerde matrices, en geeft tegelijk een gratis PD-check
        (LinAlgError bij niet-PD → Tikhonov regularisatie + retry).
        """
        if self.cholesky_interval <= 0:
            return
        if self._sm_steps_since_recompute < self.cholesky_interval:
            return
        self._sm_steps_since_recompute = 0

        d = self._effective_dim
        if d <= 0:
            return
        I_d = np.eye(d, dtype=np.float64)

        for k in range(self.n_arms):
            B_k = (self.B[k] + self.B[k].T) / 2.0
            jitter = 1e-8
            for _attempt in range(5):
                try:
                    L = np.linalg.cholesky(B_k + jitter * I_d)
                    # B_inv = L^{-T} · L^{-1} via twee triangulaire solves.
                    Y = np.linalg.solve(L, I_d)
                    B_inv_k = np.linalg.solve(L.T, Y)
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10.0
            else:
                # Laatste redmiddel — pseudo-inverse blijft eindig.
                B_inv_k = np.linalg.pinv(B_k + 1e-3 * I_d)

            # Re-symmetrize na floating-point drift.
            B_inv_k = (B_inv_k + B_inv_k.T) / 2.0
            self.B[k] = B_k
            self.B_inv[k] = B_inv_k
            self.theta_hat[k] = (B_inv_k @ self.y[k]).flatten()

    # ---------------------------------------------------------------- #
    # Vol-aware gamma scheduler
    # ---------------------------------------------------------------- #
    def set_gamma_dynamic(self, vol_norm: float) -> float:
        """Pas γ aan op basis van een genormaliseerde volatiliteits-proxy ∈ [0, 1].

        vol_norm = 0 (kalme markt)  → γ = gamma_max  (lange halfwaardetijd, traag vergeten)
        vol_norm = 1 (turbulent)    → γ = gamma_min  (korte halfwaardetijd, snel vergeten)

        Roep aan vóór elke update_bandit() of periodiek (bv. elke N bars) met een
        externe vol-voorspeller (GARCH / regime-classifier / realised-vol z-score).
        Retourneert de nieuwe γ voor logging.
        """
        v = float(np.clip(vol_norm, 0.0, 1.0))
        new_gamma = self.gamma_max - (self.gamma_max - self.gamma_min) * v
        self.gamma = float(np.clip(new_gamma, 1e-6, 1.0 - 1e-6))
        return self.gamma

    # ---------------------------------------------------------------- #
    # Delayed-feedback proxy (intraday MTM)
    # ---------------------------------------------------------------- #
    def update_bandit_proxy(
        self,
        context: Any,
        raw_model_probs: Any,
        intraday_mtm_long: float,
        intraday_mtm_short: float,
    ) -> None:
        """Tussentijdse θ-update op basis van mark-to-market PnL vóór trade-close.

        Bij horizon t_max=24 bars zou de bandit anders 24 bars op verouderde θ
        beslissen (Delayed Bandit Feedback). Deze proxy doet een gedempte update
        (gewicht = proxy_decay) op basis van de unrealized PnL zodat θ continu
        meebeweegt. Bij trade-close wordt update_bandit() alsnog met het volledige
        gerealiseerde rendement aangeroepen — maar dan zit θ al dichter bij de
        ware waarde.

        PROXY-FIX (Item 5):
            De oude implementatie deed een gewogen Sherman-Morrison update via
            ``_update_arm_weighted``: zowel ``y`` ALS ``B`` / ``B⁻¹`` werden
            meegewogen. Daardoor kromp ``B⁻¹`` voortijdig — Thompson Sampling
            las dat als "meer data dan we hebben" en ging te snel exploiteren.

            Nieuwe implementatie houdt ``B`` en ``B⁻¹`` strikt vast totdat de
            echte trade sluit. Alleen ``y`` (en daarmee ``θ̂``) krijgt een
            tijdelijke verschuiving; deze wordt op trade-close in
            ``update_bandit()`` weer afgetrokken vóórdat de gerealiseerde
            rank-1 update plaatsvindt.
        """
        if self.proxy_decay <= 0.0:
            return

        context_arr = np.asarray(context, dtype=np.float64).flatten()
        if self.context_dim == 0 and len(context_arr) > 0:
            self.context_dim = len(context_arr)
            self._init_matrices()
        if self.context_dim == 0 or len(context_arr) != self.context_dim:
            return

        x_feat = self._apply_kernel(context_arr)
        if x_feat.size == 0 or self._effective_dim == 0:
            return

        probs = np.asarray(raw_model_probs, dtype=np.float64).flatten()
        if len(probs) != self.n_arms:
            padded = np.full(self.n_arms, 0.5, dtype=np.float64)
            padded[: min(len(probs), self.n_arms)] = probs[: self.n_arms]
            probs = padded

        # CHIEF AUDIT 2026-05-23 (P-7): expliciete side krijgt voorrang op
        # de fragile min_conf_long-threshold-detectie. Legacy-pad blijft
        # voor backwards-compat met oude opgeslagen ensembles, met warning.
        if self.side is not None:
            _is_long_ens = (self.side == "LONG")
        else:
            logger.warning(
                "ContextualBanditEnsemble.side niet gezet — terugval op legacy "
                "min_conf_long-detectie. Set ``side=\"LONG\"/\"SHORT\"`` bij "
                "constructie om dit pad te verlaten (deprecated)."
            )
            _is_long_ens = self.models[0].min_conf_long < 1.0 if self.models else True
        decay = float(np.clip(self.proxy_decay, 0.0, 1.0))
        if decay <= 0.0:
            return

        # Zorg dat de tracking-buffer dezelfde dim heeft als de huidige y.
        if (
            len(self._y_proxy_shift) != self.n_arms
            or any(s.shape != self.y[k].shape for k, s in enumerate(self._y_proxy_shift))
        ):
            self._y_proxy_shift = [np.zeros_like(self.y[k]) for k in range(self.n_arms)]

        # AUDIT-FIX (Round 3 — tanh MTM proxy stabilisation):
        # Update online EMA μ/σ voor tanh-normalisering met de twee proxy-values
        # van deze bar (long en short worden allebei als input beschouwd).
        _PROXY_EMA_ALPHA: float = 0.01
        for _raw in (intraday_mtm_long, intraday_mtm_short):
            if np.isfinite(_raw):
                self._proxy_r_n += 1
                if self._proxy_r_n == 1:
                    self._proxy_r_mu  = float(_raw)
                    self._proxy_r_var = 1e-6
                else:
                    _delta = float(_raw) - self._proxy_r_mu
                    self._proxy_r_mu  += _PROXY_EMA_ALPHA * _delta
                    self._proxy_r_var  = (
                        (1.0 - _PROXY_EMA_ALPHA) * self._proxy_r_var
                        + _PROXY_EMA_ALPHA * _delta * _delta
                    )
        _proxy_sigma: float = float(np.sqrt(max(self._proxy_r_var, 1e-9)))

        def _tanh_squash(r: float) -> float:
            """Normaliseer via z-score + tanh → ∈ (-1, +1)."""
            z = (r - self._proxy_r_mu) / _proxy_sigma
            return float(np.tanh(z))

        for arm_k in range(self.n_arms):
            prob_win_k = probs[arm_k]
            if _is_long_ens:
                proxy_r = intraday_mtm_long if prob_win_k >= 0.5 else intraday_mtm_short
            else:
                proxy_r = intraday_mtm_short if prob_win_k >= 0.5 else intraday_mtm_long
            # Tanh squash vóór clip: satureer fat-tail outliers zacht.
            proxy_r = _tanh_squash(proxy_r)
            r_clipped = self._reward_clipper.clip(arm_k, proxy_r)
            # Tijdelijke y-shift (geen B / B⁻¹ aanraken):
            #   shift = decay * r_clipped * x_feat
            shift = decay * float(r_clipped) * np.asarray(x_feat, dtype=np.float64)
            self.y[arm_k] = self.y[arm_k] + shift
            self._y_proxy_shift[arm_k] = self._y_proxy_shift[arm_k] + shift

            # AUDIT-FIX (issue #7 — MTM Proxy Drift / norm cap):
            #   _y_proxy_shift accumuleert zonder begrenzing gedurende een
            #   multi-bar positie.  Na T bars is de totale shift O(T × decay × clip).
            #   Bij een positie van 48 bars (24h), decay=0.99, clip=0.02:
            #     max_shift ≈ 48 × 0.99 × 0.02 ≈ 0.95 × |x_feat|
            #   → θ̂ drifts systematisch weg van de gerealiseerde reward-ruimte.
            #   Oplossing: begrens de L2-norm van de cumulatieve proxy-shift per arm.
            #   De cap is proportioneel aan de y-norm zodat de verhouding shift/y
            #   begrensd blijft (hier: max 20% van |y|, afgestemd empirisch).
            _Y_PROXY_MAX_FRAC = 0.20
            y_norm = float(np.linalg.norm(self.y[arm_k]))
            shift_norm = float(np.linalg.norm(self._y_proxy_shift[arm_k]))
            max_shift_norm = _Y_PROXY_MAX_FRAC * max(y_norm, 1e-9)
            if shift_norm > max_shift_norm and shift_norm > 1e-12:
                # Schaal de accumulatieve shift terug naar de cap en pas y aan.
                scale_back = max_shift_norm / shift_norm
                corrected_shift = self._y_proxy_shift[arm_k] * scale_back
                # Trek het overschot af uit y.
                excess = self._y_proxy_shift[arm_k] - corrected_shift
                self.y[arm_k] = self.y[arm_k] - excess
                self._y_proxy_shift[arm_k] = corrected_shift

            # θ̂ direct herberekenen op de bestaande (ongewijzigde) B⁻¹.
            self.theta_hat[arm_k] = (self.B_inv[arm_k] @ self.y[arm_k]).flatten()

        # Belangrijk: GEEN _sm_steps_since_recompute++ en GEEN
        # _maybe_cholesky_recompute() — er zijn geen rank-1 updates op B
        # geweest, dus de Cholesky-staat is ongewijzigd.

    def _update_arm_weighted(
        self, arm_index: int, x: np.ndarray, reward: float, weight: float = 1.0
    ) -> None:
        """Sherman-Morrison update met optionele weegfactor (voor proxy-updates).

        weight=1.0 → identiek aan _update_arm. weight<1.0 → schaalt de rank-1
        bijdrage zodat de proxy minder gewicht heeft dan een gerealiseerde reward.
        """
        if arm_index < 0 or arm_index >= self.n_arms:
            return
        w = float(np.clip(weight, 0.0, 1.0))
        if w <= 0.0:
            return

        x_vec = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        # Geweegd: vervang x → √w · x in de outer-product zodat bijdrage = w·xxᵀ.
        xw = np.sqrt(w) * x_vec

        self.B[arm_index] = self.gamma * self.B[arm_index] + xw @ xw.T
        self.y[arm_index] = self.gamma * self.y[arm_index] + (w * reward) * x_vec.flatten()

        A_inv = self.B_inv[arm_index] / self.gamma
        denom = 1.0 + float(xw.T @ A_inv @ xw)
        if abs(denom) < 1e-12:
            denom = 1e-12 if denom >= 0 else -1e-12
        num = A_inv @ xw @ xw.T @ A_inv

        self.B_inv[arm_index] = A_inv - (num / denom)
        self.B_inv[arm_index] = (self.B_inv[arm_index] + self.B_inv[arm_index].T) / 2.0
        self.theta_hat[arm_index] = (self.B_inv[arm_index] @ self.y[arm_index]).flatten()
