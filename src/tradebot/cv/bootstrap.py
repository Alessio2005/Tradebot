"""Sequential Bootstrap (López de Prado AFML §4.5).

Migrated from legacy sequential_bootstrap.py.
Bug-fix: bar-space vs sample-space index separation preserved.

Sequential Bootstrap (López de Prado, AFML 2018) — correcte bar-space implementatie.

KRITIEKE BUG-FIX (issue #1 — SB Index):
  De originele code gebruikte ``t0_indices = np.arange(n_samples)`` waardoor
  sample-indices [0, 1, …, n-1] werden vergeleken met absolute bar-indices in
  t1_indices (bv. 1523, 2047).  Dit mengde twee verschillende indexruimtes:

    * sample-space  : 0 … n_samples-1
    * bar-space     : 0 … max_bar_idx  (kan 50 000+ zijn)

  Gevolgen:
    1. phi[] werd geïndexeerd over een quasi-leeg bereik [sample_idx .. bar_idx],
       wat vrijwel elke overlap vals telde.
    2. De early-exit (``if t_start_j > t_end_sel: break``) vergeleek een
       sample-index (≤ n_samples) met een bar-index (>> n_samples) → triggerde
       nooit → O(n²) innerloop liep altijd volledig door.
    3. phi-array-grootte = max(t1_indices) kon 50 000 entries alloceren voor
       een dataset van 200 samples → verspilling van geheugen en cache.

FIX:
  * Voeg ``t0_indices: np.ndarray`` toe aan de functiesignatuur (vereiste input:
    de bar-index waarop elk event *begint*, chronologisch gesorteerd).
  * Remap beide arrays naar *relatieve bar-space* door min(t0_indices) af te
    trekken zodat phi compact blijft: grootte = max_rel + 2 (typisch << n_bars).
  * Alle vergelijkingen werken nu in dezelfde relatieve ruimte → early-exit
    werkt correct → O(N·overlap) gemiddeld i.p.v. O(N²).

Pylance-compliance:
  * Expliciete np.ndarray type-annotaties op alle parameters en return.
  * Geen impliciete Any; njit-functies zijn untyped vanuit Pylance-perspectief
    maar de wrapper-docstring documenteert de contracten.
"""
from __future__ import annotations

import numpy as np
from numba import njit

# =============================================================================
# NUMBA KERNEL — relatieve bar-space, O(N·overlap)
# =============================================================================

@njit(cache=True)
def _sb_kernel(
    t0_rel: np.ndarray,   # relatieve bar-start indices (int64, gesorteerd)
    t1_rel: np.ndarray,   # relatieve bar-end   indices (int64)
    n_draws: int,
    seed: int,            # Numba-internal RNG seed (-1 = do not seed)
) -> np.ndarray:
    """Interne Sequential Bootstrap kernel — werkt volledig in relatieve bar-space.

    Parameters
    ----------
    t0_rel : 1-D int64 array, shape (n_samples,)
        Relatieve startbar van elk event (= t0_abs - min(t0_abs)).
        Verondersteld gesorteerd (chronologisch).
    t1_rel : 1-D int64 array, shape (n_samples,)
        Relatieve eindbar van elk event  (= t1_abs - min(t0_abs)).
    n_draws : int
        Aantal trekkingen (= gewenste bag-grootte).

    Returns
    -------
    chosen_indices : 1-D int64 array, shape (n_draws,)
        Indices in [0, n_samples) die de bootstrap-bag vormen.
    """
    if seed >= 0:
        np.random.seed(seed)

    n_samples = len(t1_rel)

    # phi: telt hoe vaak elke relatieve bar is gedekt door de huidige bag.
    # Grootte = max(t1_rel) + 2 — compact dankzij de remapping.
    max_rel: int = 0
    for x in t1_rel:
        max_rel = max(max_rel, x)
    phi = np.zeros(max_rel + 2, dtype=np.float64)

    chosen_indices = np.zeros(n_draws, dtype=np.int64)
    # Initieel: alle samples even uniek (u_avg = 1.0).
    u_avg = np.ones(n_samples, dtype=np.float64)

    for i in range(n_draws):
        # ── Stap 1: bereken trekkingskansen o.b.v. huidige uniqueness ────
        total_u = 0.0
        for k in range(n_samples):
            total_u += u_avg[k]

        if total_u > 1e-9:
            probs = u_avg / total_u
        else:
            probs = np.ones(n_samples, dtype=np.float64) / float(n_samples)

        # ── Stap 2: weighted random choice via cumulatieve som ────────────
        cum = np.cumsum(probs)
        r = np.random.random()
        sel = np.searchsorted(cum, r)
        if sel >= n_samples:
            sel = n_samples - 1
        chosen_indices[i] = sel

        # ── Stap 3: update phi met het getrokken event ────────────────────
        t_s = t0_rel[sel]   # relatieve start van geselecteerde sample
        t_e = t1_rel[sel]   # relatieve eind  van geselecteerde sample
        for t in range(t_s, t_e + 1):
            phi[t] += 1.0

        # ── Stap 4: herbereken u_avg voor de VOLGENDE trekking ────────────
        # We berekenen de verwachte uniqueness van elk event j als het aan de
        # bag zou worden toegevoegd: u(j) = mean_t∈[t0j,t1j] [1/(phi[t]+1)].
        #
        # Early-exit werkt nu correct:
        #   * ``t_end_j < t_s``: event j eindigt vóór de geselecteerde start
        #     → kan niet overlappen → sla over.
        #   * ``t_start_j > t_e``: event j begint na de geselecteerde eind
        #     → en omdat t0_rel gesorteerd is, geldt dit voor alle volgende j
        #     → breek de lus.
        for j in range(n_samples):
            t_start_j = t0_rel[j]
            t_end_j   = t1_rel[j]

            if t_end_j < t_s:
                # Event j volledig vóór geselecteerde — geen overlap, geen update nodig.
                continue
            if t_start_j > t_e:
                # Event j volledig ná geselecteerde — en alle volgende ook (gesorteerd).
                break

            duration = (t_end_j - t_start_j) + 1
            if duration <= 0:
                u_avg[j] = 0.0
                continue

            overlap_sum = 0.0
            for t in range(t_start_j, t_end_j + 1):
                # 1 / (phi[t] + 1): als phi[t]=0 → volledig uniek; als phi[t]=1 → helft uniek.
                overlap_sum += 1.0 / (phi[t] + 1.0)
            u_avg[j] = overlap_sum / float(duration)

    return chosen_indices


# =============================================================================
# PUBLIEKE WRAPPER — converteert absolute bar-indices naar relatieve ruimte
# =============================================================================

def get_sequential_bootstrap_indices(
    t0_indices: np.ndarray,
    t1_indices: np.ndarray,
    n_draws: int,
    seed: int | None = None,
) -> np.ndarray:
    """Sequential Bootstrap trekkingen met correcte bar-space overlap-detectie.

    Implementeert het algoritme uit López de Prado (2018) §4.5 waarbij de
    uniqueness van elk event wordt berekend ten opzichte van de *bar*-tijdlijn,
    niet ten opzichte van de sample-lijst.

    Parameters
    ----------
    t0_indices : np.ndarray, shape (n_samples,), dtype int-compatible
        Absolute bar-index waarop elk event *begint* (= index in de ruwe
        bar-DataFrame vóór event-filtering).  Moet chronologisch gesorteerd zijn.
    t1_indices : np.ndarray, shape (n_samples,), dtype int-compatible
        Absolute bar-index waarop elk event *eindigt* (triple barrier t1).
        Moet element-wise >= t0_indices zijn.
    n_draws : int
        Aantal bootstrap-trekkingen (bag-grootte).  Typisch = n_samples voor
        een volledige bootstrap bag; groter voor oversampling.

    Returns
    -------
    np.ndarray, shape (n_draws,), dtype int64
        Indices in [0, n_samples) — gebruik als ``df.iloc[result]`` of
        als ``sample_weight``-selector.

    Notes
    -----
    **Waarom t0_indices verplicht?**
    Zonder de werkelijke bar-startpositie is het onmogelijk om de overlapping
    in bar-space te berekenen.  ``np.arange(n_samples)`` als proxy was een
    stille bug: het mengde sample-space met bar-space, waardoor phi altijd
    bijna-nul was en de uniqueness-gewichten de bag niet konden decorreleren.

    **Relatieve remapping:**
    Intern worden beide arrays verschoven met ``min(t0_indices)`` zodat de
    phi-array slechts ``max(t1) - min(t0) + 2`` entries breed is.  Dit houdt
    geheugen en cache compact ongeacht de absolute bar-index.

    **Pylance / type-safety:**
    De functie accepteert elk integer-compatible array en cast intern naar
    ``np.int64``.  De kernel is ``@njit``; Pylance ziet het als ``Any``-return
    maar de publieke wrapper is volledig getypeerd.

    Examples
    --------
    >>> import numpy as np
    >>> t0 = np.array([0, 5, 10, 15], dtype=np.int64)
    >>> t1 = np.array([8, 12, 18, 22], dtype=np.int64)
    >>> idx = get_sequential_bootstrap_indices(t0, t1, n_draws=4)
    >>> assert idx.shape == (4,) and idx.dtype == np.int64
    """
    t0 = np.asarray(t0_indices, dtype=np.int64)
    t1 = np.asarray(t1_indices, dtype=np.int64)

    if len(t0) == 0 or len(t1) == 0:
        return np.zeros(n_draws, dtype=np.int64)

    if len(t0) != len(t1):
        raise ValueError(
            f"t0_indices en t1_indices moeten dezelfde lengte hebben, "
            f"maar kregen {len(t0)} en {len(t1)}."
        )

    # Valideer dat t1 >= t0 element-wise (anders is de barrier inconsistent).
    if np.any(t1 < t0):
        raise ValueError(
            "t1_indices moet element-wise >= t0_indices zijn "
            "(barrier-eindtijd vóór starttijd)."
        )

    # Remap naar relatieve bar-space zodat phi compact blijft.
    origin: int = int(t0.min())
    t0_rel = t0 - origin
    t1_rel = t1 - origin

    _seed: int = int(seed) if seed is not None else -1
    return _sb_kernel(t0_rel, t1_rel, int(n_draws), _seed)
