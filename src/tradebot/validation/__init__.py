"""L11 — Statistical Validation & Falsification.

De poort waar elk Phase 6-model doorheen moet. Twee soorten bewijs wonen hier:

    data_adequacy   KAN deze data de vraag beantwoorden?  (vóór de fit)
    econometrics    MAG dit model op deze reeks?          (vóór de fit)
    vol_metrics     WON het model?                        (na de fit)

De volgorde is niet toevallig. Op 1.743 bars en 6 gecertificeerde reeksen is de
duurste fout niet een verkeerd antwoord maar een antwoord op een vraag die deze
data niet kan dragen — dat ziet er statistisch geldig uit en is het niet.
"""
from __future__ import annotations

__all__: list[str] = []
