"""Prijzen, implied volatility en Greeks met QuantLib (detailontwerp sectie 5).

* `ql_env`       de QuantLib-omgeving: `evaluationDate` alleen via een contextmanager, en de
                 looptijd op seconden.
* `engines`      `Method`, de engine-instellingen en `engine_for` (welke engine waarvoor).
* `vanilla`      één waardering van een vanilla optie (Europees of Amerikaans).
* `implied_vol`  IV met expliciete foutcodes, nooit een stille NaN.
* `greeks`       Greeks per aandeel, analytisch of uit de engine waar dat kan, anders met
                 bump-and-reprice.
"""
