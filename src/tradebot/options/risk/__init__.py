"""De risicopoort van de options-setup (detailontwerp 8).

Eén ingang: `gate.RiskGate.review`. Alleen die maakt een `ApprovedOrder`, en alleen een
`ApprovedOrder` gaat naar een broker. De grenzen zijn fracties van het kapitaal en staan in de
sectie `risk` van `conf/options/default.yaml`.
"""
