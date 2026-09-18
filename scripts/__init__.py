"""De poort-tooling als reguliere package, zodat niets hem kan overschaduwen.

WAAROM DIT BESTAND BESTAAT
==========================
`tests/unit/test_clean.py`, `test_file_size_ratchet.py` en
`test_reachability_map.py` importeren `scripts.<module>`. Zonder dit bestand is
`scripts/` een NAMESPACE-package, en een namespace-package verliest het altijd
van een REGULIERE package met dezelfde naam -- ongeacht de volgorde van
`sys.path`, dus ook ongeacht de `sys.path.insert(0, _REPO_ROOT)` in
`tests/conftest.py`.

Gemeten 2026-09-18: op een interpreter met `fitz 0.0.1.dev2` of
`google-auth-oauthlib 1.2.1` geinstalleerd -- beide leveren een top-level
`scripts/__init__.py` in site-packages -- viel de testcollectie om met

    ModuleNotFoundError: No module named 'scripts.check_file_size'

en dat leest als een kapotte repository terwijl er niets kapot was.

Geen van beide pakketten staat in de lockfiles, dus CI loopt dit vandaag niet.
Dat is precies de reden om het nu te sluiten en niet later: de kosten zijn dit
bestand, en de storing die het voorkomt is er een die zich voordoet als een
defect in de poorten zelf.
"""
