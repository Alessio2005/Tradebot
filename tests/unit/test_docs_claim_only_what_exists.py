# tests/unit/test_docs_claim_only_what_exists.py
"""Documentatiedrift is een gate, geen opruimactie. Stage E-3, no-go 20.

*"Nul documenten die niet-bestaande artefacten claimen."* Een eenmalige
opruiming haalt dat een dag lang; daarna loopt het opnieuw uit elkaar, want een
verwijzing verrot stil. Wat gemeten kan worden, hoort een poort te zijn.

WAT DEZE TEST WEL EN NIET AFDWINGT
===================================
Hij leest elk `docs/*.md`, verzamelt elke padverwijzing tussen backticks, en
eist dat zij oplost. Wat hij NIET doet is verbieden dat een document een
ontbrekend pad NOEMT: `docs/data_dictionary.md` legt uit dat
`artefacts/portfolio/equity_curve.parquet` sinds Phase 5 niet meer bestaat, en
dat is juist goede documentatie. Datzelfde onderscheid kwam eerder in deze fase
terug bij `min_confidence`: een test die het documenteren van een besluit
verbiedt, dwingt af dat besluiten ongedocumenteerd blijven.

Het onderscheid wordt gemaakt met een EXPLICIETE lijst hieronder, met per
verwijzing de reden. Dat is bewust handwerk: elke uitzondering is dan een
gelezen besluit in plaats van een stilzwijgende doorlaat, net zoals de
`GOVERNED`-tuple in `scripts/check_hardcoded_params.py`.

De lijst kan ook niet verrotten: zodra een pad erop WEL gaat bestaan, faalt de
test net zo goed. Een uitzondering die niet meer nodig is, moet weg.

Ref: fase-opdracht Stage E-3, exit-criterium E3, no-go 20.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

PATH_REF = re.compile(
    r"`((?:src|tests|apps|conf|docs|reports|artefacts|scripts|infra|data)"
    r"/[A-Za-z0-9_./-]+)`"
)

#: Verwijzingen naar paden die BEWUST niet bestaan, met de reden. Elke regel is
#: een gelezen besluit; wie er een toevoegt, verklaart waarom het geen defect is.
KNOWN_ABSENT: dict[str, str] = {
    # Vervallen met de backtesters die Phase 5 heeft verwijderd. De documenten
    # noemen ze om die verwijdering uit te leggen.
    "artefacts/portfolio/equity_curve.parquet":
        "data_dictionary.md §7 legt uit dat Phase 5 dit artefact liet vervallen",
    "artefacts/alpha_signals/": "vervallen stage-output, zie architecture.md §2",
    "reports/tearsheets/": "vervallen stage-output, zie architecture.md §2",
    "artefacts/portfolio/": "vervallen stage-output, zie architecture.md §2",
    # Governance-artefacten die pas bij hun eerste gebeurtenis ontstaan.
    "artefacts/governance/gap_ledger.jsonl":
        "wordt pas aangemaakt bij het eerste datagat; DATA_REGISTER.md zegt dat",
    "artefacts/governance/sign_off_log.jsonl":
        "nog niet in gebruik; model_risk_policy.md markeert dat expliciet",
    # Configuratie die geen checkout ooit ziet. `.gitignore` regel 34 (`env/`,
    # geschreven voor virtualenvs) sluit ook de Hydra-`env`-groep uit die
    # `conf/config.yaml` in zijn `defaults:` noemt. Het pad corrigeren zou
    # liegen -- er is geen tweede locatie -- en het bestand aanmaken zou een
    # prod-profiel verzinnen dat niemand heeft gemeten.
    "conf/env/prod.yaml":
        "niet getrackt: .gitignore `env/` sluit de Hydra-env-groep uit; "
        "model_risk_policy.md par. 2 legt die afwezigheid expliciet uit",
    # Dezelfde oorzaak als hierboven, nu voor de map zelf: RISK_MANDATE.md §6
    # punt 4 noemt `conf/env/` expliciet als niet-bestaand ("een tweede,
    # stille bron van risicowaarden zodra iemand hem aanmaakt") terwijl
    # `conf/config.yaml` `defaults: - env: dev` declareert. Fase 11, stap 1.3:
    # zonder deze regel is dit pad wél in de repo aangetroffen -- niet
    # getrackt, maar aanwezig als lokale, gegitignorede bestanden op minstens
    # één ontwikkelmachine (`conf/env/{ci,dev,prod,staging}.yaml`) -- wat
    # bevestigt dat RISK_MANDATE §6.4 een reëel risico beschrijft en geen
    # hypothetisch. `_resolves()` leest schijf, niet git; op een verse kloon
    # bestaat de map niet en dit pad hoort hier net als zijn buurregel.
    "conf/env/":
        "niet getrackt: dezelfde .gitignore-regel als conf/env/prod.yaml; "
        "RISK_MANDATE.md §6 punt 4 legt die afwezigheid expliciet uit",
    # Verwijderd in fase 11, stap 5.2 (R-3: één route per grootheid). Het was
    # een tweede fundingroute die een niet-gecertificeerde bron las en bij een
    # ontbrekend bestand stil nullen teruggaf. CODE_REGISTER.md noemt het pad
    # om uit te leggen dat het weg is en waarom.
    "data/funding.py":
        "verwijderd in fase 11 stap 5.2; CODE_REGISTER.md legt uit waarom",
    # Een deliverable die niet kan bestaan. CHAIN_A_STATUS.md stelt vast dat
    # "keten A" nooit heeft bestaan -- geen `CHAIN_A_STATUS.md` in enige
    # branch, geen `momentum_alpha` in enige commit behalve de commit die de
    # masterprompt zelf schreef. D19 vraagt om een SCORE van die keten; er valt
    # niets te scoren. Het pad corrigeren kan niet (er is geen tweede locatie)
    # en het rapport schrijven zou een meting verzinnen.
    "reports/phase10_chain_a_score.md":
        "kan niet bestaan: CHAIN_A_STATUS.md bewijst dat keten A nooit heeft "
        "bestaan, dus D19 heeft geen onderwerp",
    # Onjuiste paden die als correctie worden geciteerd.
    "artefacts/tca/coefficients.yaml":
        "tca_methodology.md citeert dit pad om de correctie te tonen",
    # Artefacten uit het Wave-tijdperk. De documenten die ze noemen dragen een
    # GEARCHIVEERD-kop; zie `test_historical_docs_are_marked_as_such`.
    "artefacts/wave20_runlog/": "Wave-tijdperk",
    "artefacts/wave20_runlog/w22_eq_units.log": "Wave-tijdperk",
    "artefacts/wave20_runlog/w23c_eval.log": "Wave-tijdperk",
    "artefacts/killgates/fx_carry.json": "Wave-tijdperk",
    "artefacts/broad_perp_ohlcv.parquet": "Wave-tijdperk",
    "artefacts/tracks_breadth/": "Wave-tijdperk",
    "artefacts/crypto_hourly.parquet": "Wave-tijdperk",
    "artefacts/governance/hypothesis_ledger_staging_w27.json": "Wave-tijdperk",
    "scripts/w28_phantom_rebalance_impact.py": "Wave-tijdperk",
    "src/tradebot/risk/factor_alpha.py":
        "Wave-tijdperk; de huidige module heet `risk/factor_risk.py`",
}


@lru_cache(maxsize=1)
def _dvc_declared() -> frozenset[str]:
    """Elk pad dat `dvc.yaml` als stage-output of dependency noemt.

    Deze tellen als OPLOSSEND ook wanneer zij niet op schijf staan: zij ontstaan
    bij `dvc repro` en hun afwezigheid op een machine die de pijplijn nog niet
    heeft gedraaid, is de normale toestand. Ze in een handmatige uitzonderings-
    lijst zetten zou betekenen dat die lijst meegroeit met de pijplijn, en dan
    veroudert zij gegarandeerd.
    """
    raw = (ROOT / "dvc.yaml").read_text(encoding="utf-8")
    declared: set[str] = set()
    for match in re.finditer(r"^\s*-\s*([A-Za-z0-9_./$\{\}-]+):?\s*$", raw, re.M):
        path = match.group(1)
        declared.add(path)
        # `artefacts/features/${item}.parquet` -> ook `artefacts/features/`.
        stem = path.split("${")[0]
        if stem.endswith("/"):
            declared.add(stem)
            declared.add(stem.rstrip("/"))
        elif "/" in stem:
            directory = stem.rsplit("/", 1)[0]
            declared.add(directory + "/")
            declared.add(directory)
    return frozenset(declared)


def _resolves(ref: str) -> bool:
    """Bestaat dit pad, op een van de manieren waarop het geschreven wordt?

    Documenten schrijven `data/pit_store.py` waar `src/tradebot/data/pit_store.py`
    wordt bedoeld, en `apps/live_trader` zonder extensie. Beide zijn leesbaar en
    geen defect. Een pad dat `dvc.yaml` declareert telt ook: zie
    :func:`_dvc_declared`.
    """
    candidates = [ROOT / ref, ROOT / "src" / "tradebot" / ref]
    if not Path(ref).suffix:
        candidates += [ROOT / f"{ref}.py",
                       ROOT / "src" / "tradebot" / f"{ref}.py"]
    if any(c.exists() for c in candidates):
        return True
    return ref in _dvc_declared() or ref.rstrip("/") in _dvc_declared()


def _references(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {m.group(1).split("::")[0].rstrip(".,;:")
            for m in PATH_REF.finditer(text)}


class TestEveryReferenceResolves:
    def test_no_document_claims_a_path_that_does_not_exist(self) -> None:
        offenders: dict[str, list[str]] = {}
        for doc in sorted(DOCS.glob("*.md")):
            bad = sorted(r for r in _references(doc)
                         if not _resolves(r) and r not in KNOWN_ABSENT)
            if bad:
                offenders[doc.name] = bad
        assert not offenders, (
            f"Deze documenten verwijzen naar paden die niet bestaan: "
            f"{offenders}. Corrigeer de verwijzing, of zet hem met een reden in "
            f"KNOWN_ABSENT wanneer het document juist uitlegt dat het pad weg "
            f"is. No-go 20.")

    def test_the_exception_list_does_not_rot(self) -> None:
        """Een uitzondering voor iets dat inmiddels bestaat, moet weg."""
        resurrected = sorted(r for r in KNOWN_ABSENT if _resolves(r))
        assert not resurrected, (
            f"Deze paden staan als 'bewust afwezig' maar bestaan inmiddels: "
            f"{resurrected}. Haal ze uit KNOWN_ABSENT.")

    def test_the_exception_list_is_actually_used(self) -> None:
        """Een uitzondering die nergens wordt genoemd, is dode ballast."""
        cited: set[str] = set()
        for doc in DOCS.glob("*.md"):
            cited |= _references(doc)
        unused = sorted(set(KNOWN_ABSENT) - cited)
        assert not unused, (
            f"Deze uitzonderingen worden door geen enkel document meer "
            f"genoemd: {unused}. Haal ze weg.")


class TestEveryDocumentDeclaresItsStatus:
    """E-3: elk document draagt een verificatiedatum of wordt gearchiveerd."""

    def test_no_document_is_undated(self) -> None:
        undated = [
            doc.name for doc in sorted(DOCS.glob("*.md"))
            if not re.search(
                r"Geverifieerd tegen de codebase op|GEARCHIVEERD|"
                r"Gecontroleerd op",
                doc.read_text(encoding="utf-8", errors="ignore"))
        ]
        assert not undated, (
            f"Deze documenten zeggen niet wanneer zij voor het laatst tegen de "
            f"code zijn gehouden: {undated}. E-3 eist een verificatiedatum of "
            f"een archiefmarkering.")


class TestTheDetectorCanGoRed:
    """Zonder deze drie bewaken de tests hierboven niets."""

    def test_a_dangling_reference_is_detected(self, tmp_path: Path) -> None:
        doc = tmp_path / "x.md"
        doc.write_text("zie `src/tradebot/bestaat_niet_xyz.py` hiervoor.\n",
                       encoding="utf-8")
        refs = _references(doc)
        assert refs == {"src/tradebot/bestaat_niet_xyz.py"}
        assert not _resolves("src/tradebot/bestaat_niet_xyz.py")

    def test_a_real_reference_resolves(self) -> None:
        assert _resolves("src/tradebot/backtest/engine.py")

    @pytest.mark.parametrize("ref,expected", [
        ("data/pit_store.py", True),      # geschreven zonder src/tradebot-prefix
        ("apps/live_trader", True),       # geschreven zonder .py
        ("conf/risk/default.yaml", True),
    ])
    def test_the_writing_conventions_are_understood(
            self, ref: str, expected: bool) -> None:
        assert _resolves(ref) is expected
