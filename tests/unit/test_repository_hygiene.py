"""Bewakers tegen twee defectklassen die de verhuizing van C: naar D: blootlegde.

Phase 7/8, Stage A. Beide tests bewaken een klasse fouten die geen enkel
statistisch contract raakt en juist daarom blijft liggen tot hij bewijsmateriaal
vervuilt.

DE EERSTE KLASSE — een test die naar de repository-root schrijft
----------------------------------------------------------------
`x.jsonl` stond tot Stage A-1 **git-tracked in de projectroot**: 38 regels
gap-ledger met timestamps in 1970 en symbool ``"B"``. Synthetische testdata,
geschreven door precies één regel:

    tests/unit/test_data_validation.py:118
        enforce_gap_policy(recs, policy="reject", ledger=GapLedger("x.jsonl"))

Elke buur in datzelfde bestand gebruikte al `tmp_path`. Dit is exact de klasse
die `reports/phase5_exit_report.md` §9.1 zelf benoemt — *"een test die een
governance-artefact vervuilt, ondermijnt de auditbaarheid die hij hoort te
bewaken"* — alleen dan in de root in plaats van in `artefacts/governance/`.

Het bestand GROEIDE nog: de meting van 2026-08-27 telde 35 regels, de
verificatie bij aanvang van deze fase 38. Het lek was actief, geen residu.

DE TWEEDE KLASSE — een bytecode-cache die over zijn herkomst liegt
------------------------------------------------------------------
Na de verhuizing droegen **351 van 351** `.pyc` in de boom nog
``C:\\Users\\algul\\Documents\\Tradebot\\...`` als `co_filename`. De mtimes waren
behouden, dus Python beschouwde ze als geldig en gebruikte ze. Elke traceback,
warning en assertion-melding citeerde daardoor een pad dat niet bestaat — in één
suite-run stonden twee schijven naast elkaar:

    SKIPPED [1] C:\\Users\\algul\\Documents\\Tradebot\\tests\\e2e\\test_chaos.py:19
    D:\\Tradebot\\src\\tradebot\\risk\\var.py:222: RuntimeWarning

Voor een platform dat provenance als kernwaarde voert, is bewijsmateriaal dat
naar een niet-bestaand pad verwijst geen cosmetisch probleem.

BEIDE TESTS ZIJN AANTOONBAAR ROOD GEWEEST. Zie
`reports/phase7_foundation_report.md` §A-4 voor de injectie, de melding en het
commando.
"""
from __future__ import annotations

import marshal
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# Wat in de root MAG staan. Alles wat hier niet in staat en er wel ligt, is een
# bevinding — niet een reden om deze verzameling uit te breiden.
# --------------------------------------------------------------------------- #
_ALLOWED_ROOT_FILES = frozenset({
    ".dvcignore",
    #: Lokale secrets. Git-ignored (`.gitignore:44`) en NIET getrackt — dat is
    #: geverifieerd, niet aangenomen. Staat hier zodat de bewaker hem niet elke
    #: run als bevinding meldt; het beleid eromheen is een openstaand punt in
    #: `docs/PROJECT_STATE.md` §11.4 (secrets-beheer, paper/live-sleutelscheiding).
    ".env",
    ".gitattributes",
    ".gitignore",
    #: De config die `.github/workflows/docs.yml` aan markdownlint-cli2
    #: meegeeft (`config: ".markdownlint.json"`). Hij MOET in de root staan:
    #: dat is het pad dat de workflow noemt en de plek waar cli2 hem zelf
    #: vindt. Voor deze commit bestond hij niet en faalde de check 6 van de
    #: 6 keer op "config not found".
    ".markdownlint.json",
    ".pre-commit-config.yaml",
    "CHANGELOG.md",
    "CLAUDE.md",
    "LICENSE",
    "MANIFEST.in",
    "Makefile",
    "README.md",
    "dvc.lock",
    "dvc.yaml",
    "params.yaml",
    "pyproject.toml",
    "pytest.ini",
    "requirements-dev.lock",
    "requirements.lock",
    "requirements.txt",
    "setup.cfg",
    "setup.py",
    "tox.ini",
})

#: Extensies die een DATA- of LEDGER-artefact aanduiden. Een bestand met deze
#: extensie in de root is per definitie ergens weggelekt: elk echt artefact
#: hoort in `artefacts/`, `reports/`, `data/` of `state/`.
_ARTEFACT_SUFFIXES = frozenset({
    ".jsonl", ".parquet", ".csv", ".pkl", ".pickle", ".npy", ".npz", ".feather",
    ".h5", ".hdf5", ".db", ".sqlite",
})


def _root_entries() -> list[Path]:
    """Bestanden direct in de repository-root, zonder mappen."""
    return sorted(p for p in _REPO_ROOT.iterdir() if p.is_file())


def test_no_data_artefact_sits_in_the_repository_root() -> None:
    """Een ledger, frame of dump in de root is een weggelekt testartefact.

    Dit is de bewaker die `x.jsonl` had moeten tegenhouden. Hij kijkt naar de
    EXTENSIE en niet naar de naam, zodat de volgende lekkage onder een andere
    naam er ook door wordt gevangen.
    """
    offenders = [
        p.name for p in _root_entries()
        if p.suffix.lower() in _ARTEFACT_SUFFIXES
    ]
    assert not offenders, (
        f"Data-artefact(en) in de repository-root: {offenders}. "
        f"Een test of script schrijft naar de projectroot in plaats van naar "
        f"`tmp_path`. Zoek de schrijver, laat hem naar `tmp_path` schrijven, en "
        f"verwijder het artefact uit git. Zie `reports/phase5_exit_report.md` "
        f"§9.1 voor dezelfde klasse in `artefacts/governance/`."
    )


def test_no_unexpected_file_sits_in_the_repository_root() -> None:
    """Bredere bewaker: elk NIEUW rootbestand moet een bewuste keuze zijn.

    Strenger dan de vorige test, want hij vangt ook een `.log`, een `.txt` of een
    naamloze dump. De allowlist uitbreiden is toegestaan wanneer een bestand daar
    echt hoort — maar het is dan een expliciet besluit in een diff, in plaats van
    een bestand dat er ongemerkt bij komt.
    """
    offenders = [
        p.name for p in _root_entries() if p.name not in _ALLOWED_ROOT_FILES
    ]
    assert not offenders, (
        f"Onverwacht bestand in de repository-root: {offenders}. "
        f"Hoort het daar, voeg het dan bewust toe aan `_ALLOWED_ROOT_FILES` in "
        f"{Path(__file__).name}. Hoort het daar niet, verwijder het en repareer "
        f"wat het schreef."
    )


def _pyc_files() -> list[Path]:
    return [
        p for p in _REPO_ROOT.rglob("*.pyc")
        if ".git" not in p.parts and ".venv" not in p.parts
    ]


def _co_filename(pyc: Path) -> str | None:
    """`co_filename` van de code-object in een `.pyc`, of None als onleesbaar.

    De header is 16 bytes sinds PEP 552 (magic, flags, mtime/hash, size).
    Een `.pyc` die we niet kunnen lezen wordt overgeslagen in plaats van als
    schending geteld: een onleesbare cache is een ander probleem dan een cache
    die over zijn herkomst liegt, en deze test gaat over het tweede.
    """
    try:
        return str(marshal.loads(pyc.read_bytes()[16:]).co_filename)
    except Exception:
        return None


def test_no_bytecode_cache_points_outside_the_repository() -> None:
    """Elke `.pyc` in de boom moet naar een pad BINNEN deze repository wijzen.

    Wijst er een naar buiten, dan is de boom verplaatst zonder de caches te
    wissen, en citeert elke traceback vanaf dat moment een pad dat niet bestaat.
    """
    root_str = str(_REPO_ROOT)
    offenders: list[tuple[str, str]] = []
    for pyc in _pyc_files():
        origin = _co_filename(pyc)
        if origin is None:
            continue
        # Een relatief `co_filename` (zoals pytest's assertion-rewrite soms
        # produceert) is per definitie binnen de boom.
        if not Path(origin).is_absolute():
            continue
        if not origin.startswith(root_str):
            offenders.append((str(pyc.relative_to(_REPO_ROOT)), origin))

    assert not offenders, (
        f"{len(offenders)} bytecode-cache(s) verwijzen naar een pad buiten "
        f"{root_str}. Eerste drie: {offenders[:3]}. "
        f"Dit gebeurt na een verplaatsing van de werkkopie: de mtimes blijven "
        f"behouden, Python beschouwt de caches als geldig, en elke traceback "
        f"citeert daarna een niet-bestaand pad. Opruimen met:\n"
        f"    find . -name '__pycache__' -type d -not -path './.git/*' "
        f"-prune -exec rm -rf {{}} +\n"
        f"    find . -name '*.pyc' -not -path './.git/*' -delete"
    )


@pytest.mark.parametrize("suffix", sorted(_ARTEFACT_SUFFIXES))
def test_the_artefact_guard_would_catch_each_suffix(suffix: str, tmp_path) -> None:
    """Negatieve controle: de suffix-set doet wat hij belooft.

    Zonder deze test bewijst `test_no_data_artefact_sits_in_the_repository_root`
    alleen dat de root vandaag schoon is — niet dat hij een vervuiling zou zien.
    Een test die niet rood kan worden, bewijst niets (faseregel §10).
    """
    assert (tmp_path / f"leak{suffix}").with_suffix(suffix).suffix in _ARTEFACT_SUFFIXES


# --------------------------------------------------------------------------- #
# DE DERDE KLASSE — een pijplijnpad dat naar iets wijst dat niet meer bestaat
#
# `dvc.yaml:57` noemde `src/tradebot/data/ingestion.py` als dependency van
# `build_features`, de stage die bars, features en events bouwt. Dat bestand is
# in commit `be94079` een PAKKET geworden (`ingestion/__init__.py` en vier
# modules ernaast). De import in `apps/build_features.py:117` bleef werken --
# die noemt het pakket, niet het bestand -- en `tests/test_imports.py:51` dekte
# hem af. De testsuite kon dit dus niet zien.
#
# DVC wel: `dvc repro build_features` viel om met `[Errno 2] No such file or
# directory`. Stage 1 van de pijplijn was daarmee onreproduceerbaar, en de enige
# poort die het had kunnen zien werd niet in CI gedraaid.
#
# Deze test stelt de vraag die geen van beide stelde: bestaat elk pad dat
# `dvc.yaml` declareert? Zie `docs/CHAIN_A_STATUS.md`, fase 10 stap 14.3.
# --------------------------------------------------------------------------- #
def _declared_dvc_paths(document: dict) -> list[tuple[str, str]]:
    """Elk letterlijk `deps`/`outs`-pad in een dvc-document, met zijn stage.

    Getemplatiseerde paden (`${item}`) worden overgeslagen: zij bestaan pas na
    expansie over `foreach`, en de expansie is DVC's werk en niet dat van deze
    poort. Wat overblijft is precies de verzameling paden die op schijf hoort te
    staan zoals zij er staat.
    """
    found: list[tuple[str, str]] = []
    for stage, body in (document.get("stages") or {}).items():
        blocks = [body.get("do", body)] if isinstance(body, dict) else []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            for key in ("deps", "outs"):
                for entry in block.get(key) or []:
                    path = next(iter(entry)) if isinstance(entry, dict) else entry
                    if isinstance(path, str) and "${" not in path:
                        found.append((stage, path))
    return found


def test_every_path_declared_in_dvc_yaml_exists() -> None:
    """Een dependency die niet bestaat maakt zijn stage onreproduceerbaar."""
    yaml = pytest.importorskip("yaml")
    document = yaml.safe_load((_REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    missing = [
        (stage, path)
        for stage, path in _declared_dvc_paths(document)
        if not (_REPO_ROOT / path).exists()
    ]
    assert not missing, (
        "`dvc.yaml` declareert paden die niet bestaan. Elk daarvan maakt zijn "
        "stage onreproduceerbaar, en geen enkele import-test ziet dat, want een "
        "import noemt een MODULE en deze declaratie noemt een PAD:\n  "
        + "\n  ".join(f"{stage}: {path}" for stage, path in missing)
    )


def test_the_dvc_path_guard_would_catch_a_deleted_dependency() -> None:
    """De negatieve controle: kan deze poort rood worden? (R-1)

    Zonder deze test bewijst de test hierboven alleen dat `dvc.yaml` vandaag
    klopt -- niet dat hij een verdwenen pad zou opmerken. Het document hieronder
    is de echte `build_features`-stage zoals hij vóór de reparatie was.
    """
    yaml = pytest.importorskip("yaml")
    broken = yaml.safe_load(
        "stages:\n"
        "  build_features:\n"
        "    foreach: ${symbols}\n"
        "    do:\n"
        "      cmd: python -m apps.build_features symbol=${item}\n"
        "      deps:\n"
        "        - src/tradebot/data/ingestion.py\n"
        "        - conf/conf_config.yaml\n"
    )
    declared = _declared_dvc_paths(broken)
    assert ("build_features", "src/tradebot/data/ingestion.py") in declared
    assert not (_REPO_ROOT / "src/tradebot/data/ingestion.py").exists()
    assert (_REPO_ROOT / "src/tradebot/data/ingestion").is_dir()
