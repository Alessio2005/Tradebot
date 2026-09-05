"""AD-23 definieert een domein. Een register dat een heropening belooft zonder
een bron uit dat domein te noemen, belooft iets waarvoor geen data bestaat --
en dat is precies de tegenstrijdigheid waaruit later een verboden hertest
ontstaat.

De poort werkt met een WHITELIST. Dat is bewust: een blacklist is nooit
volledig, en zij dwingt dit project om de uitgesloten ruimte te blijven
benoemen in het document dat haar afschaft.

De poort toetst KOLOM 4 van elke `F<n>` / `H<n>` / `DI-<n>`-rij -- de
heropeningsconditie -- ongeacht of die kolom het woord "heropen" bevat. Een
eerdere ontwerpversie triggerde op dat woord als sleutel; het woord staat
uitsluitend in de kolomkop van `FALSIFICATION_REGISTER.md` ("Heropening alleen
als...") en in GEEN van de twintig regels zelf, dus die versie sloeg elke echte
rij over. `test_scanner_flags_a_real_shaped_row_without_the_word_heropen`
bewijst dat de huidige poort dat gat niet heeft.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_domain_consistency.py"


def test_scanner_reports_clean_repository() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--strict"],
        capture_output=True, text=True, cwd=REPO, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_scanner_goes_red_on_a_reopening_without_a_domain_source(
    tmp_path: Path,
) -> None:
    """De negatieve controle. Zonder deze test toetst de poort haar eigen vorm."""
    register = tmp_path / "FALSIFICATION_REGISTER.md"
    register.write_text(
        "| F99 | verzonnen unit | bewijs | heropening zodra er een fijnere "
        "waarneming beschikbaar is |\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--strict", "--register", str(register)],
        capture_output=True, text=True, cwd=REPO, check=False,
    )
    assert result.returncode == 1
    assert "F99" in result.stdout


def test_scanner_flags_a_real_shaped_row_without_the_word_heropen(
    tmp_path: Path,
) -> None:
    """De echte controle. Elke rij in het echte register heeft de vorm
    `| F<n> | hypothese | bewijs | conditie |` -- vier kolommen, en de conditie
    zelf bevat het woord "heropen" NERGENS (dat woord staat alleen in de
    kolomkop). Een poort die op dat woord triggert, slaat elke echte rij over
    en meldt een vervuild register als schoon. Deze rij heeft precies die vorm
    en moet toch worden geraakt."""
    register = tmp_path / "FALSIFICATION_REGISTER.md"
    register.write_text(
        "| F97 | verzonnen unit | bewijs | alleen bij een geheel nieuwe "
        "asset-klasse |\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--strict", "--register", str(register)],
        capture_output=True, text=True, cwd=REPO, check=False,
    )
    assert result.returncode == 1
    assert "F97" in result.stdout


def test_a_reopening_that_names_a_domain_source_is_accepted(
    tmp_path: Path,
) -> None:
    """De poort mag geen legitieme heropening blokkeren; anders wordt zij
    genegeerd, en een genegeerde poort is geen poort."""
    register = tmp_path / "FALSIFICATION_REGISTER.md"
    register.write_text(
        "| F98 | verzonnen unit | bewijs | heropening bij meer historie in "
        "daily_ohlcv |\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--strict", "--register", str(register)],
        capture_output=True, text=True, cwd=REPO, check=False,
    )
    assert result.returncode == 0, result.stdout


def test_a_bare_refusal_is_accepted_without_a_domain_source(
    tmp_path: Path,
) -> None:
    """Een conditie die niets belooft ("nooit", "n.v.t.") kan ook niets buiten
    het domein beloven. F4, F9 en F12 in het echte register hebben deze vorm."""
    register = tmp_path / "FALSIFICATION_REGISTER.md"
    register.write_text(
        "| F96 | verzonnen unit | bewijs | nooit -- geen enkele voorwaarde "
        "hierboven weerlegt dit |\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--strict", "--register", str(register)],
        capture_output=True, text=True, cwd=REPO, check=False,
    )
    assert result.returncode == 0, result.stdout


def test_a_refusal_that_names_an_alternative_asset_class_is_still_flagged(
    tmp_path: Path,
) -> None:
    """De weigering-categorie is smal en mag geen sluiproute worden. Een rij die
    met "n.v.t." begint maar vervolgens wel degelijk een concreet
    heropeningspad buiten crypto noemt (zoals F8 in het echte register doet),
    is geen kale weigering en moet gewoon worden geraakt."""
    register = tmp_path / "FALSIFICATION_REGISTER.md"
    register.write_text(
        "| F95 | verzonnen unit | bewijs | n.v.t. binnen crypto; in equities "
        "is dit een ander, gedocumenteerd premium |\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--strict", "--register", str(register)],
        capture_output=True, text=True, cwd=REPO, check=False,
    )
    assert result.returncode == 1
    assert "F95" in result.stdout


def test_a_non_daily_resolution_in_conf_is_refused(tmp_path: Path) -> None:
    """B-1. Eén meetklok, en de poort werkt in BEIDE richtingen: elke declaratie
    die niet exact `1d` is, is een tweede meetklok in wording. Deze controle
    gebruikt daarom een grovere resolutie -- de poort hoeft niet te weten welke
    resoluties er buiten het domein bestaan, alleen welke erin zit."""
    conf = tmp_path / "feed.yaml"
    conf.write_text("bar_resolution: 1w\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--strict", "--conf-root", str(tmp_path)],
        capture_output=True, text=True, cwd=REPO, check=False,
    )
    assert result.returncode == 1
