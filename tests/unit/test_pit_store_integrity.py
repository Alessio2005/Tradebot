"""De gecertificeerde PIT-store staat in git en klopt met zijn manifest.

Phase 7/8, nagekomen op Stage A-1. Tot 2026-08-29 stond `data/pit_store/` onder
DVC met `data/pit_store.dvc` als pointer naar een remote. Die remote bestond
niet:

    $ dvc pull
    No remote provided and no default remote set.

    $ cat .dvc/config
    [core]
        no_scm = True

De store stond dus op precies één fysieke schijf — hetzelfde single point of
failure dat no-go 2 voor de git-historie aanwees, alleen dan voor de data waar
elk onderzoeksresultaat naar verwijst. En `research_gates.yml` viel er in CI op
om, bij de allereerste run die dit project ooit heeft gehad.

DE NIEUWE ROL VAN `data/pit_store.dvc`
======================================
De data staat nu in git. Het `.dvc`-bestand is daarmee geen pointer meer maar een
**onafhankelijk integriteitsmanifest**: het legt vast dat de store uit 228
bestanden van samen 2.340.376 bytes bestaat, met DVC-dirhash
`e04fff202fdc77d375a990fa99c43c3d.dir`.

Dit bestand maakt dat manifest LOAD-BEARING. Zonder deze test zou het een
achtergebleven artefact zijn dat niets bewaakt — precies de klasse die deze fase
overal elders opruimt.

WAT HET BEWAAKT, EN WAAROM DAT NIET THEORETISCH IS
==================================================
De dirhash breekt zodra ook maar één byte in de store verandert. Twee reële
manieren waarop dat kan gebeuren zonder dat iemand iets bedoelt:

* **Regeleinde-conversie.** 114 van de 228 bestanden zijn JSON met CRLF. Wie
  cloont met `core.autocrlf=true` — de default van menig Windows-installatie —
  krijgt ze omgezet terug. `.gitattributes` zet daarom `data/pit_store/** -text`;
  deze test is de controle dát dat werkt.
* **Een script dat de store "opschoont"** — herschrijft, hersorteert of
  opnieuw comprimeert. Parquet is niet byte-stabiel onder herschrijven.

Een gewijzigde store zonder gewijzigde `data_hashes.json` maakt elk artefact dat
naar die hashes verwijst stilzwijgend onjuist. Audit §26 verklaart zo'n artefact
INVALIDE; deze test zorgt dat je het merkt op de dag dat het gebeurt.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "data" / "pit_store"
MANIFEST = ROOT / "data" / "pit_store.dvc"


def _files() -> list[Path]:
    return sorted(p for p in STORE.rglob("*") if p.is_file())


def _dvc_dir_hash(root: Path) -> str:
    """Reken de DVC-dirhash na.

    DVC hasht een directory als de md5 van een JSON-lijst met per bestand zijn
    md5 en zijn relatieve pad, gesorteerd op pad. Deze functie is bewust een
    HERIMPLEMENTATIE en roept DVC niet aan: een controle die het gecontroleerde
    gereedschap gebruikt, valt met dat gereedschap om.
    """
    entries = [
        {"md5": hashlib.md5(f.read_bytes()).hexdigest(),
         "relpath": f.relative_to(root).as_posix()}
        for f in sorted(p for p in root.rglob("*") if p.is_file())
    ]
    entries.sort(key=lambda e: e["relpath"])
    blob = json.dumps(entries, sort_keys=True).encode("utf-8")
    return hashlib.md5(blob).hexdigest() + ".dir"


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert MANIFEST.is_file(), (
        f"{MANIFEST.relative_to(ROOT)} ontbreekt. Dat is het enige onafhankelijke "
        f"integriteitsrecord van de PIT-store; zonder hem is 'de store is "
        f"ongewijzigd' een bewering.")
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["outs"][0]


class TestTheStoreIsInGit:
    """De reden dat dit bestand bestaat."""

    def test_the_store_directory_exists(self) -> None:
        assert STORE.is_dir(), (
            "data/pit_store/ ontbreekt. Onder DVC was dit normaal — je deed "
            "`dvc pull`. Sinds Phase 7/8 staat de store in git, dus een "
            "ontbrekende map betekent dat er iets is verwijderd.")

    def test_the_store_is_not_gitignored(self) -> None:
        """De negatie in `.gitignore` moet NA `*.parquet` staan, en de map zelf
        mag niet uitgesloten zijn — git daalt niet af in een uitgesloten map."""
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "!data/pit_store/**" in text
        lines = [ln.strip() for ln in text.splitlines()]
        assert "data/pit_store/" not in lines, (
            "data/pit_store/ staat weer in .gitignore; git negeert de negatie "
            "dan, want hij daalt niet af in een uitgesloten map")

    def test_gitattributes_pins_the_store_as_binary(self) -> None:
        """Zonder `-text` breekt een clone met autocrlf=true de hashes."""
        path = ROOT / ".gitattributes"
        assert path.is_file(), (
            ".gitattributes ontbreekt. Zonder die regel converteert git de 114 "
            "JSON-bestanden op een machine met autocrlf=true, en breekt de "
            "dirhash op een plek waar niemand iets heeft gewijzigd.")
        assert "data/pit_store/** -text" in path.read_text(encoding="utf-8")


class TestTheStoreMatchesItsManifest:
    def test_the_file_count_matches(self, manifest: dict) -> None:
        assert len(_files()) == int(manifest["nfiles"])

    def test_the_total_size_matches(self, manifest: dict) -> None:
        total = sum(f.stat().st_size for f in _files())
        assert total == int(manifest["size"]), (
            f"de store is {total} bytes, het manifest zegt {manifest['size']}. "
            f"Er is data toegevoegd, verwijderd of herschreven zonder dat "
            f"data/pit_store.dvc is bijgewerkt.")

    def test_the_dvc_directory_hash_matches(self, manifest: dict) -> None:
        """De beslissende toets: één byte verschil breekt hem.

        Slaagt dit niet terwijl niemand iets heeft gewijzigd, kijk dan eerst
        naar `git config core.autocrlf` en naar `.gitattributes`.
        """
        computed = _dvc_dir_hash(STORE)
        assert computed == manifest["md5"], (
            f"DVC-dirhash wijkt af.\n"
            f"  berekend  : {computed}\n"
            f"  manifest  : {manifest['md5']}\n"
            f"De gecertificeerde store is niet meer bit-identiek aan wat de 18 "
            f"data_hashes in artefacts/governance/data_hashes.json beschrijven. "
            f"Elk artefact dat daarnaar verwijst, is daarmee onjuist geworden.")


class TestTheCheckCanGoRed:
    """Een integriteitscontrole die nooit rood is geweest, bewijst niets."""

    def test_a_changed_byte_breaks_the_hash(self, tmp_path: Path) -> None:
        """Op een KOPIE, zodat de echte store niet wordt aangeraakt."""
        import shutil

        copy = tmp_path / "store"
        shutil.copytree(STORE, copy)
        assert _dvc_dir_hash(copy) == _dvc_dir_hash(STORE), (
            "de kopie wijkt al af van het origineel; de toets hieronder meet "
            "dan iets anders dan hij beweert")

        victim = sorted(p for p in copy.rglob("*.json"))[0]
        raw = victim.read_bytes()
        victim.write_bytes(raw.replace(b"\r\n", b"\n"))    # exact het CRLF-scenario
        assert _dvc_dir_hash(copy) != _dvc_dir_hash(STORE), (
            "regeleinde-conversie in een JSON-bestand veranderde de dirhash "
            "NIET; deze controle vangt dan precies het scenario niet waarvoor "
            "hij bestaat")

    def test_a_removed_file_breaks_the_count(self, tmp_path: Path) -> None:
        import shutil

        copy = tmp_path / "store"
        shutil.copytree(STORE, copy)
        sorted(p for p in copy.rglob("*") if p.is_file())[0].unlink()
        n = len([p for p in copy.rglob("*") if p.is_file()])
        assert n != len(_files())
