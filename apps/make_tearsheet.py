# apps/make_tearsheet.py
"""Stage 4.5 — Generate backtest tearsheet (JSON + optional HTML).

Entry-point: ``tb-make-tearsheet`` (defined in pyproject.toml).

Hydra config: conf/make_tearsheet.yaml
  result_path: artefacts/portfolio/result.joblib
  output_dir: reports/tearsheets/
  study_name: null    # null = auto-detect from result_path
  n_trials: 1         # for Deflated Sharpe
  bars_per_year: 8760 # 365*24 hourly
  render_html: false  # true = requires matplotlib + jinja2
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import hydra
import joblib
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="make_tearsheet", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    from tradebot.reporting import save_tearsheet, print_tearsheet

    result_path = Path(cfg.result_path)
    if not result_path.exists():
        raise FileNotFoundError(f"result_path not found: {result_path}")

    result = joblib.load(result_path)
    study_name = cfg.get("study_name") or result_path.stem
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{study_name}_tearsheet.json"
    save_tearsheet(
        result,
        output_path=json_path,
        study_name=study_name,
        bars_per_year=int(cfg.get("bars_per_year", 8760)),
        n_trials=int(cfg.get("n_trials", 1)),
    )

    print_tearsheet(result, study_name=study_name)

    render_html = bool(cfg.get("render_html", False))
    if render_html:
        try:
            _render_html(json_path, output_dir, study_name)
        except ImportError as exc:
            logger.warning("HTML rendering skipped (missing deps): %s", exc)


def _render_html(json_path: Path, output_dir: Path, study_name: str) -> None:
    """Render JSON tearsheet to HTML using jinja2 template."""
    import json
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError as exc:
        raise ImportError("pip install jinja2 for HTML tearsheet rendering") from exc

    template_dir = Path(__file__).parent / "templates"
    if not template_dir.exists():
        logger.warning("No templates/ dir found — skipping HTML render.")
        return

    env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=True)
    try:
        tmpl = env.get_template("tearsheet.html.j2")
    except Exception:
        logger.warning("tearsheet.html.j2 not found in %s", template_dir)
        return

    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)

    html = tmpl.render(**data)
    out = output_dir / f"{study_name}_tearsheet.html"
    out.write_text(html, encoding="utf-8")
    logger.info("HTML tearsheet: %s", out)


if __name__ == "__main__":
    main()
