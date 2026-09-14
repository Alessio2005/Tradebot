# PHASE 0 - FALLBACK REGISTER (stille degradatie)

**Gegenereerd:** 2026-08-22T17:42:44+00:00  
**Generator:** `scripts/audit_fallbacks.py`  
**Scope:** `src`  
**Bindend brondocument:** `docs/ARCHITECTUUR_AUDIT_2026-08-22.md` sectie 5.2

Elke regel hieronder is een codepad waarin een fout wordt opgevangen en het
systeem doorgaat met een naievere benadering, zonder dat de operator dit merkt.

---

## Samenvatting

| Categorie | Aantal | Betekenis |
|---|---:|---|
| `SWALLOWED_EXCEPT` | 35 | specifieke except die de fout inslikt |
| **TOTAAL** | **35** | |

**Blokkerend (breekt de build): 0** - **advies (geregistreerd, beoordeeld): 35**

Blokkerend zijn exact de condities uit exit criterium 1 plus deliverable 8:
`try/except ImportError`, bare `except:`, `except Exception` zonder re-raise, en
`warnings.warn` gevolgd door een degraded return. `SWALLOWED_EXCEPT` - een nauwe,
expliciet benoemde exceptie die niet her-raist - is advies: een `except LinAlgError`
die een bijna-singuliere covariantiematrix regulariseert is numerieke hygiene, geen
stille degradatie van een model naar een naievere benadering. Elke advies-bevinding
is stuk voor stuk beoordeeld; wie wel een modelwissel bleek, is gerepareerd.

De audit noemt in sectie 5.2 *12 locaties*. Deze scan meet **35**.
De audittelling was een steekproef; deze AST-scan is uitputtend.

---

## Bevindingen per categorie

### SWALLOWED_EXCEPT (35) - ADVIES

| Bestand:regel | Functie | Vangt | Gedegradeerd model | Naief alternatief |
|---|---|---|---|---|
| `src/tradebot/__init__.py:28` | `(module)` | `PackageNotFoundError` | `__version__: str = version('tradebot')` | `__version__ = '0.4.0'` |
| `src/tradebot/compliance/circuit_log.py:130` | `CircuitLog.read_all` | `JSONDecodeError, TypeError` | `d = json.loads(line); entries.append(CircuitLogEntry(**d))` | `logger.warning('CircuitLog: skipped malformed line.')` |
| `src/tradebot/data/crypto.py:352` | `ParquetStorage.prune_old_data` | `ValueError` | `if int(year_dir.name) < cutoff_date.year - 1: shutil.rmtree(year_dir) deleted_count += ...` | `continue` |
| `src/tradebot/data/crypto.py:368` | `ParquetStorage.prune_old_data` | `ValueError, OSError` | `date_part = pq_file.stem[-7:]; file_date = datetime.strptime(date_part, '%Y-%m').replace(tzinfo=timezone.utc); if file_date < cutoff_date.replace(day=1): pq_...` | `continue` |
| `src/tradebot/data/crypto_macro.py:156` | `CryptoMacroFetcher.calculate_macro_features` | `AttributeError, TypeError` | `ecdf = funding.rolling('30D', min_periods=10).rank(pct=True)` | `ecdf = funding.rolling('30D', min_periods=10).apply(lambda w: w.rank(pct=True).iloc[-1]...` |
| `src/tradebot/data/equity_universe.py:65` | `build_universe` | `ConnectionError` | `_, failures = stooq.fetch_universe([to_stooq_symbol(t) for t in tickers], market='equit...; used = 'stooq'` | `if source == 'stooq': raise; used = 'yfinance'` |
| `src/tradebot/data/perp_feed.py:66` | `fetch_universe` | `TypeError, ValueError` | `launch_ms = int(s.get('launchTime', 0) or 0)` | `launch_ms = 0` |
| `src/tradebot/data/sources/base.py:78` | `http_get_text` | `RequestException` | `resp = requests.get(url, headers=headers or _UA, timeout=timeout); resp.raise_for_status(); return resp.text` | `if attempt == 4: raise; time.sleep(2.0 ** attempt)` |
| `src/tradebot/data/sources/wiki_constituents.py:92` | `fetch_membership_events` | `ValueError` | `added_col = _col(changes, 'added', 'ticker'); removed_col = _col(changes, 'removed', 'ticker')` | `added_col = _col(changes, 'added'); removed_col = _col(changes, 'removed')` |
| `src/tradebot/data/sources/yfinance_backup.py:80` | `fetch_universe_yf` | `OSError, ValueError, KeyError, TypeError, RuntimeError` | `raw = yf.download(tickers=chunk, period='max', interval='1d', group_by='ticker', auto_a...` | `for s in chunk: failures[yf_map[s]] = f'chunk download failed: {exc}'; continue` |
| `src/tradebot/execution/simulator.py:344` | `LOBSimulator.simulate_fills` | `KeyError` | `entry_loc = bars_df.index.get_loc(order.entry_bar_ts)` | `entry_loc = bars_df.index.searchsorted(order.entry_bar_ts)` |
| `src/tradebot/features/_ta_kernels.py:595` | `get_optimal_d` | `ValueError, LinAlgError` | `schwert_max = int(np.ceil(12.0 * (len(clean_vals) / 100.0) ** 0.25)); adf_max_lag = max(1, min(schwert_max, len(clean_vals) // 4)); adf_p: float = float(adfu...` | `continue` |
| `src/tradebot/features/fracdiff.py:194` | `min_frac_diff.adf_p` | `ValueError, LinAlgError` | `stat, p, *_ = adfuller(y, maxlag=10, regression='c', autolag=None); return (float(p), len(y))` | `return (1.0, len(y))` |
| `src/tradebot/features/stationarity_gate.py:124` | `check_feature_stationarity` | `ValueError, LinAlgError` | `_, p_value, *_ = adfuller(series.astype(np.float64), maxlag=maxlag, regression='c', aut...` | `logger.debug('ADF failed on %s: %s', col, exc); report.skipped.append(col); continue` |
| `src/tradebot/live/circuit_breaker.py:298` | `CircuitBreaker._check_prior_trips` | `JSONDecodeError, KeyError` | `entry = json.loads(line.strip()); if not entry.get('acknowledged', True): trip_ts = pd.Timestamp(entry['ts']) if trip_ts ...` | `continue` |
| `src/tradebot/live/engine.py:310` | `LiveEngine.run` | `TimeoutError` | `event = await asyncio.wait_for(self._queue.get(), timeout=1.0)` | `now = pd.Timestamp.now(tz='UTC'); self._cb.check(now); if self._cb.is_active: await self._halt() break; continue` |
| `src/tradebot/live/exchange_status.py:63` | `monitor_exchange_status` | `CancelledError` | `async with session.get(BYBIT_STATUS_URL, timeout=aiohttp.ClientTimeout(total=10)) as re...` | `logger.info('Exchange status monitor cancelled.'); return None` |
| `src/tradebot/live/exchange_status.py:70` | `monitor_exchange_status` | `ClientError, TimeoutError, OSError` | `async with session.get(BYBIT_STATUS_URL, timeout=aiohttp.ClientTimeout(total=10)) as re...` | `logger.error('Exchange status check failed: %s', exc)` |
| `src/tradebot/live/execution_controller.py:203` | `ExecutionController.size_orders` | `ValueError` | `self._check_position_limits(symbol, qty_base, notional, current_notionals)` | `logger.error('ExecutionController: %s', exc); continue` |
| `src/tradebot/live/execution_controller.py:211` | `ExecutionController.size_orders` | `ValueError` | `self._fat_finger_check(symbol, qty_base, price_for_ff)` | `logger.error('ExecutionController: %s', exc); continue` |
| `src/tradebot/live/judge_gate.py:171` | `build_judge_gates` | `FileNotFoundError` | `sym_gates[side] = JudgeGate(sym, side, judge_dir, artefacts_dir)` | `logger.warning('JudgeGate: skipping %s/%s — %s', sym, side, exc)` |
| `src/tradebot/live/signal_runner.py:124` | `SignalRunner.predict` | `TypeError` | `result = sig.predict(features, bar_ts=bar_ts)` | `result = sig.predict(features)` |
| `src/tradebot/live/signal_runner.py:253` | `SignalRunner.predict_on_event` | `TypeError` | `result = predict_fn(features, bar_ts=bar_ts)` | `result = predict_fn(features)` |
| `src/tradebot/oms/audit_log.py:120` | `AuditLog.read_as_dataframe` | `JSONDecodeError` | `records.append(json.loads(line))` | `logger.warning('AuditLog: skipped malformed line.')` |
| `src/tradebot/oms/audit_log.py:164` | `AuditLog._load_last_hash` | `JSONDecodeError` | `entry = json.loads(line); if 'hash' in entry: last_hash = entry['hash']` | `continue` |
| `src/tradebot/oms/audit_log.py:166` | `AuditLog._load_last_hash` | `OSError` | `with open(self._path, encoding='utf-8') as fh: for line in fh: line = line.strip() if l...` | `pass (fout genegeerd)` |
| `src/tradebot/registry/catalog.py:121` | `ModelCatalog.query` | `JSONDecodeError, TypeError` | `d = json.loads(line); rec = ModelRecord.from_dict(d)` | `continue` |
| `src/tradebot/registry/hypothesis_ledger.py:165` | `HypothesisLedger.merge_staging` | `OSError` | `staging_path.unlink()` | `staging_path.write_text('[]', encoding='utf-8')` |
| `src/tradebot/train/ensemble.py:369` | `ContextualBanditEnsemble._sample_theta_raw` | `LinAlgError` | `sampled = np.random.multivariate_normal(np.asarray(theta_hat_k, dtype=np.float64), cov)` | `self._singular_cov_count = getattr(self, '_singular_cov_count', 0) + 1; if self._singular_cov_count % 10 == 0: logger.warning('Thompson: %d singular-cov fall...` |
| `src/tradebot/train/ensemble.py:565` | `ContextualBanditEnsemble._pre_scale_once._scale` | `RuntimeError` | `aligned = ref_model._align_input_custom(arr, target_feats, timeframe)` | `return raw` |
| `src/tradebot/train/ensemble.py:878` | `ContextualBanditEnsemble._maybe_resymmetrize` | `LinAlgError` | `self.B_inv[k] = np.linalg.inv(self.B[k])` | `regularization = 0.0001 * np.eye(d); self.B[k] += regularization; self.B_inv[k] = np.linalg.inv(self.B[k])` |
| `src/tradebot/train/ensemble.py:916` | `ContextualBanditEnsemble._maybe_cholesky_recompute` | `LinAlgError` | `L = np.linalg.cholesky(B_k + jitter * I_d); Y = np.linalg.solve(L, I_d); B_inv_k = np.linalg.solve(L.T, Y); break` | `jitter *= 10.0` |
| `src/tradebot/train/meta_train.py:116` | `train_judge` | `ValueError` | `X_judge = build_judge_features(X_primary, oos_probs)` | `logger.error('[%s/%s] Judge feature build mislukt: %s', sym, side, exc); return None` |
| `src/tradebot/train/thompson.py:91` | `LedoitWolfThompsonSampler.sample_theta` | `LinAlgError` | `L = np.linalg.cholesky(cov + 1e-06 * np.eye(d))` | `evals, evecs = np.linalg.eigh(cov); evals_clipped = np.maximum(evals, 1e-06); cov_pd = evecs @ np.diag(evals_clipped) @ evecs.T; try: L = np.linalg.cholesky(...` |
| `src/tradebot/volatility/yang_zhang.py:70` | `get_yang_zhang_volatility` | `AttributeError, TypeError` | `if isinstance(df.index, pd.DatetimeIndex): freq = getattr(df.index, 'freq', None) if fr...` | `logger.debug('Yang-Zhang intraday guard inconclusive: %s', exc)` |

