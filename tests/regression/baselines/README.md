# Frozen regression baselines

`gk_volatility.npy` and `sequential_bootstrap_indices.npy` are the outputs that
`tests/regression/test_bitequivalence.py` compares the current code against.
They are produced by `tests/regression/_current.py` and written by

    python -m tests.regression.regenerate_baselines

They are committed (see the exception in `.gitignore`, DI-25), and they change
only together with an intentional, reviewed change of the algorithm behind them.

First committed on top of `adc6402` with Python 3.13.16, numpy 1.26.4,
pandas 2.3.3 and numba 0.67.0 from `requirements.lock`. This is a
golden master of the code as it was on that commit, not a value taken from a
reference implementation.
