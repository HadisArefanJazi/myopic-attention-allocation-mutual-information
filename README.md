# Myopic Attention Allocation: Mutual Information

Research implementation of myopic attention-allocation policies using mutual information as the attention measure.

The experiment uses the covariance update:

```text
P_t = (P_{t-1}^{-1} + A_t)^{-1}
G(P_t, P_{t-1}) = 0.5 * log |P_{t-1} P_t^{-1}| = 0.5 * log det(I + P_{t-1} A_t)
```

The original single-file script has been organized into modules while preserving the research assumptions and default numerical settings.

## Scenarios

- Scenario 1: trace cost, `Tr(P_t)`.
- Scenario 2: off-diagonal cost, `2 * |P_12|`.
- Scenario 3: Frobenius cost, `||P_t||_F`.

Total cost and regret remain trace-based, matching the original implementation.

## Installation

```bash
python -m pip install -e ".[dev]"
```

## Run

Default run:

```bash
python -m myopic_mi.main
```

The default grid step is `0.0005`, which is computationally expensive. For a quick smoke run:

```bash
python -m myopic_mi.main --horizon 2 --gamma-start 0.5 --gamma-stop 0.5 --grid-step 0.1 --lmax 0.5 --skip-figures
```

## Test

```bash
pytest
```

## Repository Structure

```text
myopic-attention-allocation-mutual-information/
├── src/
│   └── myopic_mi/
│       ├── __init__.py
│       ├── costs.py
│       ├── dynamics.py
│       ├── experiment.py
│       ├── main.py
│       ├── plotting.py
│       └── solver.py
├── tests/
│   ├── test_experiment.py
│   └── test_solver.py
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── pyproject.toml
```

## Limitations

- The exact grid can be slow for the default full experiment.
- Generated figures and MATLAB files are intentionally not committed.
- No new algorithms, datasets, or research claims are introduced by this refactor.
