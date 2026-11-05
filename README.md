# senescence-thresholds

Monte Carlo threshold analysis for senescence biomarkers. Dey Lab, Fred
Hutchinson Cancer Center.

## The problem

The lab measures biomarker intensities and wants to know at what level of
expression a cell counts as senescent. Turning a continuous intensity into a
binary senescent/not call requires a cutoff, and the cutoff is a free
parameter. Two defensible conventions on the same data can produce
percent-senescent figures tens of points apart.

This repo treats the cutoff as a parameter to be swept rather than a decision
to be made once, and reports how much of the final number is biology versus
thresholding choice.

Three reasons that matters here specifically:

1. **No single marker defines senescence.** The SenNet Biomarkers Working
   Group recommends panels rather than any one stain, and offers no single
   consensus rule for combining them. So the combination logic is a free
   parameter too, and belongs in the same sensitivity analysis as the cutoff.

2. **Marker intensity is graded, not binary.** Ashraf, Fernandez and Spencer
   (*Nat Commun* 2023) showed by single-cell time-lapse that the staining
   intensity of p21, Lamin B1, 53BP1, LAMP1, IL-8 and cell size all scale with
   the *duration of cell-cycle withdrawal* rather than with senescence per se.
   Any binary call is a lossy projection of a continuous state. Quantifying
   how lossy is part of the output, not a caveat on it.

3. **The source paper did not do this.** Minot et al. (*Sci Rep* 14:23646,
   2024, Dey corresponding) scored senescence transcriptionally: SenMayo, 125
   genes, top 10% of cells called senescent. Stromal cells carried the signal
   (76% senescent in their 10x data; 10.4% vs 6.9% between CRC and health
   consortia, stromal *p* < 0.03, macrophage *p* < 10⁻⁴). p21 staining was
   detectable but too sparse to quantify. The intensity-threshold question is
   the open one.

## The three experiments

### 1. Threshold sensitivity (`threshold_sensitivity`)

Sweeps the cutoff rule across its plausible range and records
percent-senescent plus control separation at each point. Bootstraps the
negative control to put a confidence interval on the cutoff itself and
propagate it forward.

Three rule families, each parameterized by a single scalar `k` so they sweep on
a common axis:

| family | cutoff | range |
|---|---|---|
| `sd` | control mean + k · SD | k ∈ [1, 3] |
| `mad` | control median + k · 1.4826 · MAD | k ∈ [1, 3] |
| `percentile` | k-th percentile of control | k ∈ [90, 99] |

`mad` is included because immunofluorescence negative controls routinely carry
a few very bright cells (debris, mitotic figures, antibody aggregate) and those
few points inflate the SD enough to push a mean+2SD cutoff well above where it
belongs. Where `sd` and `mad` disagree across the sweep, that disagreement is a
reportable finding about the data, not a nuisance.

Every cutoff is calibrated on the **negative control only**. A threshold
derived partly from the positive control bakes the expected effect into the
measurement, and the separation you then report is partly an artifact of where
you chose to draw the line.

Headline output is `pct_pos_range`: how many percentage points the
positive-control figure moves across the whole cutoff grid. A marker spanning
30 points is not a marker you can quote a single percentage for.

### 2. Panel-rule search (`panel_rule_search`)

Monte Carlo over (combination rule) × (per-marker cutoff rule), scored against
the controls by Youden's J. Combination rules enumerated are AND, OR, and
m-of-n over every marker subset, deduplicated (AND ≡ n-of-n, OR ≡ 1-of-n).

Ranked by **median** J across cutoff draws, not best J. A rule with a high max
but a poor median works only at one hand-picked cutoff, and that is the result
that fails to replicate.

### 3. Power and noise (`power_curve`, `min_detectable_effect`)

Resamples from the control distribution under an assumed effect and asks how
many cells per well and replicates are needed to detect a given shift in
percent-senescent.

The simulated design is the nested one a wet lab actually runs: cells in wells,
wells on plates, with a per-plate multiplicative gain shared across all cells
on that plate (`plate_gain_sd`, default 0.18 in log space). That nested gain is
what eats the apparent power a naive per-cell calculation promises, and it is
the most common reason a percent-senescent figure fails to reproduce across
plates. Set it to 0 and the curve is optimistic in a way that will not survive
contact with real data.

This is the experiment to produce first. It constrains the experimental design
rather than describing data already collected.

## Setup

```bash
pip install -e ".[dev]"
pytest                                  # 35 tests
python -m senescence.cli demo           # full run on synthetic data
```

Synthetic data is generated from the panel config, so the whole pipeline runs
and is testable before any real sheet arrives.

## When the sheets arrive

Edit `config/panel.yaml` and nothing else. Marker names, intensity column
names, per-marker direction, control labels and grouping columns all live
there; no module in `src/` needs touching to swap in the real panel.

```bash
python -m senescence.cli ingest data/raw/sheet.xlsx   # validate first
python -m senescence.cli sweep  data/raw/sheet.xlsx
python -m senescence.cli panels data/raw/sheet.xlsx
python -m senescence.cli power  data/raw/sheet.xlsx --marker p16
```

Run `ingest` before anything else. It validates the sheet against the panel and
fails loudly on a missing marker column or a missing negative control, rather
than letting either propagate into a percent-senescent number that looks fine.

Two settings in `panel.yaml` are worth getting right, because both fail
silently:

- **`direction`** per marker. `up` for p16, p21, SA-β-gal, γH2AX; `down` for
  LMNB1, Ki67, HMGB1. A `down` marker's senescent call is the lower tail. Set
  it backwards and the marker inverts while still producing plausible numbers.
- **`measurement.level`**. `per_cell` enables the full threshold analysis.
  `per_well` means the sheets are pre-summarized, single-cell cutoff sweeps are
  not meaningful, and only experiment 3 applies.

## Data handling

`data/raw/` is gitignored from the first commit, as are all `.xlsx` and `.csv`
outside tests. The sheets derive from human material in a clinical research
setting; even de-identified marker intensities belong in the lab's own storage
under its own access controls, not in a git history that cannot truly be purged
once pushed. Record provenance (who sent which file, when, which experiment)
here in the README rather than committing the files.

## Layout

```
config/panel.yaml          markers, directions, controls  <- the only file to edit
src/senescence/
  panel.py                 config loading + validation
  io.py                    Excel ingest, schema checks, log transform
  thresholds.py            cutoff rule families
  rules.py                 AND / OR / m-of-n panel logic
  metrics.py               Youden's J, AUC, percent-senescent
  montecarlo.py            the three experiments
  simulate.py              synthetic per-cell data
  cli.py                   entry points
tests/                     35 tests, including a null-calibration check
results/                   generated CSVs (gitignored)
```

## References

- Minot SS, Li N, Srinivasan H, … Dey N. Colorectal cancer-associated bacteria
  are broadly distributed in global microbiomes and drivers of precancerous
  change. *Sci Rep* 14:23646 (2024). doi:10.1038/s41598-024-70702-1
- Ashraf HM, Fernandez B, Spencer SL. The intensities of canonical senescence
  biomarkers integrate the duration of cell-cycle withdrawal. *Nat Commun*
  14:4527 (2023). doi:10.1038/s41467-023-40132-0
- SenNet Biomarkers Working Group. SenNet recommendations for detecting
  senescent cells in different tissues. *Nat Rev Mol Cell Biol* (2024).
  doi:10.1038/s41580-024-00738-8
- Single-cell fluorescence imaging reveals heterogeneity in senescence
  biomarkers and identifies rapamycin-responsive sub-populations. *Aging Cell*
  24:e70209 (2025). doi:10.1111/acel.70209
- Cougnoux A, Dalmasso G, … Bonnet R. Bacterial genotoxin colibactin promotes
  colon tumour growth by inducing a senescence-associated secretory phenotype.
  *Gut* 63:1932 (2014).
