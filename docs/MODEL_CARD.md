# Model Card — Transaction Classifier

A model card following the structure of Mitchell et al. (2019), "Model Cards for Model
Reporting." Numbers below are from the synthetic-data run reported in `reports/metrics.json`
and `reports/model_comparison.json`; see the [README](../README.md) for reproduction commands.

## Model Details

- **Task:** multi-class classification of financial transactions into French accounting
  (Plan Comptable Général) codes.
- **Architecture:** XGBoost gradient-boosted trees (500 estimators, depth 6, lr 0.05,
  `min_child_weight` 10, `gamma` 0.5), one-vs-rest over **80 account classes**.
- **Inputs / features:** three TF-IDF vectorizers (word n-grams on the `description` and
  `remarks` fields; character n-grams on combined text, capturing morphological variants
  like `COTISATION`/`COTISATIONS`) plus
  config-driven domain indicators (URSSAF/TVA deadlines, entity detection, SEPA fields),
  numeric amount features (magnitude buckets, round-amount and salary-range flags), and date
  features. Domain indicators are loaded from a YAML profile (`config/profiles/french_treasury.yaml`).
- **Output:** ranked **top-K** account codes with per-code confidence scores.
- **Versioning:** each training run emits a timestamped artifact directory (model,
  vectorizers, label encoder, checksum manifest); promotion is an atomic `current` symlink
  swap, hot-reloaded by the serving layer.
- **License / contact:** MIT · nlorber2211@gmail.com.

## Intended Use

- **Primary use:** decision support for bookkeepers/accountants — surface a short ranked
  list of likely account codes for a transaction so a human selects/confirms. The top-K
  design is deliberate: top-1 is moderate, but the correct code is in the top-3 ~83% of the
  time and top-5 ~91% of the time (synthetic), which fits a "suggest, human confirms" loop.
- **Intended users:** accounting/finance teams with a human in the loop.
- **Out of scope / misuse:** autonomous ledger posting without human review; tax or legal
  compliance determinations; non-French charts of accounts; any setting where a wrong code
  carries unreviewed financial/legal consequence. The model produces suggestions, not
  authoritative classifications.

## Factors

- **Account-code frequency:** the 80-class distribution is long-tailed. Rare codes
  (<~50 training samples) are materially harder; this is the dominant performance factor.
- **Transaction-label quality:** accuracy depends on consistent, structured labels
  (e.g. `URSSAF COTISATIONS`, `PRLV SEPA CPY:FR123`). Free-form or inconsistent labels degrade
  the TF-IDF and domain signals.
- **Temporal drift:** label conventions and entity names change over time; see Caveats.

## Metrics

Evaluated on the temporal validation split (the most recent transactions, `train_ratio` 0.80;
n = **1,502** evaluation samples).

| Metric | Value |
|---|---|
| Top-1 accuracy | 0.584 |
| Top-3 accuracy | 0.830 |
| Top-5 accuracy | 0.909 |
| Top-10 accuracy | 0.987 |
| Balanced accuracy | 0.488 |

**Read the gap between accuracy (0.584) and balanced accuracy (0.488) carefully:** the macro
view is weaker than the headline because minority classes underperform. Top-1 is a poor
summary of this system; the top-K ranking metrics are the ones aligned with its intended use.

Model comparison (same hyperparameter style): XGBoost (balanced acc 0.488, F1-weighted 0.553)
beats LightGBM (0.405 / 0.515); logistic regression is not competitive (0.012 / 0.016) — the
80-class sparse-feature problem needs tree-based feature interactions.

## Training & Evaluation Data

- **Source:** **synthetic** data — 7,508 transactions across 80 account classes, produced by
  `scripts/generate_sample_data.py`. No proprietary or personal data is used or distributed.
- **Split:** temporal (chronological), **not** randomly shuffled — the earliest 80% train,
  the most recent 20% validate. This mirrors production (predict future transactions from past
  patterns) and avoids the future-information leakage a random split would introduce.
- **Known synthetic-vs-real gap:** the generator uses uniform entity distribution and random
  label templates, which removes the client-specific seasonal and entity patterns that the
  domain/date features were designed to exploit. On the synthetic set those feature families
  show marginal or negative lift. **Treat the synthetic metrics above as a lower bound**;
  on real client data with seasonal patterns and consistent entity naming, top-1 accuracy is
  expected to be substantially higher.

## Limitations & Ethical Considerations

- **Long-tail weakness:** rare account codes are predicted least reliably (balanced accuracy
  0.488). Do not rely on the model for unusual or low-frequency codes without review.
- **Domain & locale bound:** trained for the French PCG and French-language transaction
  conventions; it does not transfer to other accounting standards or languages.
- **Confidence is not calibration:** reported confidences are softmax-style scores, not
  guaranteed calibrated probabilities; a high score is not a correctness guarantee.
- **Human accountability:** because outputs feed financial records, a human must remain
  responsible for the final code. The system is an assistant, not a decision-maker.

## Caveats, Maintenance & Drift

- **Quality gate:** promotion is blocked by floor thresholds (not targets) so a
  catastrophically bad retrain (e.g. cold-start on sparse data) never replaces a good model;
  on failure the previous `current` symlink stays in place.
- **Drift monitoring:** the serving layer exposes `POST /ops/confidence-histogram` to track
  the confidence distribution over time — a leftward shift signals distribution drift. Re-check
  **balanced accuracy** (not just top-1) on freshly labeled recent data, and watch for changes
  in transaction-label vocabulary and entity naming.
- **Retraining:** the model trains in seconds, so scheduled retraining on recent labeled data
  is cheap; pair it with the quality gate above. Re-evaluate on a fresh temporal split each time
  rather than reusing an old validation window.
- **Reproduce:** `uv run python scripts/generate_sample_data.py` then `tc-train`; comparison via
  `uv run python scripts/compare_models.py`.
