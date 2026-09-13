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

Evaluated on the held-out temporal test block (the most recent 15% of transactions, which neither
early stopping nor tuning ever sees; n = **1,127** evaluation samples), with balanced class
weights (the default).

| Metric | Value |
|---|---|
| Top-1 accuracy | 0.502 |
| Top-3 accuracy | 0.807 |
| Top-5 accuracy | 0.907 |
| Top-10 accuracy | 0.988 |
| Balanced accuracy | 0.569 |

**Read accuracy (0.502) and balanced accuracy (0.569) together:** balanced class weights push
the model toward rare account codes, so the macro view sits *above* the headline. Without them
the order flips (0.581 vs 0.479, [`reports/class_weighting.json`](../reports/class_weighting.json)):
frequent codes are predicted well and rare ones are missed. Top-1 is a poor summary of this
system; the top-K ranking metrics are the ones aligned with its intended use. On this synthetic
data no classifier can exceed a Bayes-optimal top-1 of 0.678 or top-5 of 0.949 on the same rows
([`reports/ceiling.json`](../reports/ceiling.json)).

Model comparison on the same features and split
([`reports/model_comparison.json`](../reports/model_comparison.json)): on this synthetic data a
scaled logistic regression is a close competitor. It leads on top-1 (0.596 vs 0.581 unweighted)
and weighted F1, while XGBoost leads on top-5 (0.917 vs 0.889 unweighted; 0.907 vs 0.868 with
balanced weights). LightGBM trails both. Neither model was hyperparameter-searched for the
comparison.

## Training & Evaluation Data

- **Source:** **synthetic** data — 7,508 transactions across 80 account classes, produced by
  `scripts/generate_sample_data.py`. No proprietary or personal data is used or distributed.
- **Split:** temporal (chronological), **not** randomly shuffled — the earliest 70% train, the
  next 15% validate (early stopping, hyperparameter search), and the most recent 15% are a
  held-out test block read only for the reported metrics and the quality gate. This mirrors
  production (predict future transactions from past patterns), avoids the future-information
  leakage a random split would introduce, and keeps the stopping round from being chosen on the
  same rows that grade it.
- **Class weighting:** training rows carry balanced sample weights (`balanced_class_weights`,
  default on), so rare account codes weigh as much as frequent ones in the loss. On the test
  block this costs about 8pp top-1 and 1pp top-5 for +9pp balanced accuracy
  ([`reports/class_weighting.json`](../reports/class_weighting.json)); every manifest records
  per-class recall.
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

- **Quality gate:** promotion is blocked by floor thresholds (not targets): a retrain must beat
  the majority-class baseline and chance by `min_lift`, and may not lose more than
  `max_accuracy_drop` (default 0.01) accuracy against the promoted model. An equal-quality
  retrain is promoted; a regression (e.g. cold-start on sparse data) is not, and the previous
  `current` symlink stays in place.
- **Drift monitoring:** `POST /ops/drift` scores a batch by Population Stability Index against
  reference distributions frozen in the manifest at training time — six input features
  (`amount`, `desc_len`, `is_debit`, `has_reference`, `amount_bucket`, `weekday`) plus the
  predicted-class and confidence distributions. Thresholds are the standard `< 0.10` stable,
  `< 0.25` moderate, above that significant. No labels required, so it runs on live traffic.
  `POST /ops/confidence-histogram` remains available for the raw confidence shape.
  Two caveats: the output reference is the model's *own validation predictions*, not the true
  label distribution, since an imbalanced multi-class model systematically under-predicts rare
  classes; and the predicted-class PSI needs roughly ten samples per class before it says
  anything — smaller batches leave most classes empty and inflate the score.
  None of this replaces re-checking **balanced accuracy** (not just top-1) on freshly labeled
  recent data, or watching for changes in transaction-label vocabulary and entity naming.
- **Retraining:** the model trains in seconds, so scheduled retraining on recent labeled data
  is cheap; pair it with the quality gate above. Re-evaluate on a fresh temporal split each time
  rather than reusing an old validation window.
- **Reproduce:** `uv run python scripts/generate_sample_data.py` then `tc-train`; comparison via
  `uv run python scripts/compare_models.py`.
