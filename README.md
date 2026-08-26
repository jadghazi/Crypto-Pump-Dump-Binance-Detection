# Detection of Cryptocurrency Pump-and-Dump Schemes

**A research extension of *"The Doge of Wall Street"* (La Morgia et al., 2021) — a stricter, leakage-resistant re-benchmark of pump-and-dump classifiers on Binance market microstructure data.**

📄 **[Read the full report](./Sect1_G6_Report.pdf)**

> MSBA 315 – Machine Learning and Predictive Analytics, Spring 2026 · Suliman S. Olayan School of Business, American University of Beirut

---

## Research Context

Pump-and-dump schemes are coordinated manipulations where organizers use Telegram/Discord groups to inflate a low-volume coin before selling at the peak, leaving retail traders with the losses.

La Morgia et al. (2021), *"The Doge of Wall Street: Analysis and Detection of Pump and Dump Cryptocurrency Manipulations,"* is the foundational study in this space: they identified 327 confirmed pump events from public Telegram channels on Binance and reported a best result of **F1 = 0.945** (Random Forest, 25s resolution) using plain 5-fold cross-validation.

This project re-benchmarks their setup and asks whether that result holds up under a methodology that closes the main evaluation gaps we identified in the original study.

## Research Questions

- **RQ1.** Which classifier family produces the strongest validation F1 under 1:1500+ class imbalance, and how does the ranking shift across 5s / 15s / 25s temporal resolutions?
- **RQ2.** Does SMOTE, native class weighting, or focal loss give the best precision-recall trade-off, and does the answer depend on the model family?
- **RQ3.** Does tuning the decision threshold on validation meaningfully beat the default 0.5 cutoff?
- **RQ4.** Under an event-disjoint train/validation/test protocol with a single audited test read and bootstrap confidence intervals, does the benchmark match or exceed the prior reported SOTA of F1 = 0.945?

## Identified Gaps in the Original Study

| Gap in La Morgia et al. (2021) | This project's fix |
|---|---|
| Plain 5-fold cross-validation — a pump spans many consecutive rows, so windows from the same event can leak into both train and test folds | **Event-disjoint splits** via `GroupShuffleSplit`, grouped on `pump_index`, with a separate held-out test set never touched during model selection |
| Headline metric is ROC-AUC, which saturates under extreme imbalance | **PR-AUC as the primary metric** (Davis & Goadrich, 2006) — we show ROC-AUC ≥ 0.998 even for a model with F1 = 0.21 |
| Point estimates only, no uncertainty quantification | **1,000-iteration event-level bootstrap** for 95% CIs on F1, precision, recall, and PR-AUC |
| No check for deployment-shift or split-dependent luck | **Chronological retrain** + **5-fold GroupKFold CV** on the training pool as independent robustness checks |
| No feature-leakage audit | Explicit audit for per-event normalization and lookahead leakage in the released features |

## What Was Benchmarked

Five classifiers × three temporal resolutions, same twelve Binance microstructure features as the original study:

- Gaussian Naive Bayes (baseline)
- Random Forest (class-weighted)
- XGBoost (`scale_pos_weight`)
- Linear SVM (calibrated, balanced)
- DNN with focal loss (γ=2, α=0.25)

## Results

| Model | Resolution | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| **Random Forest** (val-tuned τ=0.66) | 15s | 0.930 | 0.841 | **0.883** | 0.953 |

- **95% bootstrap CI on test F1: [0.818, 0.943]** — brackets the prior SOTA of 0.945; the bootstrap probability of beating 0.945 outright is only 2.2%
- **Chronological retrain F1: 0.906** · **GroupKFold CV F1: 0.913 ± 0.035** (25s) — both consistent with the main result, indicating it isn't an artifact of one favorable split
- Every model hit **ROC-AUC ≥ 0.998**, including Naive Bayes (F1 ≈ 0.21) — direct evidence that ROC-AUC is the wrong headline metric under this imbalance
- `std_rush_order`, `avg_rush_order`, and `std_trades` jointly account for ~74% of Random Forest's feature importance; time-of-day features contribute <2.5% despite visible clustering, meaning the model is learning microstructure, not a time-of-day heuristic

**Bottom line:** performance lands at **statistical parity** with La Morgia et al. (2021), not a clear improvement in raw score — but it's a materially more trustworthy number, obtained under a protocol designed to remove the leakage and metric-choice issues in the original benchmark.

## Methodology Summary

1. **Data:** SystemsLab-Sapienza pump-and-dump dataset (La Morgia et al., 2021) — 12 market-microstructure features, 3 temporal resolutions, Jan 2018–Jan 2021, 327 events
2. **Splits:** 208 training-fit / 53 validation / 66 test events, zero shared events across partitions
3. **Imbalance handling:** class weighting for tree ensembles/SVM, focal loss for the DNN, SMOTE evaluated separately for Naive Bayes
4. **Model selection:** validation-only, threshold swept from 0.01–0.99, single audited test read for the winning model
5. **Robustness checks:** GroupKFold CV, chronological retrain, 1,000-iteration event-level bootstrap

## Tech Stack

Python · scikit-learn · XGBoost · PyTorch/TensorFlow (focal-loss DNN) · pandas · matplotlib

Jad Ghazi 

Presented to Dr. Wael Khreich

## Primary Reference

La Morgia, M., Mei, A., Sassi, F., & Stefa, J. (2021). *The doge of Wall Street: Analysis and detection of pump and dump cryptocurrency manipulations.* arXiv preprint arXiv:2105.00733.
