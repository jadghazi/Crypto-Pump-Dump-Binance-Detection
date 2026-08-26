# Detection of Cryptocurrency Pump-and-Dump Schemes

A multi-resolution benchmark of five supervised classifiers for detecting Telegram-coordinated pump-and-dump schemes on Binance, using event-disjoint evaluation to avoid the leakage issues common in prior work.

📄 **[Read the full report](./Sect1_G6_Report.pdf)**

> MSBA 315 – Machine Learning and Predictive Analytics, Spring 2026 · Suliman S. Olayan School of Business, American University of Beirut

## Overview

Pump-and-dump schemes are coordinated manipulations where organizers use Telegram/Discord groups to inflate a low-volume coin before selling at the peak, leaving retail traders with the losses. This project builds a real-time detection framework on Binance market microstructure data, extending the benchmark methodology of La Morgia et al. (2021) with a stricter, leakage-resistant evaluation protocol.

**Five classifiers** were benchmarked — Naive Bayes, Random Forest, XGBoost, Linear SVM, and a focal-loss DNN — across **three temporal resolutions** (5s, 15s, 25s) on **327 confirmed pump events**.

## Key Contributions

- **Event-disjoint train/validation/test splits** (via `GroupShuffleSplit`) to prevent windows from the same pump event leaking across partitions — a risk in the plain 5-fold CV used by prior work
- **Validation-based threshold tuning** instead of a fixed 0.5 cutoff
- **PR-AUC-first evaluation**, demonstrating that ROC-AUC is uninformative under 1:1500+ class imbalance
- **Event-level bootstrap confidence intervals** (1,000 iterations) for statistically grounded comparison with prior state-of-the-art
- **Chronological robustness check** and **GroupKFold cross-validation** to rule out a lucky split
- **Feature-leakage audit** confirming no per-event normalization or lookahead bias

## Results

| Model | Resolution | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| **Random Forest** (val-tuned τ=0.66) | 15s | 0.930 | 0.841 | **0.883** | 0.953 |

- 95% bootstrap CI on test F1: **[0.818, 0.943]** — brackets the prior reported SOTA of 0.945 (La Morgia et al., 2021)
- Chronological retrain F1: 0.906 · GroupKFold CV F1: 0.913 ± 0.035 (25s)
- Every model reached ROC-AUC ≥ 0.998, including the weakest (Naive Bayes, F1 ≈ 0.21) — illustrating why ROC-AUC is the wrong headline metric here
- `std_rush_order`, `avg_rush_order`, and `std_trades` account for ~74% of Random Forest's feature importance

**Interpretation:** the result is best read as a methodological improvement rather than a raw score increase — matching prior SOTA performance under a much stricter, less leakage-prone protocol.

## Methodology Summary

1. **Data:** SystemsLab-Sapienza pump-and-dump dataset (La Morgia et al., 2021) — 12 market-microstructure features, 3 temporal resolutions, Jan 2018–Jan 2021
2. **Splits:** 208 training-fit / 53 validation / 66 test events, no shared events across partitions
3. **Imbalance handling:** class weighting for tree ensembles/SVM, focal loss for the DNN, SMOTE evaluated for Naive Bayes
4. **Model selection:** validation-only, with a single audited test read for the winning model
5. **Robustness checks:** GroupKFold CV, chronological retrain, event-level bootstrap CIs

## Tech Stack

Python · scikit-learn · XGBoost · PyTorch/TensorFlow (focal-loss DNN) · pandas · matplotlib

Jad Ghazi

Presented to Dr. Wael Khreich

## Reference

La Morgia, M., Mei, A., Sassi, F., & Stefa, J. (2021). The doge of Wall Street: Analysis and detection of pump and dump cryptocurrency manipulations. *arXiv preprint arXiv:2105.00733*.
