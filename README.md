# Predictive Alerting
  
Binary classifier that predicts whether an engine will fail within the next **H** cycles, given the last **W** cycles of sensor readings. Trained on the NASA CMAPSS FD001 benchmark.

---

## Problem Formulation

At each timestep *t*, the model receives a window of *W* consecutive sensor readings and outputs the probability that a failure will occur within the next *H* cycles. This is framed as binary classification:

```

label(t) = 1 if RUL(t + W) ≤ H

0 otherwise

```

`RUL` (Remaining Useful Life) is the number of cycles until the engine fails. A label of 1 means the engine is inside the danger zone — an alert should fire.

**H = 30, W = 30.** These values match the standard horizon used in the CMAPSS literature and give the alerting system a 30-cycle warning window, which is roughly 30% of the average engine lifetime in FD001.

---

## Dataset

**NASA CMAPSS FD001** — a widely used turbofan engine degradation benchmark.

- 100 engines, each run to failure under a single operating condition and single fault mode

- 14 sensor channels (after preprocessing removes constant/near-constant sensors)

- Labels are heavily imbalanced: most timesteps are in the "normal" region; only the final cycles of each engine are incidents

The dataset is downloaded automatically via `kagglehub` from [faresls/fd001-prepared-data](https://www.kaggle.com/datasets/faresls/fd001-prepared-data).

---

## Modeling Choices

### Why LSTM

Engine degradation is a sequential process, a sensor reading only makes sense in the context of the readings before it. LSTMs are a natural fit: they learn temporal patterns across the window without requiring manual feature engineering (e.g. rolling means, derivatives).

Alternatives considered:

| Model               | Thoughts                                                                 |
|---------------------|------------------------------------------------------------------------------|
| Logistic Regression | Cannot capture temporal structure without manual feature engineering         |
| RNN                 | Struggles with vanishing gradients and poor memory over longer sequences        |
| GRU                 | Architecturally similar to LSTM with comparable performance on this task     |
| 1D-CNN              | Captures local patterns well but misses long-range dependencies              |
| Transformer         | Stronger on long sequences but its juts an overkill for W=30 and adds training complexity   |

### Architecture

```

Input: (batch, W=30, n_sensors)

└─ LSTM (2 layers, hidden=64, dropout=0.3)

└─ last hidden state → Linear(64→32) → ReLU → Dropout(0.3) → Linear(32→1)

└─ raw logit (sigmoid applied at inference)

```

-  **2 layers:** enough capacity to model degradation curves without overfitting on a 100-engine dataset

-  **hidden=64:** keeps the model small relative to the dataset size

-  **dropout=0.3:** regularises both the LSTM stack and the classifier head

### Class Imbalance

Incidents are a minority class. Using standard BCE loss would push the model to predict "normal" almost always. To compensate, the loss is weighted by the inverse class ratio:

```python

pos_weight = (1 - y_train.mean()) / y_train.mean()

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

```

This makes each positive sample count proportionally more during training.

### Train / Test Split

Split is done with `GroupShuffleSplit` on `unit_nr`. This guarantees that no engine appears in both train and test.

---

## Evaluation Setup

### Metrics

| Metric           | Why It Matters Here                                          |
|------------------|--------------------------------------------------------------|
| Precision        | How often an alert is a real incident     									  |
| Recall           | How often real incidents are caught     										  |
| F1               | Harmonic balance of the two                                  |
| ROC-AUC          | Threshold-independent ranking quality                        |
| Confusion Matrix | Raw counts of TP / FP / FN / TN                              |

In an alerting context, **recall is usually more important than precision** since a missed failure is more costly than a false alarm. This informs threshold selection.

### Alert Threshold

The model outputs a probability in [0, 1]. A threshold converts this to a binary alert. The default is **0.5**, but this is rarely optimal. The evaluation includes a threshold sweep to show the precision/recall trade-off at different operating points, and identifies the threshold that maximises F1.

In production, the threshold would be chosen based on the relative cost of a missed incident vs. a false alarm.

---

## Results

Run `python main.py` to reproduce. Example output:

```
── Classification Report (threshold=0.50) ──
              precision    recall  f1-score   support
      Normal       0.98      0.97      0.97      2850
    Incident       0.88      0.89      0.89       620
    accuracy                           0.96      3470
   macro avg       0.93      0.93      0.93      3470
weighted avg       0.96      0.96      0.96      3470

ROC-AUC: 0.9905

Confusion Matrix:
[[2775   75]
 [  68  552]]

── Threshold Sweep ──
 Threshold | Precision | Recall |     F1
------------------------------------------
      0.20 |    0.8372 | 0.9210 | 0.8771
      0.30 |    0.8543 | 0.9081 | 0.8804
      0.40 |    0.8672 | 0.8952 | 0.8810
      0.50 |    0.8804 | 0.8903 | 0.8853
      0.60 |    0.8907 | 0.8806 | 0.8856
      0.70 |    0.9028 | 0.8694 | 0.8858
```

F1 changes very little for different thresholds, which means the model is confident. For an alerting use case, the right threshold depends on the cost of a false alarm vs a missed incident.
  
## Limitations

-  **Single operating condition.** FD001 has one fault mode and one operating regime. Real systems have multiple. The model would need retraining or a multi condition formulation

-  **Fixed horizon H.** The model answers one specific question: "will failure happen in the next 30 cycles?" A real system might need variable lead times or a regression head to estimate RUL directly.

-  **No online adaptation.** Once deployed, sensor distributions drift as engines age or operating conditions change. The model has no mechanism to update without full retraining.

-  **Threshold is static.** A fixed alert threshold ignores time-of-day, maintenance schedules, or operational context that would affect acceptable false alarm rates.

---

## Adapting to a Real Alerting System
 
-  **Adapt recall based on additioanl information.**

-  **Stream inference.** Replace batch DataLoader with a sliding buffer that processes each new sensor reading as it arrives.

-  **Monitor for drift.** Track for distribution shift. If the model's average confidence drifts over time, that's a signal to retrain. Ideally the model should be able to incorporate new incoming data into future predictions without requiring a full retrain
