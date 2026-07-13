# Project 20 · CNC Anomaly Sentinel — Statistical Z-Score Detection

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/ZScore_Anomaly_Detection/blob/main/20_ZScore_Anomaly_Detection.ipynb)

> *"Normal is not a feeling. Normal is a distribution. And distributions don't lie."*

---

## 🎯 Business Problem

Walter Shewhart published the three-sigma rule in 1924. A century later, CNC machines still fail in predictable, statistical ways — and most plants still rely on a human operator noticing the sound of chatter before a tool breaks.

This project asks a different question: **what if the process monitored itself?**

A high-speed machining center generates four continuous sensor signals — vibration, cutting force, tool temperature, and energy consumption. Under normal operation, each signal follows a near-Gaussian distribution with a well-defined mean and standard deviation. An anomaly is simply a moment when one or more signals exceed that boundary by a statistically meaningful margin. The mathematics are not new. The operational discipline required to apply them consistently is rarer than it should be.

Z-Score detection does not require labeled failure data. It does not require a training phase. It does not require a GPU. It requires one thing that most operations already have: **a process that has been running long enough to know what normal looks like.** When those conditions are met, this method is not a compromise — it is the right tool. Reaching for a more complex model would introduce opacity without adding detection power.

This notebook documents that argument from first principles, applies it to 1,634 machining cycles across three production shifts, and quantifies exactly where statistical detection succeeds — and where it does not.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Records | 1,634 machining cycles — three production shifts |
| Features | 4 continuous sensor signals |
| Target | No label in raw data — derived by engineering limits and Z-Score rule |
| Anomaly rate (engineering-validated) | 6.5% (106 / 1,634 cycles) |
| Anomaly rate (Z-Score detected) | 5.6% (91 / 1,634 cycles) |
| Source | CNC machining center cycle logs — spindle, force, thermal, and drive channels |

### Feature Table

| Feature | Type | Unit | Normal Range | Failure Signature |
|---|---|---|---|---|
| `vibration_mm_s` | float | mm/s | 3.2 – 5.2 | Chatter · Tool resonance · Bearing wear |
| `cutting_force_n` | float | N | 95 – 145 | Tool dulling · Material hardness · Chip clogging |
| `tool_temp_c` | float | °C | 48 – 68 | Thermal runaway · Dry cutting · Coolant failure |
| `energy_kwh` | float | kWh | 1.9 – 2.9 | Mechanical binding · Motor overload · Friction |

### Key EDA Findings

- **Right-tail extension in all sensors** — histograms with μ ± 3σ boundaries reveal a distinct cluster of high-value readings that separate from the normal distribution, visible before any detection logic is applied. However, a secondary cluster of borderline force readings (162–174 N) sits between the engineering limit (160 N) and the 3σ threshold (174.3 N) — real anomalies that Z-Score cannot reach.
- **High inter-sensor correlation in severe events** — sensors move together under abnormal conditions. An anomaly that elevates vibration tends to simultaneously elevate cutting force and temperature. Single-sensor borderline exceedances are the exception — and the primary source of missed detections.
- **Vibration is the dominant trigger** — it flags 100% of the Z-Score-detected cycles (91/91), confirming that spindle dynamics are the primary failure mode signature for the events the method catches. The 15 missed cycles show only isolated force elevation, without the vibration signature that Z-Score relies on.

---

## 🤖 Model

### Why Z-Score for This Problem

The decision to use Z-Score over a supervised classifier — or even Isolation Forest — is deliberate and worth explaining.

Supervised methods require labeled examples of failure. In most real plants, failure events are rare, inconsistently logged, and not available at the start of a monitoring program. Z-Score requires only the process baseline. It is also fully transparent: a maintenance technician can look at a Z-Score of 4.46 on the vibration channel and immediately understand both the severity of the deviation and which physical subsystem to investigate.

The three-sigma threshold encodes a specific statistical claim: **under a Gaussian distribution, a reading beyond ±3σ occurs by random chance in only 0.27% of cases.** If the empirical rate is significantly higher, the process is communicating a real signal. At 6.5% engineering-validated anomalies in this dataset — versus 0.27% expected by chance — the signal is unambiguous. Z-Score captures 91 of those 106 events (Recall = 0.858), missing the 15 borderline force exceedances that sit inside the statistical threshold but outside the engineering limit.

### Decision Rule

$$Z_i = \frac{x_i - \mu}{\sigma}$$

**Flag condition:** a cycle is anomalous if $\max_i |Z_i| > 3.0$ for any of the four sensors.

This OR-gate logic captures multi-mode failures: cases where no single sensor reaches 3σ independently but the combined pattern across sensors is outside the normal operating envelope.

### Statistical Validation

The D'Agostino K² normality test rejects normality on the **full dataset** — as expected, because the 5% anomaly records create heavy tails that inflate skewness and kurtosis. Running the same test on **normal-only cycles** restores approximate normality for all four sensors, confirming that the process baseline is genuinely Gaussian and the Z-Score assumption holds where it matters.

---

## 📈 Key Results

| Metric | Value | Operational Meaning |
|---|---|---|
| **Precision** | 1.000 | Zero false alarms — every Z-flagged cycle is a real event |
| **Recall** | 0.858 | 15 missed detections — borderline force anomalies below the 3σ threshold |
| **F1-Score** | 0.924 | Strong but not perfect — honest reflection of the statistical gap |
| **Accuracy** | 0.991 | 1,619 of 1,634 cycles correctly classified |
| **Anomaly rate (engineering-validated)** | 6.5% (106 / 1,634) | Far above the 0.27% chance rate — real process excursions |
| **Anomaly rate (Z-Score detected)** | 5.6% (91 / 1,634) | 15 borderline cycles fall in the gap between eng. limit and 3σ |
| **Threshold** | \|Z\| > 3.0 | Three-sigma, 99.73% statistical confidence |

**Confusion Matrix (vs. engineering-validated ground truth):**

|  | Predicted Normal | Predicted Anomaly |
|---|---|---|
| **Actually Normal** | 1,528 ✅ | 0 |
| **Actually Anomaly** | 15 ⚠ | 91 ✅ |

**Why Precision = 1.000 is structural, not lucky.** The engineering limits (vibration > 7.5 mm/s, force > 160 N, temp > 75°C, energy > 3.2 kWh) sit below the corresponding 3σ thresholds (7.79 / 174.26 / 82.55 / 3.58). Any cycle that crosses a 3σ boundary has already exceeded an engineering limit — Z-Score cannot produce false positives by construction in this operational window.

**Why Recall = 0.858 is the honest number.** The 15 missed cycles represent borderline tool-wear events where cutting force rises to 162–169 N (above the 160 N engineering limit) without triggering the vibration signature that Z-Score primarily responds to. These cycles cross the process limit but not the statistical threshold — a gap of approximately 5–14 N that the 3σ rule cannot bridge. Tightening the threshold to |Z| > 2.5 would catch them but would also generate false alarms across the normal operating population.

---

## 🔍 Sensor Contribution to Detection

| Sensor | Flags (\|Z\| > 3) | % of Detected (91) | Physical Interpretation |
|---|---|---|---|
| `vibration_mm_s` | **91** | 100.0% | Primary trigger — spindle dynamics dominate all detected events |
| `cutting_force_n` | 74 | 81.3% | Secondary — tool load correlation; present in severe events |
| `tool_temp_c` | 72 | 79.1% | Secondary — thermal signature; lags vibration by ~1–2 cycles |
| `energy_kwh` | 72 | 79.1% | Secondary — drive system response; correlates with force |

Vibration is the first sensor to cross the three-sigma boundary in all 91 detected events. Force, temperature, and energy provide corroborating signals in approximately 80% of cases — confirming multi-mode failure signatures. The 15 undetected cycles show only isolated force elevation (Z ≈ 2.4–2.9 on force channel alone) — below the detection threshold and without the vibration co-signature.

**Anomaly profile — mean sensor values by class (Z-Score detected anomalies vs. normal cycles):**

| Sensor | Normal μ | Anomaly μ | Ratio | Δ (in σ) |
|---|---|---|---|---|
| `vibration_mm_s` | 4.22 mm/s | 8.57 mm/s | 2.0× | +3.66σ |
| `cutting_force_n` | 120.16 N | 177.1 N | 1.5× | +3.16σ |
| `tool_temp_c` | 57.84 °C | 85.8 °C | 1.5× | +3.39σ |
| `energy_kwh` | 2.39 kWh | 3.72 kWh | 1.6× | +3.32σ |

---

## 🔭 Simulator — Three Reference Scenarios

The `detect_anomaly()` function evaluates new sensor readings against the process baseline in real time, returning a decision and the exact Z-Score per sensor.

| Scenario | Vibration | Force | Temp | Energy | Decision |
|---|---|---|---|---|---|
| **A — Stable** | 4.1 mm/s | 118.0 N | 57.5 °C | 2.35 kWh | ✅ Normal |
| **B — Borderline** | 7.2 mm/s | 145.0 N | 69.0 °C | 3.00 kWh | ✅ Normal (monitor) |
| **C — Critical** | 9.1 mm/s | 182.0 N | 88.0 °C | 3.90 kWh | ⚠ Anomaly |

Scenario B is operationally informative: vibration reaches Z = 2.65 — not yet an anomaly, but close enough to warrant monitoring. This is the kind of early warning signal that a binary alarm threshold (fixed engineering limit) would miss entirely.

---

## 🗂️ Repository Structure

```
ZScore_Anomaly_Detection/
├── 20_ZScore_Anomaly_Detection.ipynb   # Educational notebook (no outputs)
├── data_analysis.csv                   # 1,634-cycle dataset (complete)
├── requirements.txt
└── README.md
```

> 📦 **Full Project Pack** — complete 1,634-cycle dataset, notebook with full outputs,
> presentation deck (PPTX + PDF), and `app.py` real-time simulator available on
> [Gumroad](https://lozanolsa.gumroad.com).

---

## 🚀 How to Run

**Option 1 — Colab (recommended, no setup):**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/ZScore_Anomaly_Detection/blob/main/20_ZScore_Anomaly_Detection.ipynb)

**Option 2 — Local:**

```bash
git clone https://github.com/LozanoLsa/ZScore_Anomaly_Detection.git
cd ZScore-Anomaly-Detection-CNC
pip install -r requirements.txt
jupyter notebook 20_ZScore_Anomaly_Detection.ipynb
```

**Requirements:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`

---

## 💡 Key Learnings

1. **Model complexity should match problem complexity.** When the failure physics is well-understood and the process baseline is Gaussian, Z-Score detection is not a simplification — it is the correct solution. Introducing more complexity would trade interpretability for no gain in detection power.

2. **Unsupervised anomaly detection has a structural advantage in industrial settings.** Most facilities do not have clean, labeled failure logs. Z-Score requires only a baseline period of normal operation — a condition almost every running process can provide.

3. **The three-sigma boundary is a statistical claim, not a magic number.** Under a Gaussian distribution, \|Z\| > 3 occurs by chance in 0.27% of readings. When the engineering-validated anomaly rate is 6.5% — and Z-Score detects 5.6% — both numbers are telling a consistent story. The gap between them (0.9%) is not a model failure; it is the boundary condition where statistical and engineering limits diverge.

4. **Precision = 1.000 is structural.** In this operational window, every Z-Score flag corresponds to a real engineering limit violation by construction — the statistical thresholds sit above the process limits. Zero false alarms are not a lucky outcome; they reflect the geometry of the detection problem. Recall = 0.858 is the honest constraint: the 15 missed cycles live in the gap between where the process says "too much" and where the statistics agree.

5. **Vibration is the leading indicator — and the right one to instrument first.** Of the four sensors, vibration crosses the three-sigma boundary in 100% of the detected events. The 15 missed anomalies share one property: they show isolated force elevation without the spindle vibration signature. If a single sensor had to be selected for a lean monitoring program, vibration is the one.

6. **Statistical validation of the normality assumption closes the loop.** Running the D'Agostino K² test on normal-only cycles — not the full dataset — confirms that the Gaussian assumption holds for the process baseline. This is not a formality; it is the proof that the Z-Score threshold is meaningful. The full dataset fails the normality test precisely because the anomalous tail inflates skewness — which is itself confirmation that the method is working.

---

## 👤 Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning
GitHub: [LozanoLsa](https://github.com/LozanoLsa) · Gumroad: [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com)

*Turning Operations into Predictive Systems — Clone it. Fork it. Improve it.*
