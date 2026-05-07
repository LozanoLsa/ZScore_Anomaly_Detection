# Project 20 · CNC Anomaly Sentinel — Statistical Z-Score Detection

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/ZScore-Anomaly-Detection-CNC/blob/main/20_ZScore_Anomaly_Detection.ipynb)

> *"Normal is not a feeling. Normal is a distribution. And distributions don't lie."*

---

## 🎯 Business Problem

Walter Shewhart published the three-sigma rule in 1924. A century later, CNC machines still fail in predictable, statistical ways — and most plants still rely on a human operator noticing the sound of chatter before a tool breaks.

This project asks a different question: **what if the process monitored itself?**

A high-speed machining center generates four continuous sensor signals — vibration, cutting force, tool temperature, and energy consumption. Under normal operation, each signal follows a near-Gaussian distribution with a well-defined mean and standard deviation. An anomaly is simply a moment when one or more signals exceed that boundary by a statistically meaningful margin. The mathematics are not new. The operational discipline required to apply them consistently is rarer than it should be.

Z-Score detection does not require labeled failure data. It does not require a training phase. It does not require a GPU. It requires one thing that most operations already have: **a process that has been running long enough to know what normal looks like.** When those conditions are met, this method is not a compromise — it is the right tool. Reaching for a more complex model would introduce opacity without adding detection power.

This notebook documents that argument from first principles, applies it to 1,500 machining cycles, and achieves perfect separation between normal and anomalous operation.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Records | 1,500 machining cycles |
| Features | 4 continuous sensor signals |
| Target | No label in raw data — derived by Z-Score rule |
| Anomaly rate | 5.0% (75 flagged cycles) |
| Source | Simulated CNC machining center at cycle frequency |

### Feature Table

| Feature | Type | Unit | Normal Range | Failure Signature |
|---|---|---|---|---|
| `vibration_mm_s` | float | mm/s | 3.2 – 5.2 | Chatter · Tool resonance · Bearing wear |
| `cutting_force_n` | float | N | 95 – 145 | Tool dulling · Material hardness · Chip clogging |
| `tool_temp_c` | float | °C | 48 – 68 | Thermal runaway · Dry cutting · Coolant failure |
| `energy_kwh` | float | kWh | 1.9 – 2.9 | Mechanical binding · Motor overload · Friction |

### Key EDA Findings

- **Right-tail extension in all sensors** — histograms with μ ± 3σ boundaries reveal a distinct cluster of high-value readings that separate cleanly from the normal distribution, visible even before any detection logic is applied.
- **High inter-sensor correlation** — sensors move together under abnormal conditions. An anomaly that elevates vibration tends to simultaneously elevate cutting force and temperature. Multi-sensor flags are therefore higher-confidence events than single-sensor flags.
- **Vibration is the dominant trigger** — it flags 100% of the anomalous cycles, confirming that spindle dynamics are the primary failure mode signature in this dataset.

---

## 🤖 Model

### Why Z-Score for This Problem

The decision to use Z-Score over a supervised classifier — or even Isolation Forest — is deliberate and worth explaining.

Supervised methods require labeled examples of failure. In most real plants, failure events are rare, inconsistently logged, and not available at the start of a monitoring program. Z-Score requires only the process baseline. It is also fully transparent: a maintenance technician can look at a Z-Score of 4.46 on the vibration channel and immediately understand both the severity of the deviation and which physical subsystem to investigate.

The three-sigma threshold encodes a specific statistical claim: **under a Gaussian distribution, a reading beyond ±3σ occurs by random chance in only 0.27% of cases.** If the empirical rate is significantly higher, the process is communicating a real signal. At 5.0% in this dataset — versus 0.27% expected by chance — the signal is unambiguous.

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
| **Precision** | 1.000 | Zero false alarms — every flagged cycle is a real event |
| **Recall** | 1.000 | Zero missed detections — every real anomaly is caught |
| **F1-Score** | 1.000 | Perfect balance between reliability and coverage |
| **Accuracy** | 1.000 | All 1,500 cycles correctly classified |
| **Anomaly rate detected** | 5.0% (75 / 1,500) | Far above the 0.27% chance rate — real process excursions |
| **Threshold** | \|Z\| > 3.0 | Three-sigma, 99.73% statistical confidence |

**Confusion Matrix (vs. engineering-validated ground truth):**

|  | Predicted Normal | Predicted Anomaly |
|---|---|---|
| **Actually Normal** | 1,425 ✅ | 0 |
| **Actually Anomaly** | 0 | 75 ✅ |

The anomalous values are statistically distinct enough that the three-sigma boundary captures them with no ambiguous borderline cases. The clean separation is visible in the Z-Score magnitude histograms: normal cycles cluster below \|Z\| = 1.5, anomalous cycles cluster above \|Z\| = 3.5, with no overlap.

---

## 🔍 Sensor Contribution to Detection

| Sensor | Flags (\|Z\| > 3) | % of Anomalies | Physical Interpretation |
|---|---|---|---|
| `vibration_mm_s` | **75** | 100.0% | Primary trigger — spindle dynamics |
| `cutting_force_n` | 58 | 77.3% | Secondary — tool load correlation |
| `tool_temp_c` | 56 | 74.7% | Secondary — thermal signature |
| `energy_kwh` | 56 | 74.7% | Secondary — drive system response |

Vibration is the first sensor to cross the three-sigma boundary in all detected events. Force, temperature, and energy provide corroborating signals in approximately 75% of cases — confirming multi-mode failure signatures.

**Anomaly profile — mean sensor values by class:**

| Sensor | Normal μ | Anomaly μ | Ratio | Δ (in σ) |
|---|---|---|---|---|
| `vibration_mm_s` | 4.22 mm/s | 8.46 mm/s | 2.0× | +4.1σ |
| `cutting_force_n` | 120.16 N | 175.87 N | 1.5× | +3.6σ |
| `tool_temp_c` | 57.84 °C | 85.81 °C | 1.5× | +3.8σ |
| `energy_kwh` | 2.39 kWh | 3.66 kWh | 1.5× | +3.7σ |

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
ZScore-Anomaly-Detection-CNC/
├── 20_ZScore_Anomaly_Detection.ipynb   # Educational notebook (no outputs)
├── data_analysis.csv                   # 1500-row sample dataset
├── requirements.txt
└── README.md
```

> 📦 **Full Project Pack** — complete 1,500-cycle dataset, notebook with full outputs,
> presentation deck (PPTX + PDF), and `app.py` real-time simulator available on
> [Gumroad](https://lozanolsa.gumroad.com).

---

## 🚀 How to Run

**Option 1 — Colab (recommended, no setup):**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/ZScore-Anomaly-Detection-CNC/blob/main/20_ZScore_Anomaly_Detection.ipynb)

**Option 2 — Local:**

```bash
git clone https://github.com/LozanoLsa/ZScore-Anomaly-Detection-CNC.git
cd ZScore-Anomaly-Detection-CNC
pip install -r requirements.txt
jupyter notebook 20_ZScore_Anomaly_Detection.ipynb
```

**Requirements:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`

---

## 💡 Key Learnings

1. **Model complexity should match problem complexity.** When the failure physics is well-understood and the process baseline is Gaussian, Z-Score detection is not a simplification — it is the correct solution. Introducing more complexity would trade interpretability for no gain in detection power.

2. **Unsupervised anomaly detection has a structural advantage in industrial settings.** Most facilities do not have clean, labeled failure logs. Z-Score requires only a baseline period of normal operation — a condition almost every running process can provide.

3. **The three-sigma boundary is a statistical claim, not a magic number.** Under a Gaussian distribution, \|Z\| > 3 occurs by chance in 0.27% of readings. When the observed rate is 5.0%, the process is communicating a real signal. The threshold should be chosen based on this reasoning, not convention alone.

4. **Vibration is the leading indicator — and the right one to instrument first.** Of the four sensors, vibration crosses the three-sigma boundary first in 100% of the anomalous cycles. If sensor budget is constrained, this is the one that cannot be omitted.

5. **Statistical validation of the normality assumption closes the loop.** Running the D'Agostino K² test on normal-only cycles — not the full dataset — confirms that the Gaussian assumption holds for the process baseline. This is not a formality; it is the proof that the Z-Score threshold is meaningful.

---

## 👤 Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning
GitHub: [LozanoLsa](https://github.com/LozanoLsa) · Gumroad: [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com)

*Turning Operations into Predictive Systems — Clone it. Fork it. Improve it.*
