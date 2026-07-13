"""
app.py — CNC Anomaly Sentinel Dashboard
LozanoLsa · Project 20 · Z-Score Anomaly Detection · 2026 · FREE PROJECT

Algorithm: Statistical Z-Score (|Z| > 3σ threshold)
Domain: CNC Machining — Real-Time Cycle Anomaly Detection
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CNC Anomaly Sentinel · Z-Score · LozanoLsa",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── LIGHT THEME CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;600&family=Instrument+Serif:ital@0;1&display=swap');

:root {
    --bg:       #ffffff;
    --surface:  #f8fafc;
    --card:     #f8fafc;
    --card2:    #f1f5f9;
    --border:   #e2e8f0;
    --border2:  #cbd5e1;
    --red:      #dc2626;
    --red2:     #ef4444;
    --ok:       #16a34a;
    --warn:     #d97706;
    --blue:     #2563eb;
    --purp:     #7c3aed;
    --text:     #0f172a;
    --text2:    #1e293b;
    --muted:    #64748b;
    --muted2:   #94a3b8;
    --fh: 'Syne', sans-serif;
    --fm: 'JetBrains Mono', monospace;
    --fs: 'Instrument Serif', Georgia, serif;
}

.stApp { background: var(--bg) !important; color: var(--text); font-family: var(--fh); }
.block-container { padding: 1.8rem 2.4rem 3rem !important; max-width: 1400px !important; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] { background: var(--surface) !important;
    border-right: 1px solid var(--border) !important; }

[data-testid="stSlider"] [role="slider"] { background: var(--red) !important;
    border: 2px solid var(--red2) !important; box-shadow: 0 0 6px rgba(220,38,38,0.3) !important; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { font-family: var(--fm) !important;
    font-size: 0.65rem !important; color: var(--red) !important; background: #fff !important;
    border: 1px solid var(--border) !important; padding: 1px 5px !important; border-radius: 3px !important; }
[data-testid="stSlider"] > div > div > div > div { background: var(--red) !important; }

[data-testid="stSelectbox"] > div > div { background: #fff !important;
    border: 1px solid var(--border2) !important; color: var(--text) !important;
    font-family: var(--fm) !important; font-size: 0.78rem !important; border-radius: 3px !important; }

[data-testid="stMetric"] { background: #fff !important; border: 1px solid var(--border) !important;
    border-top: 2px solid var(--red) !important; padding: 1rem 1.1rem 0.9rem !important;
    border-radius: 3px !important; box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important; }
[data-testid="stMetricLabel"] > div { font-family: var(--fm) !important; font-size: 0.6rem !important;
    text-transform: uppercase !important; letter-spacing: 0.18em !important;
    color: var(--muted) !important; font-weight: 400 !important; }
[data-testid="stMetricValue"] > div { font-family: var(--fm) !important;
    font-size: 1.7rem !important; font-weight: 600 !important;
    color: var(--text) !important; line-height: 1.1 !important; }

[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid var(--border) !important;
    gap: 0 !important; background: transparent !important; }
[data-testid="stTabs"] [role="tab"] { font-family: var(--fm) !important;
    font-size: 0.68rem !important; text-transform: uppercase !important;
    letter-spacing: 0.12em !important; color: var(--muted) !important;
    padding: 0.5rem 1.2rem !important; border: none !important;
    border-radius: 0 !important; background: transparent !important; transition: all 0.2s !important; }
[data-testid="stTabs"] [role="tab"]:hover { color: var(--red) !important;
    background: rgba(220,38,38,0.04) !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: var(--red) !important;
    border-bottom: 2px solid var(--red) !important; background: transparent !important; }
[data-testid="stTabsContent"] { padding-top: 1.4rem !important; }

[data-testid="stAlert"] { border-radius: 2px !important; font-family: var(--fm) !important;
    font-size: 0.75rem !important; }
[data-testid="stExpander"] { background: #fff !important; border: 1px solid var(--border) !important;
    border-radius: 2px !important; box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important; }
[data-testid="stExpander"] summary { font-family: var(--fm) !important;
    font-size: 0.72rem !important; color: var(--text) !important; }

[data-testid="stDataFrame"] { border: 1px solid var(--border) !important;
    border-radius: 2px !important; box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important; }
[data-testid="stDataFrame"] th { font-family: var(--fm) !important;
    font-size: 0.62rem !important; text-transform: uppercase !important;
    letter-spacing: 0.12em !important; background: var(--card2) !important;
    color: var(--muted) !important; border-bottom: 1px solid var(--border) !important; }
[data-testid="stDataFrame"] td { font-family: var(--fm) !important;
    font-size: 0.72rem !important; color: var(--text2) !important;
    background: #fff !important; }

hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }
[data-testid="stCaptionContainer"] p { font-family: var(--fm) !important;
    font-size: 0.62rem !important; color: var(--muted) !important; letter-spacing: 0.08em !important; }
h1, h2, h3 { font-family: var(--fh) !important; color: var(--text) !important; }
p, li { font-family: var(--fh) !important; font-size: 0.88rem !important; color: var(--text2); }

/* LSA COMPONENTS — Light */
.lsa-header { border-bottom: 1px solid var(--border); padding-bottom: 1.2rem; margin-bottom: 0.2rem; }
.lsa-project-tag { font-family: var(--fm); font-size: 0.6rem; color: var(--red);
    text-transform: uppercase; letter-spacing: 0.22em; margin-bottom: 4px; }
.lsa-title { font-family: var(--fh); font-size: 1.85rem; font-weight: 800;
    color: var(--text); line-height: 1.1; letter-spacing: -0.02em; }
.lsa-tagline { font-family: var(--fs); font-style: italic; font-size: 0.9rem;
    color: var(--muted); margin-top: 4px; }
.lsa-chip { display: inline-block; background: #fef2f2; border: 1px solid #fecaca;
    color: var(--red); font-family: var(--fm); font-size: 0.58rem;
    letter-spacing: 0.1em; text-transform: uppercase; padding: 2px 8px;
    border-radius: 2px; margin-right: 5px; }
.lsa-chip-free { display: inline-block; background: #f0fdf4; border: 1px solid #bbf7d0;
    color: var(--ok); font-family: var(--fm); font-size: 0.58rem;
    letter-spacing: 0.1em; text-transform: uppercase; padding: 2px 8px;
    border-radius: 2px; margin-right: 5px; }
.lsa-section { font-family: var(--fm); font-size: 0.6rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.2em; margin-bottom: 10px;
    padding-bottom: 5px; border-bottom: 1px solid var(--border); }
.lsa-footer { margin-top: 2.5rem; padding-top: 0.8rem; border-top: 1px solid var(--border);
    font-family: var(--fm); font-size: 0.58rem; color: var(--muted);
    letter-spacing: 0.1em; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ─── MATPLOTLIB LIGHT THEME ───────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8fafc",
    "axes.edgecolor":   "#e2e8f0",
    "axes.labelcolor":  "#0f172a",
    "xtick.color":      "#64748b",
    "ytick.color":      "#64748b",
    "text.color":       "#0f172a",
    "grid.color":       "#e2e8f0",
    "grid.linestyle":   "--",
    "grid.alpha":       0.7,
    "legend.facecolor": "white",
    "legend.edgecolor": "#e2e8f0",
})

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DATA_PATH     = "data_analysis.csv"
DATA_PATH_ALT = "20_ZScore_CNC_Anomaly/data_analysis.csv"
THRESHOLD     = 3.0

FEATURES = ["vibration_mm_s", "cutting_force_n", "tool_temp_c", "energy_kwh"]
LABELS   = ["Vibration (mm/s)", "Cutting Force (N)", "Tool Temp (°C)", "Energy (kWh)"]
LIMITS   = {"vibration_mm_s": 7.5, "cutting_force_n": 160,
            "tool_temp_c": 75, "energy_kwh": 3.2}

# ─── LIGHT PALETTE ───────────────────────────────────────────────────────────
C_RED   = "#dc2626"
C_RED2  = "#ef4444"
C_OK    = "#16a34a"
C_WARN  = "#d97706"
C_BLUE  = "#2563eb"
C_PURP  = "#7c3aed"
C_TEXT  = "#0f172a"
C_TEXT2 = "#1e293b"
C_MUTED = "#64748b"
C_CARD  = "#f8fafc"
C_CARD2 = "#f1f5f9"
C_BORD  = "#e2e8f0"

SENSOR_COLORS = [C_RED, C_BLUE, C_WARN, C_PURP]

# ─── DATA & Z-SCORES ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    for path in [DATA_PATH, DATA_PATH_ALT]:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            continue
    st.error("data_analysis.csv not found. Place the file in the same folder as app.py.")
    st.stop()

df    = load_data()
means = df[FEATURES].mean()
stds  = df[FEATURES].std()
z_df  = (df[FEATURES] - means) / stds
z_df.columns = [f"z_{f}" for f in FEATURES]

df["anomaly"]      = (z_df.abs() > THRESHOLD).any(axis=1).astype(int)
df["anomaly_real"] = (
    (df["vibration_mm_s"]  > LIMITS["vibration_mm_s"])  |
    (df["cutting_force_n"] > LIMITS["cutting_force_n"]) |
    (df["tool_temp_c"]     > LIMITS["tool_temp_c"])     |
    (df["energy_kwh"]      > LIMITS["energy_kwh"])
).astype(int)
df_full = pd.concat([df, z_df], axis=1)

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="lsa-header">
    <div class="lsa-project-tag">ML Project #20 · Z-Score · Anomaly Detection · CNC Machining</div>
    <div class="lsa-title">3σ Away From Normal Is Never a Coincidence</div>
    <div class="lsa-tagline">No model training. No labels. Just the statistical distance from what the process looked like when it was healthy.</div>
    <div style="margin-top:10px;">
        <span class="lsa-chip">Z-SCORE · |Z| &gt; 3.0</span>
        <span class="lsa-chip">4 SENSORS</span>
        <span class="lsa-chip">{df['anomaly'].sum()} ANOMALIES · {df['anomaly'].mean()*100:.1f}%</span>
        <span class="lsa-chip">NO TRAINING REQUIRED</span>
        <span class="lsa-chip-free">FREE PROJECT</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TOP KPI ROW ──────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Cycles",       f"{len(df):,}",               "Machining records")
k2.metric("Anomalies Detected", f"{df['anomaly'].sum()}",      f"{df['anomaly'].mean()*100:.1f}% · |Z|>3")
k3.metric("Sensors Monitored",  "4",                           "Vib · Force · Temp · Energy")
k4.metric("Z-Score Threshold",  "3.0 σ",                      "Industry standard")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "DATA EXPLORER", "MODEL PERFORMANCE", "SCENARIO SIMULATOR", "RISK DRIVERS", "ACTION PLAN"
])

# ══ TAB 1 ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="lsa-section">// CNC machining dataset — feature distributions</div>',
                unsafe_allow_html=True)

    col_l, col_r = st.columns([1.2, 1])
    with col_l:
        st.markdown('<div class="lsa-section">// 4-sensor distributions with 3σ boundaries</div>',
                    unsafe_allow_html=True)
        fig, axes = plt.subplots(2, 2, figsize=(9, 6))
        for ax, feat, lbl, col in zip(axes.flat, FEATURES, LABELS, SENSOR_COLORS):
            mu, sig = df[feat].mean(), df[feat].std()
            ax.hist(df[feat], bins=40, color=col, edgecolor="white",
                    alpha=0.75, linewidth=0.3)
            ax.axvline(mu,         color=C_TEXT,  lw=2,   ls="--", alpha=0.7)
            ax.axvline(mu + 3*sig, color=C_RED,   lw=1.5, ls=":",  alpha=0.9)
            ax.axvline(mu - 3*sig, color=C_RED,   lw=1.5, ls=":",  alpha=0.9)
            ax.set_title(lbl, fontsize=9, fontweight="bold")
            ax.grid(True, alpha=0.5)
        fig.suptitle("Sensor Distributions  (dashed = μ · dotted = μ ± 3σ)",
                     fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()
        st.caption("Red dotted lines = 3σ boundary. Cycles beyond are anomaly candidates.")

    with col_r:
        st.markdown('<div class="lsa-section">// Sensor statistics</div>',
                    unsafe_allow_html=True)
        desc       = df[FEATURES].describe().T.round(3)
        desc.index = LABELS
        st.dataframe(desc[["mean", "std", "min", "max"]], use_container_width=True)

        st.markdown('<div class="lsa-section" style="margin-top:14px;">// Anomaly count</div>',
                    unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(4, 2.5))
        counts = [(df["anomaly"] == 0).sum(), (df["anomaly"] == 1).sum()]
        bars   = ax2.barh(["Normal", "Anomaly"], counts,
                          color=[C_OK, C_RED], alpha=0.85, edgecolor="white", height=0.5)
        for bar, v in zip(bars, counts):
            ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                     f"{v:,}", va="center", fontsize=9, color=C_TEXT)
        ax2.set_xlabel("Cycles")
        ax2.grid(True, axis="x", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True); plt.close()

    st.divider()
    st.markdown('<div class="lsa-section">// Z-score time series — sensor deviation over time</div>',
                unsafe_allow_html=True)
    fig3, axes3 = plt.subplots(4, 1, figsize=(13, 8), sharex=True)
    for ax, feat, lbl, col in zip(axes3, FEATURES, LABELS, SENSOR_COLORS):
        z_col = f"z_{feat}"
        ax.plot(z_df[z_col].values, color=col, lw=0.8, alpha=0.9)
        ax.axhline( THRESHOLD, color=C_RED, lw=1.5, ls="--", alpha=0.8)
        ax.axhline(-THRESHOLD, color=C_RED, lw=1.5, ls="--", alpha=0.8)
        ax.fill_between(range(len(z_df)), THRESHOLD,
                        z_df[z_col].clip(lower=THRESHOLD), color=C_RED, alpha=0.3)
        ax.fill_between(range(len(z_df)), -THRESHOLD,
                        z_df[z_col].clip(upper=-THRESHOLD), color=C_RED, alpha=0.3)
        ax.set_ylabel(f"|Z|\n{lbl}", fontsize=7.5)
        ax.grid(True, alpha=0.4)
    axes3[-1].set_xlabel("Cycle index")
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True); plt.close()
    st.caption("Red zones = cycles where |Z| > 3.0. Vibration shows the most extreme deviations.")

# ══ TAB 2 ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="lsa-section">// Z-Score vs ground-truth anomaly labels</div>',
                unsafe_allow_html=True)
    cm   = confusion_matrix(df["anomaly_real"], df["anomaly"])
    tn, fp, fn, tp = cm.ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    acc  = (tp + tn) / len(df)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  f"{acc:.4f}")
    m2.metric("Precision", f"{prec:.4f}")
    m3.metric("Recall",    f"{rec:.4f}")
    m4.metric("F1 Score",  f"{f1:.4f}")
    m5.metric("Threshold", f"|Z| > {THRESHOLD}")

    st.divider()
    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="lsa-section">// Confusion matrix</div>', unsafe_allow_html=True)
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", ax=ax_cm,
                    xticklabels=["Pred Normal", "Pred Anomaly"],
                    yticklabels=["True Normal", "True Anomaly"],
                    cmap=sns.light_palette(C_RED, as_cmap=True),
                    linewidths=0.5, linecolor="white",
                    cbar=False, annot_kws={"size": 14, "weight": "bold"})
        ax_cm.tick_params(colors=C_TEXT, labelsize=9)
        plt.tight_layout()
        st.pyplot(fig_cm, use_container_width=True); plt.close()
        st.caption(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    with cb:
        st.markdown('<div class="lsa-section">// Z-Score threshold sensitivity</div>',
                    unsafe_allow_html=True)
        thresholds = [2.0, 2.5, 3.0, 3.5, 4.0]
        sens_rows  = []
        for t in thresholds:
            pred_t = (z_df.abs() > t).any(axis=1).astype(int)
            cm_t   = confusion_matrix(df["anomaly_real"], pred_t)
            tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
            pr_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
            rc_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
            f1_t = 2 * pr_t * rc_t / (pr_t + rc_t) if (pr_t + rc_t) > 0 else 0
            sens_rows.append({
                "Threshold": f"|Z| > {t}",
                "Flagged":   int(pred_t.sum()),
                "Recall":    f"{rc_t:.4f}",
                "Precision": f"{pr_t:.4f}",
                "F1":        f"{f1_t:.4f}",
            })
        st.dataframe(pd.DataFrame(sens_rows), use_container_width=True, hide_index=True)
        st.caption(f"Selected threshold = 3.0 balances recall with acceptable false-alarm rate.")

    st.divider()
    st.markdown('<div class="lsa-section">// Z-Score distribution — normal vs anomaly</div>',
                unsafe_allow_html=True)
    fig_z, axes_z = plt.subplots(2, 2, figsize=(13, 6))
    for ax, feat, lbl, col in zip(axes_z.flat, FEATURES, LABELS, SENSOR_COLORS):
        z_col  = f"z_{feat}"
        z_norm = df_full.loc[df["anomaly"] == 0, z_col].abs()
        z_anom = df_full.loc[df["anomaly"] == 1, z_col].abs()
        ax.hist(z_norm, bins=25, color=col,   alpha=0.60, label="Normal",  density=True)
        ax.hist(z_anom, bins=15, color=C_RED, alpha=0.80, label="Anomaly", density=True)
        ax.axvline(THRESHOLD, color=C_RED, lw=2, ls="--", label=f"|Z|={THRESHOLD}")
        ax.set_title(lbl, fontsize=10, fontweight="bold")
        ax.set_xlabel("|Z-Score|")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig_z, use_container_width=True); plt.close()
    st.caption("Detected anomalies cluster above |Z|=3. Borderline force cycles (FNs) sit in the gap between engineering limit and 3σ threshold.")

# ══ TAB 3 ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="lsa-section">// Real-time cycle classifier</div>',
                unsafe_allow_html=True)
    st.caption("Enter current sensor readings. The Z-Score engine classifies the cycle immediately.")

    # Result LEFT · Controls RIGHT
    col_out, col_inp = st.columns([3, 2])

    with col_inp:
        st.markdown(f"""
        <div style="background:#fff;border:1px solid {C_BORD};
                    border-left:3px solid {C_RED};border-radius:2px;
                    padding:1rem 1.2rem;margin-bottom:14px;">
            <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                        text-transform:uppercase;letter-spacing:.18em;">// Sensor readings</div>
        </div>
        """, unsafe_allow_html=True)
        vib   = st.slider("Vibration (mm/s)",    0.5,  12.0,  2.0, 0.1)
        force = st.slider("Cutting Force (N)",  20.0, 200.0, 80.0, 1.0)
        temp  = st.slider("Tool Temp (°C)",     20.0, 100.0, 45.0, 0.5)
        energy= st.slider("Energy (kWh)",        0.5,   4.5,  1.8, 0.05)

    z_vals     = {
        "vibration_mm_s":   (vib   - means["vibration_mm_s"])   / stds["vibration_mm_s"],
        "cutting_force_n":  (force - means["cutting_force_n"])  / stds["cutting_force_n"],
        "tool_temp_c":      (temp  - means["tool_temp_c"])      / stds["tool_temp_c"],
        "energy_kwh":       (energy- means["energy_kwh"])       / stds["energy_kwh"],
    }
    z_abs      = {k: abs(v) for k, v in z_vals.items()}
    is_anomaly = any(v > THRESHOLD for v in z_abs.values())
    max_sensor = max(z_abs, key=z_abs.get)
    max_z      = z_abs[max_sensor]

    s_color  = C_RED  if is_anomaly else C_OK
    s_label  = "ANOMALY DETECTED" if is_anomaly else "NORMAL OPERATION"
    badge_bg = "#fef2f2" if is_anomaly else "#f0fdf4"
    s_icon   = "🚨" if is_anomaly else "✅"

    with col_out:
        st.markdown(
            f'''<div style="background:#fff;border:1px solid {C_BORD};
                        border-radius:4px;padding:1.6rem 1.8rem;margin-bottom:14px;
                        box-shadow:0 2px 8px rgba(0,0,0,0.06);">
                <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                            color:{C_TEXT};margin-bottom:0.8rem;">Cycle Assessment</div>
                <div style="font-size:1.8rem;margin-bottom:6px;">{s_icon}</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1.6rem;
                            font-weight:700;color:{s_color};line-height:1.2;">{s_label}</div>
                <div style="margin-top:12px;">
                    <span style="background:{badge_bg};color:{s_color};
                                 font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                                 font-weight:600;letter-spacing:.08em;
                                 padding:5px 14px;border-radius:20px;
                                 border:1px solid {s_color}33;">
                        Max |Z| = {max_z:.3f} on {LABELS[FEATURES.index(max_sensor)]}
                    </span>
                </div>
            </div>''',
            unsafe_allow_html=True
        )

        st.markdown('<div class="lsa-section">// Z-Score per sensor</div>',
                    unsafe_allow_html=True)
        for feat, lbl, col in zip(FEATURES, LABELS, SENSOR_COLORS):
            z    = z_abs[feat]
            flag = z > THRESHOLD
            bar_color = C_RED if flag else col
            bar_w     = min(int(z / (THRESHOLD * 1.5) * 100), 100)
            flag_txt  = "  ⚠ ALARM" if flag else ""
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;
                            font-family:var(--fm);font-size:0.7rem;
                            color:{C_TEXT};margin-bottom:3px;">
                    <span>{lbl}<span style="color:{C_RED};font-weight:700;">{flag_txt}</span></span>
                    <span style="color:{bar_color};font-weight:600;">|Z| = {z:.3f}</span>
                </div>
                <div style="background:{C_CARD2};border-radius:3px;height:8px;
                            border:1px solid {C_BORD};">
                    <div style="background:{bar_color};width:{bar_w}%;
                                height:8px;border-radius:3px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        # Response decision table
        st.markdown('<div class="lsa-section" style="margin-top:14px;">// Response framework</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <table style="width:100%;font-family:var(--fm);font-size:0.68rem;
                      border-collapse:collapse;">
            <tr style="border-bottom:1px solid {C_BORD};">
                <th style="padding:5px;text-align:left;color:{C_MUTED};">|Z| Range</th>
                <th style="padding:5px;color:{C_MUTED};">Status</th>
            </tr>
            <tr style="border-bottom:1px solid {C_BORD};
                       {'background:#fef2f2' if max_z>THRESHOLD else ''}">
                <td style="padding:5px;color:{C_RED};">|Z| &gt; 3.0</td>
                <td style="padding:5px;color:{C_RED};text-align:center;">Anomaly — Halt</td>
            </tr>
            <tr style="border-bottom:1px solid {C_BORD};
                       {'background:#fffbeb' if 2<max_z<=THRESHOLD else ''}">
                <td style="padding:5px;color:{C_WARN};">2.0 &lt; |Z| ≤ 3.0</td>
                <td style="padding:5px;color:{C_WARN};text-align:center;">Warning — Monitor</td>
            </tr>
            <tr style="{'background:#f0fdf4' if max_z<=2 else ''}">
                <td style="padding:5px;color:{C_OK};">|Z| ≤ 2.0</td>
                <td style="padding:5px;color:{C_OK};text-align:center;">Normal — Continue</td>
            </tr>
        </table>
        """, unsafe_allow_html=True)

# ══ TAB 4 ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="lsa-section">// Sensor risk contribution analysis</div>',
                unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="lsa-section">// Flags per sensor (|Z| > 3)</div>',
                    unsafe_allow_html=True)
        flag_counts = {
            lbl: int((z_df[f"z_{feat}"].abs() > 3).sum())
            for feat, lbl in zip(FEATURES, LABELS)
        }
        total_anom = df["anomaly"].sum()
        flag_df = pd.DataFrame({
            "Sensor":           list(flag_counts.keys()),
            "Flags":            list(flag_counts.values()),
            "% of Anomalies":   [f"{v/total_anom*100:.0f}%" for v in flag_counts.values()],
        }).set_index("Sensor")
        st.dataframe(flag_df, use_container_width=True)

        fig_fl, ax_fl = plt.subplots(figsize=(6, 3))
        max_v  = max(flag_counts.values())
        bar_c  = [C_RED if v == max_v else C_BLUE for v in flag_counts.values()]
        bars   = ax_fl.barh(list(flag_counts.keys()), list(flag_counts.values()),
                            color=bar_c, alpha=0.85, edgecolor="white", height=0.5)
        for bar, val in zip(bars, flag_counts.values()):
            ax_fl.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                       str(val), va="center", fontsize=9, color=C_TEXT)
        ax_fl.set_xlabel("Records flagged")
        ax_fl.set_title("Sensor Contribution to Detection", fontsize=10, fontweight="bold")
        ax_fl.grid(True, axis="x", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig_fl, use_container_width=True); plt.close()
        st.caption("Vibration flags 100% of anomalies — the primary detection sensor.")

    with col_r:
        st.markdown('<div class="lsa-section">// Normal vs anomaly — mean sensor values</div>',
                    unsafe_allow_html=True)
        profile          = df.groupby("anomaly")[FEATURES].mean().T.round(2)
        profile.columns  = ["Normal", "Anomaly"]
        profile["Ratio"] = (profile["Anomaly"] / profile["Normal"]).round(2)
        profile.index    = LABELS
        st.dataframe(profile, use_container_width=True)
        st.caption("Anomalous cycles show ~2× vibration and ~1.5× force vs. normal operation.")

        st.markdown('<div class="lsa-section" style="margin-top:14px;">// Z-Score heatmap — 40 anomaly sample</div>',
                    unsafe_allow_html=True)
        anom_sample = df_full[df["anomaly"] == 1][[f"z_{f}" for f in FEATURES]].abs()
        anom_sample = anom_sample.sample(min(40, len(anom_sample)), random_state=42).reset_index(drop=True)
        anom_sample.columns = LABELS
        fig_h, ax_h = plt.subplots(figsize=(6, 3))
        sns.heatmap(anom_sample.T, cmap=sns.light_palette(C_RED, as_cmap=True),
                    vmin=0, vmax=6, linewidths=0.2, linecolor="white",
                    cbar_kws={"label": "|Z-Score|"}, annot=False, ax=ax_h)
        ax_h.set_xticklabels([], fontsize=0)
        ax_h.set_yticklabels(ax_h.get_yticklabels(), color=C_TEXT, fontsize=8, rotation=0)
        ax_h.set_title("Z-Score intensity per anomaly cycle", fontsize=9, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_h, use_container_width=True); plt.close()

# ══ TAB 5 ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="lsa-section">// Anomaly response protocol — by sensor</div>',
                unsafe_allow_html=True)

    actions = [
        {"sensor": "Vibration (mm/s)", "threshold": "> 7.5 mm/s  |  |Z| > 3.0",
         "root_cause": "Tool holder looseness · Bearing wear · Chatter resonance · Spindle imbalance",
         "immediate": "Stop cycle → Inspect tool holder torque → Check spindle bearings",
         "preventive": "Tool holder TIR check every 500 cycles · Bearing thermography monthly",
         "color": C_RED},
        {"sensor": "Cutting Force (N)", "threshold": "> 160 N  |  |Z| > 3.0",
         "root_cause": "Tool wear approaching end-of-life · Material hardness variation · Chip clogging",
         "immediate": "Reduce feed rate 20% → Inspect tool flank wear → Check chip evacuation",
         "preventive": "Replace at 80% wear limit · Material certifications per batch",
         "color": C_WARN},
        {"sensor": "Tool Temperature (°C)", "threshold": "> 75 °C  |  |Z| > 3.0",
         "root_cause": "Coolant flow interruption · Dry cutting conditions · Coating degradation",
         "immediate": "Verify coolant pressure and nozzle orientation → Inspect tool coating",
         "preventive": "Coolant filter change every 2 weeks · Pyrometer calibration monthly",
         "color": C_PURP},
        {"sensor": "Energy (kWh)", "threshold": "> 3.2 kWh  |  |Z| > 3.0",
         "root_cause": "Mechanical binding · Chuck clamping force deviation · Drive system friction",
         "immediate": "Check chuck hydraulic pressure → Inspect linear guide lubrication",
         "preventive": "Drive system inspection quarterly · Lubrication schedule via CMMS",
         "color": C_BLUE},
    ]

    for act in actions:
        with st.expander(f"{act['sensor']}  ·  Trigger: {act['threshold']}", expanded=True):
            cl, cr = st.columns([2, 1])
            with cl:
                for label, text, bg, border in [
                    ("Root Cause Candidates", act["root_cause"], "#fef2f2", C_RED),
                    ("Immediate Action",      act["immediate"],  "#fffbeb", C_WARN),
                    ("Preventive Measure",    act["preventive"], "#f0fdf4", C_OK),
                ]:
                    st.markdown(f"""
                    <div style="background:{bg};border:1px solid {border}33;
                                border-left:3px solid {border};border-radius:2px;
                                padding:8px 12px;margin-bottom:8px;">
                        <div style="font-family:var(--fm);font-size:0.6rem;
                                    color:{C_MUTED};text-transform:uppercase;
                                    letter-spacing:.15em;margin-bottom:4px;">{label}</div>
                        <div style="font-family:var(--fm);font-size:0.72rem;
                                    color:{C_TEXT2};line-height:1.6;">{text}</div>
                    </div>""", unsafe_allow_html=True)
            with cr:
                st.markdown(f"""
                <div style="background:#fff;border:1px solid {C_BORD};
                            border-top:3px solid {act['color']};border-radius:2px;
                            padding:1rem;text-align:center;margin-top:8px;
                            box-shadow:0 1px 3px rgba(0,0,0,0.06);">
                    <div style="font-family:var(--fm);font-size:0.6rem;
                                color:{C_MUTED};text-transform:uppercase;
                                letter-spacing:.1em;">Trigger</div>
                    <div style="font-family:var(--fm);font-size:0.82rem;font-weight:700;
                                color:{act['color']};margin-top:6px;line-height:1.4;">
                        {act['threshold']}</div>
                </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"""
    <div style="background:#fff;border:1px solid {C_BORD};border-radius:4px;
                padding:1.2rem 1.6rem;box-shadow:0 2px 6px rgba(0,0,0,0.06);">
        <div style="font-family:var(--fh);font-size:1rem;font-weight:700;
                    color:{C_TEXT};margin-bottom:12px;">Response Decision Framework</div>
        <table style="width:100%;font-family:var(--fm);font-size:0.72rem;border-collapse:collapse;">
            <tr style="border-bottom:1px solid {C_BORD};background:{C_CARD2};">
                <th style="padding:8px;text-align:left;color:{C_MUTED};">|Z| Range</th>
                <th style="padding:8px;color:{C_MUTED};">Classification</th>
                <th style="padding:8px;color:{C_MUTED};">Response</th>
            </tr>
            <tr style="border-bottom:1px solid {C_BORD};">
                <td style="padding:8px;color:{C_OK};font-weight:600;">|Z| ≤ 2.0</td>
                <td style="padding:8px;text-align:center;color:{C_OK};">Normal</td>
                <td style="padding:8px;color:{C_TEXT2};">Continue operation — log for trend monitoring</td>
            </tr>
            <tr style="border-bottom:1px solid {C_BORD};background:#fffbeb;">
                <td style="padding:8px;color:{C_WARN};font-weight:600;">2.0 &lt; |Z| ≤ 3.0</td>
                <td style="padding:8px;text-align:center;color:{C_WARN};">Warning zone</td>
                <td style="padding:8px;color:{C_TEXT2};">Increase monitoring frequency — plan inspection</td>
            </tr>
            <tr style="background:#fef2f2;">
                <td style="padding:8px;color:{C_RED};font-weight:600;">|Z| &gt; 3.0</td>
                <td style="padding:8px;text-align:center;color:{C_RED};">Anomaly</td>
                <td style="padding:8px;color:{C_TEXT2};">Stop cycle — execute immediate action per sensor above</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="lsa-footer">
    LozanoLsa · Turning Operations into Predictive Systems · CNC Anomaly Sentinel · Project 20 · v2.0 &nbsp;·&nbsp;
    <a href="https://github.com/LozanoLsa" style="color:{C_BLUE};text-decoration:none;">GitHub</a> &nbsp;·&nbsp;
    <a href="https://lozanolsa.gumroad.com" style="color:{C_BLUE};text-decoration:none;">Gumroad</a>
</div>
""", unsafe_allow_html=True)
