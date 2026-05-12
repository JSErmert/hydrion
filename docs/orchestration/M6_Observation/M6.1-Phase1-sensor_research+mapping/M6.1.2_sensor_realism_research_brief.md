# M6 Sensor Realism Research Brief

**Document type:** Pre-implementation research brief
**Date:** 2026-04-13
**Phase:** M6 Phase 1 — Core Sensor Realism
**Sources reviewed:** 50+ peer-reviewed articles, technical whitepapers, OEM specifications
**Status:** Final — used to parameterize M6 Phase 1 defaults in configs/default.yaml

---

## Purpose

This brief establishes source-grounded sensor model parameter ranges for the M6 Phase 1
implementation. It defines what can be modeled now, what requires future hardware calibration,
and where the literature is thin enough that simulation assumptions are unavoidable.

---

## 1. Differential Pressure Sensing (MEMS DP, 0–50 kPa)

### Noise

Low-cost MEMS differential pressure sensors in the 0–50 kPa range exhibit:
- Noise floor: 0.05–0.15 kPa (high-performance units); 0.25–0.50 kPa (low-cost, uncompensated)
- Typical nonlinearity: 0.376–2.0% FS depending on grade
- **M6 default adopted: sigma_dp = 0.25 kPa** (conservative low-cost MEMS assumption)

Sources: PMC/MEMS DP review (Nature Microsystems & Nanoengineering 2023), PMC sensor
characterization study.

### Drift

- Drift mechanism confirmed: sensor port fouling by soot, biofilm, silicone deposits causes
  slow offset creep and sensitivity reduction
- Temperature coefficient of offset: +/-0.25% FS per degC (documented)
- **Fouling-induced drift rate (kPa/hour): calibration-pending** — mechanism confirmed;
  time constant and magnitude are application-specific and not quantified in literature
- **M6 default adopted: dp_drift_rate_kPa_per_step = 0.0005** (conservative placeholder;
  must be replaced with field measurement)

Sources: STS Sensors fouling technical note; Core Sensors filtration application note;
Merit Sensor calibration whitepaper.

### Fouling Offset

- Sensor port blockage causes additive measurement bias proportional to membrane/mesh loading
- No published quantitative relationship (bias vs. loading fraction) found for laundry service
- **M6 default adopted: dp_fouling_gain = 0.5** (scales bias linearly with mesh_loading_avg;
  gain requires field calibration)
- **Status: calibration-pending**

### Latency

- MEMS DP response time: 1–50 ms (design-dependent; high-sensitivity units at upper end)
- At 10 Hz control loop (100 ms/step): 1–2 step latency is physically realistic
- **M6 default adopted: dp_latency_steps = 1** (1-step circular buffer = 100 ms lag)

Sources: Eastsensor response time reference; Analog Devices IMU coherence guide.

---

## 2. Flow Rate Sensing (0–15 L/min, Particle-Laden Wastewater)

### Sensor Type Recommendation

Electromagnetic (magnetic) flow meters are the appropriate choice for HydrOS operating
conditions:
- Handles conductive dirty water, biofilms, and suspended solids without clogging
- Faraday-principle measurement is drift-free (temperature-independent)
- Suitable for 0–15 L/min at 10 mm+ tube inner diameter
- Accuracy: +/-0.2–0.5% reading (clean); +/-0.5–1.0% (dirty water conservative margin)

Doppler ultrasonic is the second-choice option if liquid conductivity is insufficient
for a mag meter. Accuracy degrades to +/-1–5% and is particle-size dependent
(requires >100 ppm, >100 um particles).

Sources: Flowmeters.com mag meter dirty water guide; ICON Process Controls mag meter
reference; Oklahoma State University portable ultrasonic flowmeter operational guidelines.

### Noise

- **M6 default adopted: flow_noise_frac = 0.01** (1% multiplicative noise)
- Represents conservative dirty-water margin above mag meter spec (+/-0.5% reading)
- Applied as: sensor_q = truth_q * (1 + N(0, flow_noise_frac^2)) + calibration_bias

Sources: Bronkhorst accuracy vs. repeatability reference; Cadillac Meter flow accuracy guide.

### Calibration Offset

- Factory calibration tolerance: +/-0.5–1.0% FS for low-cost mag meters
- Corresponds to +/-0.075–0.15 L/min at 15 L/min full scale
- **M6 default adopted: flow_bias_std_lmin = 0.2** (per-episode sampled offset from
  N(0, 0.2^2); conservative upper bound)
- Bias is fixed for the duration of each episode (factory calibration error model)

Sources: Tameson electromagnetic flow meter calibration guide; Analog Devices EM flow
meter accuracy article.

### Latency

- Electromagnetic flow meters: electronic latency < 1 ms (Faraday principle, no transport lag)
- **M6 default: no latency buffer** — negligible vs. 100 ms step

---

## 3. Sensor Noise and Drift Model Architecture

### Adopted Framework

Standard parametric model from inertial sensor fusion literature (IMU noise model):

1. **Additive white Gaussian noise (AWGN):** fast-fluctuating zero-mean noise, independent
   per step. Dominates at short timescales. Allan deviation slope: -1/2.
   Model: y_sensor = y_truth + N(0, sigma^2)

2. **Random walk (velocity random walk):** integrated drift; dominates over hours. Allan
   deviation slope: +1/2.
   Model: bias_t = bias_{t-1} + N(0, sigma_rw^2)

Both components are implemented for the DP sensor. Flow sensor uses AWGN + fixed calibration
bias (mag meter is drift-free by design).

Sources: ethz-asl/kalibr IMU noise model documentation; Michael Wrona Allan variance
analysis; ScienceDirect Gaussian white noise reference.

---

## 4. Turbidity / Scatter as Particle Proxy

### Detection Limits

- Linear measurement range: 0–40 NTU (nephelometric, accurate)
- Nonlinear transition: 40–2,000 NTU (multiple scattering effects)
- Signal reversal/saturation: above ~2,000 NTU
- Low-cost sensors achieve < 0.01 NTU detection limit at bottom of range

### Reliability as Mass Proxy

Turbidity measures optical scattering, not particle mass. The two correlate but are not
equivalent. Turbidity is sensitive to particle size and refractive index; a given NTU reading
can correspond to very different masses depending on particle size distribution.

- **Turbidity is a reliable proxy for particle concentration, not for total captured mass**
- Mass calibration requires a laboratory-measured turbidity-to-mass curve for the specific
  particle distribution (laundry microfibers); no universal calibration is available
- **Status: turbidity-to-mass calibration is calibration-pending**

Sources: AlpHa Measure NTU reference; Fondriest turbidity measurement guide; PMC low-cost
turbidity sensor study; Authorea Arduino microplastic detection study.

---

## 5. DEP / Flow-Threshold Capture Behavior

### Published Evidence

Published microfluidic data (cell and particle trapping with DEP) confirms:
- Capture efficiency decreases with increasing flow rate
- Trend: eta_dep(Q) ~ 1/Q^alpha, where alpha ~ 0.5–1.0 (device-specific)
- Documented ~2.2x efficiency drop for 3x flow increase (alpha ~ 0.8)
- DEP capture requires force balance: F_DEP > F_drag

### HydrOS Threshold Consistency

HydrOS Q_crit = 7.7 L/min (pump_cmd = 0.22, derived from v_crit = Q_crit/A):
- Corresponds to v_crit ~ 2–3 mm/s depending on cross-sectional area assumption
- Published DEP capture operating windows: 0.1–10 mm/s (microfluidic)
- HydrOS threshold falls within the published operating window
- **Order-of-magnitude consistency: confirmed**
- **Exact threshold validation for 3-stage conical geometry: bench test required**
- **Status: plausible from first principles; calibration-pending for field confirmation**

Sources: PMC DEP-on-a-chip review; PMC DEP-enhanced microfluidic membrane filter study;
ACS direct current electrokinetic particle trapping; Interface Fluidics Peclet number guide.

---

## 6. Recommended M6 Baseline Parameters (Summary)

| Parameter | Value | Status | Source Category |
|---|---|---|---|
| dp_noise_kPa | 0.25 | Source-grounded | MEMS DP spec literature |
| dp_drift_rate_kPa_per_step | 0.0005 | Assumption-based | Mechanism confirmed; rate pending |
| dp_drift_max_kPa | 2.0 | Assumption-based | Conservative cap; no literature value |
| dp_fouling_gain | 0.5 | Assumption-based | Mechanism confirmed; gain pending |
| dp_latency_steps | 1 | Source-grounded | MEMS response time + control loop analysis |
| flow_noise_frac | 0.01 | Source-grounded | Electromagnetic mag meter accuracy specs |
| flow_bias_std_lmin | 0.2 | Source-grounded | Factory calibration tolerance literature |

---

## 7. What Can Be Implemented Now

1. DP sensor: AWGN model (sigma = 0.25 kPa), 1-step latency buffer, fouling offset term
   (gain placeholder pending calibration), random walk drift (rate placeholder pending calibration)
2. Flow sensor: multiplicative AWGN (1% reading), per-episode calibration bias offset
3. Both sensors write to sensor_state only — truth_state reads are undisturbed

---

## 8. What Must Remain Calibration-Pending

1. **DP drift rate in fouling service** — fouling-induced offset progression not quantified
   in literature; requires accelerated fouling test against reference manometer
2. **DP sensor port fouling gain** — bias vs. mesh loading relationship is application-specific;
   requires parallel sensor + manometer test in laundry-effluent service
3. **Magnetic flow meter drift in biofilm-heavy service** — mag meters are drift-free in
   clean/industrial wastewater; laundry detergent/biofilm electrode fouling not characterized
4. **Turbidity-to-mass calibration curve** — particle size distribution is laundry-specific;
   no universal published calibration
5. **DEP flow threshold** — Q_crit = 7.7 L/min is consistent with microfluidic principles
   but not directly validated for the HydrOS 3-stage conical geometry

---

## 9. Where Literature Is Thin

- Multi-stage cascade fouling dynamics (literature covers single-filter only)
- Biofilm + detergent interaction effects on sensor measurement (literature separates the two)
- DEP capture selectivity by fiber type (PET/PA/PVC permittivity data sparse)
- Turbidity sensor stability under optical lens fouling in laundry drain conditions

*Research brief compiled 2026-04-13 from 50+ peer-reviewed articles, OEM specifications,
and authoritative technical references. See M6_sensor_realism_sources_map.md for full
source traceability.*
