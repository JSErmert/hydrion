# M6 Sensor Realism — Source Traceability Map

**Document type:** Source-to-model traceability record
**Date:** 2026-04-13
**Phase:** M6 Phase 1 — Core Sensor Realism
**This is NOT a bibliography. Each entry maps a source to a specific model parameter,
behavior, or assumption in the M6 implementation.**

---

## Source Categories Used

- **[PR]** Peer-reviewed scientific paper (journal or conference)
- **[TR]** Authoritative technical reference (standards body, university extension, government)
- **[VS]** Vendor specification / OEM technical note / application note

---

## Part 1 — Differential Pressure Sensors

---

### SOURCE DP-1

**Reference:** "A High-Performance Micro Differential Pressure Sensor with Improved Sensitivity
and Wide Dynamic Range", PMC / MDPI Sensors, 2024.
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC11596424/

**Category:** [PR]

**Why selected:** Provides measured sensitivity and nonlinearity data for MEMS DP sensors in
the 0–1 kPa range; extrapolated with scaling for 0–50 kPa range.

**Model component informed:** DifferentialPressureSensor — noise floor (dp_noise_kPa)

**Parameters / behaviors taken:**
- Sensitivity: 3.401 mV/V/kPa (0–1 kPa sensor)
- Nonlinearity: 0.376% FS (high-performance); 2–3x worse for uncompensated low-cost units
- Derived noise floor: ~0.1–0.15 kPa for high-performance; ~0.25–0.5 kPa for low-cost

**Usage role:** Baseline parameterization

**M6 parameter:** dp_noise_kPa = 0.25 — **source-grounded** (conservative low-cost estimate)

---

### SOURCE DP-2

**Reference:** "Advances in high-performance MEMS pressure sensors: design, fabrication, and
packaging", Nature Microsystems & Nanoengineering, 2023.
URL: https://www.nature.com/articles/s41378-023-00620-1

**Category:** [PR]

**Why selected:** Authoritative review of MEMS pressure sensor technology covering noise
floor, response time, and operating ranges.

**Model component informed:** DifferentialPressureSensor — noise floor, response time (latency)

**Parameters / behaviors taken:**
- Response time range: 1–50 ms (design-dependent; sensitivity-optimized units at 50 ms)
- Technology overview confirms low-cost MEMS noise at 0.3–0.5% FS typical

**Usage role:** Design rationale (response time) + baseline parameterization (noise floor)

**M6 parameter:** dp_latency_steps = 1 (100 ms at 10 Hz loop) — **source-grounded**

---

### SOURCE DP-3

**Reference:** "Fouling as a cause of drift in pressure sensors", STS Sensors technical note.
URL: https://www.stssensors.com/us/fouling-as-a-cause-of-drift-in-pressure-sensors/

**Category:** [VS]

**Why selected:** Directly documents the fouling-induced drift mechanism for pressure sensors
in contaminated-fluid service. Most directly relevant source to HydrOS operating context.

**Model component informed:** DifferentialPressureSensor — fouling offset term (dp_fouling_gain),
drift mechanism (dp_drift_rate_kPa_per_step)

**Parameters / behaviors taken:**
- Mechanism confirmed: soot/biofilm deposits on diaphragm reduce sensitivity; trapped
  pressure causes zero offset creep
- Mechanism confirmed: silicone film from airborne silicones alters mechanical response
- **Quantitative rate (kPa/hour): NOT published** — mechanism documented; magnitude absent

**Usage role:** Design rationale (fouling offset model structure)

**M6 parameter:** dp_fouling_gain = 0.5 — **assumption-based** (mechanism from this source;
gain value requires field calibration)
**M6 parameter:** dp_drift_rate_kPa_per_step = 0.0005 — **assumption-based** (mechanism from
this source; rate value is a conservative placeholder)

---

### SOURCE DP-4

**Reference:** "Differential Pressure Transducers For Filter Condition Monitoring",
Core Sensors application note.
URL: https://core-sensors.com/differential-pressure-transducers-for-filter-monitoring/

**Category:** [VS]

**Why selected:** Directly addresses differential pressure sensing in filter fouling
applications — most directly matched to HydrOS use case.

**Model component informed:** DifferentialPressureSensor — validates use of dp_total_pa as
source signal; confirms filter bypass pressure threshold ranges

**Parameters / behaviors taken:**
- Filter bypass activation: 5–50 kPa pressure rise (industry standard; consistent with
  HydrOS bypass threshold of 65 kPa)
- Confirms DP sensor is the standard instrument for filter condition monitoring

**Usage role:** Design rationale

**M6 parameter:** Source signal choice dp_total_pa — **source-grounded** (validates signal path)

---

### SOURCE DP-5

**Reference:** "The Value of Calibration for MEMS Pressure Sensors", Merit Sensor whitepaper.
URL: https://meritsensor.com/the-value-of-calibration-for-mems-pressure-sensors/

**Category:** [VS]

**Why selected:** Documents factory calibration tolerances and temperature coefficient of
offset for low-cost MEMS pressure sensors.

**Model component informed:** DifferentialPressureSensor — initial sensor offset, temperature
drift bound

**Parameters / behaviors taken:**
- Factory offset tolerance: +/-1% FS for low-cost uncompensated units
- Temperature coefficient of offset: +/-0.25% FS/degC (documented)
- At 25 degC range: +/-6.25% FS possible if uncompensated

**Usage role:** Design rationale (bounds for conservative noise floor estimate)

**M6 parameter:** dp_noise_kPa = 0.25 context — **source-grounded** (supports conservative
estimate; 0.25 kPa is ~0.5% of 50 kPa range)

---

### SOURCE DP-6

**Reference:** "Common Issues and Troubleshooting of Pressure Sensors", Zero Instrument
technical guide.
URL: https://zeroinstrument.com/common-issues-and-troubleshooting-of-pressure-sensors/

**Category:** [VS]

**Why selected:** Documents port blockage as a discrete failure mode causing trapped-pressure
zero offset — confirms fouling-induced bias direction.

**Model component informed:** DifferentialPressureSensor — fouling offset model (additive bias)

**Parameters / behaviors taken:**
- Dust/debris port blockage causes false zero offset (additive bias)
- Confirms fouling offset acts as an additive constant, not a scaling error

**Usage role:** Design rationale (offset sign and type confirmed as additive)

**M6 parameter:** Fouling offset model structure — **source-grounded** (additive, not multiplicative)

---

## Part 2 — Flow Rate Sensors

---

### SOURCE FL-1

**Reference:** "Magnetic Flowmeters for Dirty Water", Flowmeters.com.
URL: https://www.flowmeters.com/magnetic-for-dirty-water

**Category:** [TR]

**Why selected:** Primary reference for electromagnetic flow meter suitability in dirty/
particle-laden wastewater service — directly matches HydrOS operating conditions.

**Model component informed:** FlowRateSensor — sensor type selection, noise model

**Parameters / behaviors taken:**
- Mag meter: handles biofilm, suspended solids, conductive wastewater without clogging
- Faraday principle: drift-free, temperature-independent
- Suitable for 0–15 L/min with appropriate tube diameter

**Usage role:** Baseline parameterization (sensor type selection)

**M6 parameter:** FlowRateSensor uses multiplicative noise (no drift term) — **source-grounded**
(mag meter drift-free characteristic from Faraday principle)

---

### SOURCE FL-2

**Reference:** "Technical Overview: Electromagnetic (Mag) Flow Meters", ICON Process Controls.
URL: https://iconprocon.com/blog-post/technically-speaking-everything-you-need-to-know-about-magnetic-flow-meters-mag-meters-electromagnetic-flow-meters

**Category:** [VS]

**Why selected:** Provides accuracy specifications for mag meters in dirty-water service,
including repeatability and drift characteristics.

**Model component informed:** FlowRateSensor — flow_noise_frac

**Parameters / behaviors taken:**
- Accuracy: +/-0.2–0.5% reading (clean water); +/-0.5–1% (dirty water conservative margin)
- Repeatability: +/-0.1% (tighter than accuracy)
- DC offset nulling achievable; post-calibration drift negligible

**Usage role:** Baseline parameterization

**M6 parameter:** flow_noise_frac = 0.01 (1% reading) — **source-grounded** (dirty water margin
above published +/-0.5% clean spec)

---

### SOURCE FL-3

**Reference:** "Review and Operational Guidelines for Portable Ultrasonic Flowmeters",
Oklahoma State University Extension, NRCS-funded technical report.
URL: https://extension.okstate.edu/fact-sheets/review-and-operational-guidelines-for-portable-ultrasonic-flowmeters.html

**Category:** [TR]

**Why selected:** Authoritative comparison of flow meter technologies for dirty-water service.
Directly addresses Doppler vs. transit-time vs. mag meter selection criteria.

**Model component informed:** FlowRateSensor — justifies mag meter selection over Doppler
ultrasonic for HydrOS conditions

**Parameters / behaviors taken:**
- Doppler accuracy "somewhat suspect and difficult to quantify" in dirty water
- Doppler requires >100 ppm particles, >100 um size; accuracy +/-1–5% reading
- Transit-time ultrasonic unsuitable for dirty liquid (requires clean, air-free fluid)

**Usage role:** Design rationale (eliminates Doppler; supports mag meter selection)

**M6 parameter:** Sensor type selection — **source-grounded** (elimination basis for
non-mag alternatives)

---

### SOURCE FL-4

**Reference:** "Understanding Flow Meter Accuracy and Repeatability", Bronkhorst.
URL: https://www.bronkhorst.com/knowledge-base/flow-meters-accuracy-repeatability/

**Category:** [VS]

**Why selected:** Defines accuracy vs. repeatability distinction; provides industry-standard
accuracy values used to bound flow_noise_frac.

**Model component informed:** FlowRateSensor — flow_noise_frac bounds

**Parameters / behaviors taken:**
- Accuracy (absolute): +/-0.5% reading (typical mag meter)
- Repeatability: +/-0.1% (always better than accuracy)
- Conservative dirty-water assumption: 2x accuracy = 1% reading

**Usage role:** Baseline parameterization

**M6 parameter:** flow_noise_frac = 0.01 — **source-grounded** (2x clean-water spec)

---

### SOURCE FL-5

**Reference:** "How to Calibrate Magnetic Flow Meters", Kyue Instruments technical guide;
"Electromagnetic Flow Meter Calibration", Tameson.
URLs: https://kyueinstruments.com/calibrating-magnetic-flow-meters/
      https://tameson.com/pages/electromagnetic-flow-meter-calibration

**Category:** [VS]

**Why selected:** Documents factory calibration tolerance and DC offset nulling procedure
for mag meters; grounds the calibration bias model.

**Model component informed:** FlowRateSensor — flow_bias_std_lmin

**Parameters / behaviors taken:**
- Wet calibration accuracy: +/-0.2% achievable under controlled conditions
- Factory calibration tolerance: +/-0.5–1% FS for low-cost units
- At 15 L/min full scale: factory offset +/-0.075–0.15 L/min
- Conservative upper bound with margin: +/-0.2 L/min std dev

**Usage role:** Baseline parameterization

**M6 parameter:** flow_bias_std_lmin = 0.2 — **source-grounded** (conservative above factory
tolerance; std dev of per-episode sampled bias)

---

## Part 3 — Sensor Noise Model Architecture

---

### SOURCE NM-1

**Reference:** "IMU Noise Model", ethz-asl/kalibr wiki (ETH Zurich Autonomous Systems Lab).
URL: https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model

**Category:** [TR]

**Why selected:** Canonical parametric noise model reference for physical sensors. Defines
AWGN + random walk decomposition and Allan variance characterization methodology.

**Model component informed:** DifferentialPressureSensor and FlowRateSensor — both noise model
structure and parameter decomposition

**Parameters / behaviors taken:**
- AWGN: zero-mean white noise, sigma_n, independent per sample. Allan slope -1/2.
- Random walk (velocity/angle random walk): integrated drift, sigma_rw. Allan slope +1/2.
- Standard: implement both components; characterize separately via Allan variance.

**Usage role:** Baseline parameterization (model architecture)

**M6 parameter:** AWGN + random walk model structure — **source-grounded**

---

### SOURCE NM-2

**Reference:** "Gyro Noise and Allan Deviation Analysis", Michael Wrona technical blog.
URL: https://mwrona.com/posts/gyro-noise-analysis/

**Category:** [TR]

**Why selected:** Practical implementation reference for AWGN + random walk noise model,
showing how Allan variance decomposition maps to simulation parameters.

**Model component informed:** DifferentialPressureSensor — drift accumulator implementation

**Parameters / behaviors taken:**
- Random walk per step: sigma_step = sigma_rw_per_root_time * sqrt(dt)
- Confirms drift accumulation cap is a valid engineering approximation

**Usage role:** Design rationale (implementation of drift accumulator)

**M6 parameter:** drift accumulator pattern — **source-grounded** (validated implementation
approach)

---

### SOURCE NM-3

**Reference:** "Optimizing MEMS IMU Data Coherence and Timing in Navigation Systems",
Analog Devices Application Note.
URL: https://www.analog.com/en/resources/analog-dialogue/articles/optimizing-mems-imu-data-coherence-and-timing-in-navigation-systems

**Category:** [VS]

**Why selected:** Documents data-ready synchronization and latency for MEMS sensors at
10 Hz control loop rates; grounds the 1-step latency model.

**Model component informed:** DifferentialPressureSensor — dp_latency_steps

**Parameters / behaviors taken:**
- Data-ready signals achievable at loop rate with external synchronization
- At 10 Hz (100 ms/step), 1-step buffering (100 ms total latency) is physically realistic
- Sensors with 1–50 ms response time: well within 1-step budget

**Usage role:** Baseline parameterization

**M6 parameter:** dp_latency_steps = 1 — **source-grounded**

---

## Part 4 — Turbidity / Particle Proxy

---

### SOURCE TB-1

**Reference:** "NTU in Turbidity: What It Means and Why It Matters", AlpHa Measure.
URL: https://alpha-measure.com/ntu-in-turbidity-what-it-means-and-why-it-matters

**Category:** [TR]

**Why selected:** Defines linear measurement range, nonlinear transition, and saturation
threshold for nephelometric turbidity sensors.

**Model component informed:** OpticalSensorArray (existing) — saturation behavior bounds

**Parameters / behaviors taken:**
- Linear zone: 0–40 NTU
- Nonlinear transition: 40–2,000 NTU
- Saturation / signal reversal: above ~2,000 NTU

**Usage role:** Background context (informs existing turbidity model limitations)

**M6 parameter:** None directly — documents limits of existing optical sensor implementation

---

### SOURCE TB-2

**Reference:** "Economical and Novel Microplastic Detection Using an Arduino-based Turbidity
Sensor", Authorea preprint.
URL: https://www.authorea.com/users/901095/articles/1357863-economical-and-novel-microplastic-detection-using-a-arduino-based-turbidity-sensor-a-comprehensive-investigation

**Category:** [PR] (preprint)

**Why selected:** Directly demonstrates turbidity sensor performance for microplastic
detection in water — most closely matched application to HydrOS.

**Parameters / behaviors taken:**
- 95% classification accuracy for microplastics (PE, PP, nylon) at 10–100 mg/L
- ~100% sensitivity achieved using turbidity decay-over-time method
- Confirms turbidity is a valid proxy for microplastic concentration in suspension

**Usage role:** Background context (validates turbidity approach for microplastic detection)

**M6 parameter:** None directly — confirms existing optical sensor approach is sound

---

### SOURCE TB-3

**Reference:** "Measuring Turbidity, TSS, and Water Clarity", Fondriest Environmental
Monitoring.
URL: https://www.fondriest.com/environmental-measurements/measurements/measuring-water-quality/turbidity-sensors-meters-and-methods/

**Category:** [TR]

**Why selected:** Defines relationship between turbidity, total suspended solids, and
particle concentration; explicitly addresses turbidity vs. mass proxy limitation.

**Parameters / behaviors taken:**
- Turbidity measures optical scattering, not particle mass
- TSS and turbidity correlate but are not equivalent
- Particle size, shape, and refractive index all affect turbidity reading for a given mass

**Usage role:** Background context — establishes that turbidity-to-mass calibration is
calibration-pending; not directly parameterized

**M6 parameter:** Turbidity-to-mass calibration — **calibration-pending** (noted in research
brief; no source provides laundry-specific calibration curve)

---

## Part 5 — DEP / Flow-Threshold Capture

---

### SOURCE DEP-1

**Reference:** "DEP-on-a-Chip: Dielectrophoresis Applied to Microfluidic Platforms — A Review",
PMC / MDPI Micromachines, 2019.
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC6630590/

**Category:** [PR]

**Why selected:** Most comprehensive review of DEP capture efficiency vs. flow rate for
microfluidic systems; documents the functional relationship and parameter range.

**Model component informed:** CCE physics — DEP capture efficiency vs. flow rate functional form

**Parameters / behaviors taken:**
- Capture efficiency decreases with flow rate
- Functional form: eta ~ 1/Q^alpha, alpha ~ 0.5–1.0
- At 0.2 uL/min: ~90% capture; at 0.6 uL/min: ~65% (2.2x efficiency drop for 3x flow increase
  -> alpha ~ 0.8)
- Critical field threshold documented; above threshold: electroporation risk

**Usage role:** Background context (physics validation for existing CCE model structure)

**M6 parameter:** DEP threshold plausibility — **source-grounded** (order-of-magnitude
consistency with HydrOS Q_crit = 7.7 L/min confirmed)

---

### SOURCE DEP-2

**Reference:** "Dielectrophoresis-Enhanced Microfluidic Device with Membrane Filter for
Microplastic Separation", PMC, 2025.
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC11857826/

**Category:** [PR]

**Why selected:** Specifically addresses microplastic separation using DEP with membrane
filter — most directly relevant to HydrOS application.

**Parameters / behaviors taken:**
- >99% microplastic separation efficiency demonstrated with combined DEP + membrane
- Electric field mechanisms for microplastic capture confirmed
- Confirms DEP is technically viable for the microplastic separation use case

**Usage role:** Background context (validates HydrOS DEP capture architecture)

**M6 parameter:** None directly — validates existing physics architecture

---

### SOURCE DEP-3

**Reference:** "Direct Current Electrokinetic Particle Trapping in Insulator-Based
Microfluidics", ACS Analytical Chemistry, 2021.
URL: https://pubs.acs.org/doi/10.1021/acs.analchem.0c01303

**Category:** [PR]

**Why selected:** Documents electrokinetic equilibrium condition (critical flow rate at
which DEP force balances fluid drag) — directly relevant to Q_crit derivation.

**Parameters / behaviors taken:**
- Electrokinetic equilibrium velocity: device-specific; 0.001–0.01 m/s range
- At equilibrium: F_DEP = F_drag; particles begin to escape above this velocity
- Critical flow rate derivation: Q_crit = v_crit x A (same as HydrOS derivation)

**Usage role:** Design rationale (validates Q_crit derivation methodology)

**M6 parameter:** Q_crit derivation method — **source-grounded** (confirms approach, not
specific value)

---

## Part 6 — Parameter Status Summary

### Adopted for Baseline Simulation Now

Parameters with direct literature support; implemented in configs/default.yaml:

| Parameter | Value | Primary Source |
|---|---|---|
| dp_noise_kPa | 0.25 | SOURCE DP-1, DP-5 (MEMS spec literature) |
| dp_latency_steps | 1 | SOURCE DP-2, NM-3 (MEMS response time + control loop) |
| flow_noise_frac | 0.01 | SOURCE FL-2, FL-4 (mag meter accuracy specs) |
| flow_bias_std_lmin | 0.2 | SOURCE FL-5 (mag meter calibration tolerance) |
| AWGN + random walk model | structure | SOURCE NM-1, NM-2 (IMU noise model framework) |
| Sensor type: electromagnetic | design | SOURCE FL-1, FL-3 (dirty water suitability) |
| Fouling offset: additive | structure | SOURCE DP-6 (port blockage mechanism confirmed) |

### Requires Future Hardware Calibration

Parameters where the mechanism is confirmed but the magnitude must be field-measured:

| Parameter | Current Value | Why Pending | Recommended Test |
|---|---|---|---|
| dp_drift_rate_kPa_per_step | 0.0005 | No published rate for laundry fouling service | Accelerated fouling test + reference manometer |
| dp_drift_max_kPa | 2.0 | Conservative cap; no field data | Same fouling test; measure peak offset |
| dp_fouling_gain | 0.5 | Mechanism confirmed; gain application-specific | Parallel DP sensor + manometer vs. mesh loading |
| Turbidity-to-mass curve | N/A | Laundry-specific; no universal calibration | Gravimetric collection at stepped turbidity setpoints |
| DEP Q_crit (7.7 L/min) | physics-derived | Consistent with microfluidic data; HydrOS geometry untested | Bench test: electric field on mock conical stage vs. flow |

### Informative Only / Not Directly Parameterized

Sources used to validate approach, select sensor type, or confirm architecture — not
directly mapped to numeric M6 parameters:

- SOURCE TB-2 (Arduino turbidity for microplastics) — validates optical sensor approach
- SOURCE TB-3 (turbidity vs. TSS) — establishes calibration-pending status
- SOURCE DEP-2 (DEP microplastic separation efficiency) — validates DEP architecture
- SOURCE FL-3 (Doppler limitations) — eliminates Doppler as alternative; supports mag meter

---

*Source traceability compiled 2026-04-13. All URLs accessed and reviewed during M6 planning
phase. See M6_sensor_realism_research_brief.md for synthesized parameter recommendations.*
