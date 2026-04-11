# M5 Physics Basis — Peer-Reviewed Citation Registry

All constants and formulas in `hydrion/physics/m5/` trace to sources listed here.
No commercial databases. Every value carries a `SOURCE:` comment in code.

---

## Rule
When any physics constant needs updating, locate a peer-reviewed source first.
Ask the user for journal access before falling back on web search.
Flag any ungrounded value with `[UNGROUNDED]` in code comments.

---

## 1. Polymer Dielectric Properties

### Relative Permittivity ε_r

| Polymer | Value | Source |
|---|---|---|
| PP (isotactic) | 2.2–2.3 | Brandrup, Immergut, Grulke. *Polymer Handbook* 4th ed. Wiley, 1999. Section VI. |
| PE (HDPE) | 2.30–2.35 | Same + Ku & Liepins. *Electrical Properties of Polymers*. Hanser, 1987. |
| PET | 3.2–3.3 @ 1kHz, 20°C | Neagu et al. *J. Appl. Phys.* 92:6365 (2002). DOI:10.1063/1.1518784 |

Note: PET ε_r is single-source and moisture-sensitive. Range 3.0–4.5 across crystallinity states.

### Volume Resistivity / Conductivity

| Polymer | ρ_v (Ω·cm) | σ (S/m) | Source |
|---|---|---|---|
| PP | 10¹⁵–10¹⁶ | ~10⁻¹⁵–10⁻¹⁶ | Ieda. *IEEE Trans. Elec. Insul.* 15:206 (1980). DOI:10.1109/TEI.1980.298314 |
| PE (HDPE) | 10¹⁵–10¹⁶ | ~10⁻¹⁵–10⁻¹⁶ | Ieda 1980; Mizutani. *IEEE TDEI* 1:923 (1994). DOI:10.1109/94.329804 |
| PET | 10¹³–10¹⁴ | ~10⁻¹³–10⁻¹⁴ | Neagu et al. 2002 (inferred from loss data — FLAG: single source) |

---

## 2. Water Permittivity

| Value | Source |
|---|---|
| ε_r(water, 20°C) = **80.20 ± 0.03** | Fernández et al. *J. Phys. Chem. Ref. Data* 24:33 (1995). DOI:10.1063/1.555977. IAPWS 1997 standard. |

---

## 3. Hamaker Constant

| Value | Source |
|---|---|
| H = **0.5 × 10⁻²⁰ J** (polyolefin/water, lower bound) | Visser (1972); Gregory (1981). Conservative lower end recommended for DLVO to avoid overestimating van der Waals attraction. NIH/ACS Publications confirm range 0.5–1.3 × 10⁻²⁰ J. |

---

## 4. Clausius-Mossotti Factor (nDEP Confirmation)

Re[K(ω→∞)] = (ε_p − ε_m) / (ε_p + 2ε_m)

| Polymer | Re[K] | Type |
|---|---|---|
| PP | −0.480 | **nDEP** |
| PE | −0.479 | **nDEP** |
| PET | −0.472 | **nDEP** |
| DC limit (all) | −0.500 | **nDEP** |

**Experimental confirmation:**
- Pethig et al. *J. Phys. D* 25:881 (1992). DOI:10.1088/0022-3727/25/5/022 — PS surrogate, nDEP in water below crossover
- Gascoyne & Vykoukal. *Electrophoresis* 23:1973 (2002). DOI:10.1002/1522-2683(200207)23:13
- RSC Advances/PMC (2025) — **direct confirmation on PP, PE, PET, PVC fragments (25–50 µm)** using FISHBONE-and-funnel electrode geometry. nDEP used to focus particles to channel center → Raman ID.

---

## 5. Rajagopalan-Tien (1976) — Liquid-Phase Filtration

**Primary source:**
Rajagopalan, R., Tien, C. (1976). Trajectory analysis of deep-bed filtration with the sphere-in-cell porous media model. *AIChE Journal* 22(3):523–533. DOI:10.1002/aic.690220316

**Formula:**
```
η_0 = 4.0 A_s^(1/3) N_Pe^(-2/3)
    + A_s N_Lo^(1/8) N_R^(15/8)
    + 0.00338 A_s N_G^(1.2) N_R^(-0.4)
```

**Bed efficiency:**
Tien, C., Payatakes, A.C. (1979). Advances in deep bed filtration.
*AIChE Journal* 25(5):737–759. DOI:10.1002/aic.690250502

**Why not Lee & Liu (1982):**
Lee & Liu 1982 (*Aerosol Sci. Tech.* 1:147) is derived for gas-phase (aerosol) filtration.
RT 1976 is the correct liquid-phase equivalent, including van der Waals and gravity terms.

---

## 6. Mesh Solidity Formula

```
α = (2 d_w / L) − (d_w / L)²
L = half-mesh size = opening / 2
```

Source: ScienceDirect filtration literature (user-provided 2025).
Design-default wire diameters flagged `[DESIGN_DEFAULT]` in code — replace with device specs.

---

## 7. Field Enhancement (Conical Geometry)

Triangular/pointed insulator geometry gives ~50× higher ∇|E|² than circular posts.
Source: PMC5507384 — Lapizco-Encinas group, iDEP insulator shape refinement.

Hemisphere-on-post approximation: β ≈ electrode_gap / tip_radius.
**This is an approximation.** Replace `DEPConfig.grad_E2` with FEM simulation value
for any quantitative hardware comparison.

---

## 8. Open Flags (Values Needing Further Grounding)

| Parameter | Status | Action Required |
|---|---|---|
| PET resistivity σ | Single source (Neagu 2002), moisture-sensitive | Seek IEEE TDEI paper for PET bulk resistivity |
| Mesh wire diameter d_w | [DESIGN_DEFAULT] — estimated | Replace with physical device measurement per stage |
| Cone half-angle optimal | No peer-reviewed number found | FEM simulation required for device geometry |
| DEP grad_E2 | Approximate (hemisphere model) | FEM simulation required |
| Fouling-capture coupling 0.10 | [DESIGN_DEFAULT] coefficient | Empirical calibration required |
