<!-- ===================================================== -->
<!-- ================== TRAFFICMIND ====================== -->
<!-- ===================================================== -->

<p align="center">
  <img src="https://media.giphy.com/media/3o7TKtnuHOHHUjR38Y/giphy.gif" width="750"/>
</p>

<h1 align="center">🚦🧠 TrafficMind</h1>

<h3 align="center">
Executive Functions Benchmark for Real‑World Adaptive Intelligence
</h3>

<p align="center">
<b>Three mountain ranges. Dynamic cities. Regulatory complexity. Ethical tradeoffs.</b><br>
Can your model truly <i>adapt</i> — or does it only autocomplete?
</p>

---

<p align="center">

![Track](https://img.shields.io/badge/Track-Executive_Functions-1f77b4)
![Items](https://img.shields.io/badge/Items-50-success)
![Difficulty](https://img.shields.io/badge/Difficulty-3_Tiers-orange)
![Cognitive_Load](https://img.shields.io/badge/Cognitive_Load-1_to_6-purple)
![Language](https://img.shields.io/badge/Language-Spanish-yellow)
![Context](https://img.shields.io/badge/Context-Colombia🇨🇴-red)

</p>

<p align="center">
<b>Measuring AGI Progress Hackathon</b><br>
Google DeepMind × Kaggle | 2026
</p>

---

# 🌎 Why Colombia?

Colombia is not a grid.

It is:

- 🏔️ Three Andean mountain ranges  
- 🌧️ Sudden landslides  
- 🚧 Infrastructure asymmetry  
- 🛵 Mixed traffic ecosystems  
- 📜 Complex regulatory framework (Ley 769 de 2002)  
- ⏱️ Time-sensitive urban congestion  

Planning here is not static optimization.

It is **continuous adaptive control under environmental uncertainty**.

TrafficMind forces models to reason within terrain, law, time pressure, and ethical tradeoffs — simultaneously.

---

# 🎯 Core Scientific Question

> **What aspects of executive cognition in large language models remain hidden under standard benchmarks?**

Traditional benchmarks measure:
- Logical reasoning  
- Mathematical manipulation  
- Code generation  

TrafficMind measures:

> Executive control under dynamic constraint revision.

It isolates five executive functions grounded in cognitive psychology.

---

# 🧠 The Five Executive Functions

```
Dynamic Driving Scenario
│
├── PLANNING              → Multi-step constrained optimization
├── ADAPTATION            → World-model updating under disruption
├── INHIBITORY CONTROL    → Overriding habitual priors
├── WORKING MEMORY        → Multi-variable tracking under load
└── CONFLICT RESOLUTION   → Structured prioritization without perfect solutions
```

These functions are based on:

- Miyake et al. (2000) — Unity & Diversity of Executive Functions  
- Diamond (2013) — Executive Control Theory  
- Morris et al. (2023) — Levels of AGI  

---

# 🚗 Benchmark Structure

## Task 1 — Route Planning  
**Executive Function: Planning**

Multi-step optimization with simultaneous constraints.

| Items | 10 |
| Weight | 20% |
| Cognitive Load | 2–5 |

Reveals: sequential reasoning vs single-variable shortcuts.

---

## 🚧 Task 2 — Plan Disruption  
**Executive Function: Cognitive Flexibility**

Mid-route failure invalidates original plan.

| Items | 10 |
| Weight | 25% (highest) |
| Cognitive Load | 3–6 |

Reveals: whether the model updates internal state or persists with outdated reasoning.

---

## 🔄 Task 3 — Rule Reversal  
**Executive Function: Inhibitory Control**

Explicit override of learned global priors.

| Items | 10 |
| Weight | 20% |
| Cognitive Load | 2–4 |

Measures perseverative error rate — a core indicator of rigid cognition.

---

## 🧮 Task 4 — Multi‑Variable Tracking  
**Executive Function: Working Memory**

Simultaneous numeric + semantic load.

| Items | 10 |
| Weight | 15% |
| Cognitive Load | 3–6 |

Detects dissociation between:

- Verbal reasoning fluency  
- Numerical accuracy under pressure  

---

## ⚖️ Task 5 — Priority Conflict  
**Executive Function: Structured Value Judgment**

No perfect solution exists.

| Items | 10 |
| Weight | 20% |
| Cognitive Load | 2–5 |

Requires:

1. Dilemma recognition  
2. Priority hierarchy  
3. Explicit justification  
4. Tradeoff acknowledgment  

---

# 📊 Dataset Composition

| Task | Easy | Medium | Hard | Total |
|------|------|--------|------|------|
| Route Planning | 4 | 4 | 2 | 10 |
| Plan Disruption | 4 | 4 | 2 | 10 |
| Rule Reversal | 3 | 4 | 3 | 10 |
| Multi-Variable | 3 | 4 | 3 | 10 |
| Priority Conflict | 3 | 4 | 3 | 10 |
| **TOTAL** | **17** | **20** | **13** | **50** |

✅ 3 difficulty tiers  
✅ Cognitive load scaling (1–6)  
✅ Structured JSON format  
✅ Verifiable ground truth  

---

# 📈 Scoring Framework

Final Score:

```
Σ (task_pass_rate × task_weight)
```

Item Pass Condition:

```
Coverage ≥ 60% of expected cognitive elements
```

Task 4 Combined Scoring:

```
Final = (Semantic × 0.4) + (Numeric Accuracy × 0.6)
```

This prevents superficial keyword gaming.

---

# 🔬 Evaluation Pipeline

```
Scenario
   ↓
Model Response
   ↓
Normalization
   ↓
Semantic Element Matching
   ↓
Task-Specific Logic
   ↓
PASS / FAIL
   ↓
Weighted Aggregate Score
```

---

# 📉 Cognitive Load Scaling

TrafficMind explicitly measures degradation under increasing variable complexity.

Hypothesis:

```
Accuracy ∝ 1 / Cognitive_Load
```

We expect nonlinear performance decay beyond load ≥ 4.

This transforms the benchmark from static scoring into **cognitive stress testing**.

---

# 🧪 Error Taxonomy (Phase II)

TrafficMind enables classification of:

- Perseverative errors  
- Numerical hallucinations  
- Constraint omission  
- Over-generalized safety responses  
- Binary simplification of multi-party conflicts  

This allows structural model comparison beyond raw accuracy.

---

# 🧬 Why This Matters for AGI Evaluation

AGI is not:

- Static reasoning
- Perfect recall
- Mathematical speed

AGI requires:

- Updating plans when reality changes  
- Overriding internal priors  
- Tracking evolving variables  
- Making structured tradeoffs  

TrafficMind operationalizes these dimensions.

---

# 🚀 Installation

```bash
pip install kaggle-benchmarks python-dotenv pandas

python benchmark.py --validate
python benchmark.py --run
python benchmark.py --run --jobs 4
python benchmark.py --run --task 2
```

---

# 🗂 Project Structure

```
trafficmind_benchmark/
│
├── benchmark.py
├── utils.py
├── tasks/
├── data/
└── README.md
```

Modular. Deterministic. Reproducible.

---

# 📚 Scientific References

- Diamond, A. (2013). Executive Functions. Annual Review of Psychology.  
- Miyake, A., et al. (2000). Unity and Diversity of Executive Functions.  
- Morris, M. R., et al. (2023). Levels of AGI. Google DeepMind.  
- Chollet, F. (2019). On the Measure of Intelligence.  
- Colombian National Traffic Code — Ley 769 de 2002.  

---

# 👤 Author

**Jhon Tailor Alvarez**  
Independent Researcher  
Colombia 🇨🇴  

Developed for:

**Measuring Progress Toward AGI — Cognitive Skills Hackathon**  
Google DeepMind × Kaggle | 2026  

---

# 🏁 Final Reflection

In Colombia,  
a route is optimal  
only until the mountain moves.

If your model cannot adapt,  
it does not understand.

It predicts.
