# 🚗 TrafficMind: Executive Functions Benchmark for Dynamic Route Planning

> **Measuring AGI Progress Hackathon** — Track: Executive Functions  
> Organized by Google DeepMind & Kaggle | 2026

---

## 📌 Overview

**TrafficMind** is a benchmark designed to evaluate **executive functions** in AI language models through realistic traffic and transportation scenarios from Colombia. 

Current AI models often "cheat" by memorizing patterns rather than truly reasoning. TrafficMind reveals *how* models think by placing them in dynamic, real-world situations where memorization alone is insufficient.

**Central Question:**
> *"What can this benchmark tell us about model behavior that we couldn't see before?"*

**Answer:** TrafficMind reveals whether a model can genuinely **plan, adapt, inhibit automatic responses, track multiple variables, and resolve conflicting priorities** — or whether it merely produces text that *sounds* correct.

---

## 🧠 Why Traffic & Transportation?

Driving and route management is a uniquely rich domain for evaluating executive functions because it **simultaneously requires ALL five executive functions**:
Real-world driving scenario
│
├── PLANNING → Multi-step route optimization with constraints
├── ADAPTATION → Responding to unexpected events mid-route
├── INHIBITORY CTRL → Overriding automatic habits with new rules
├── WORKING MEMORY → Tracking fuel, time, passengers, traffic
└── COGNITIVE FLEX → Prioritizing when objectives conflict

text

Unlike abstract reasoning puzzles, traffic scenarios have **clear correct answers**, **verifiable constraints**, and **measurable cognitive load** — making them ideal for benchmarking.

---

## 📋 The 5 Tasks

### Task 1 — Route Planning `[Executive Function: Planning]`

The model receives a transportation scenario with multiple simultaneous constraints and must generate an optimal step-by-step plan.

| Property | Value |
|---|---|
| Items | 10 (4 easy, 4 medium, 2 hard) |
| Weight | 20% |
| Cognitive Load | 2–5 variables |

**Example scenario:** *"It's Friday at 5:45 PM rush hour. An executive must reach the airport for a 7:30 PM flight. Route A is congested with roadworks. Route B is clear but longer. Route C is unknown. What do you do?"*

**What it reveals:** Can the model reason about multiple constraints simultaneously, or does it optimize only one variable at a time?

---

### Task 2 — Plan Disruption `[Executive Function: Adaptation & Flexibility]`

The model is mid-route when an unexpected event occurs (accident, mechanical failure, medical emergency). The original plan is no longer viable.

| Property | Value |
|---|---|
| Items | 10 (4 easy, 4 medium, 2 hard) |
| Weight | 25% ← Highest weight |
| Cognitive Load | 3–6 variables |

**Example scenario:** *"You're 67% through the route to the airport when a multi-vehicle accident blocks the highway ahead. The passenger's flight is in 1h 15min. Three alternative options exist with different time costs."*

**What it reveals:** Does the model update its world model when reality changes, or does it persist with the original plan? Can it generate viable alternatives under time pressure?

---

### Task 3 — Rule Reversal `[Executive Function: Inhibitory Control]`

The model faces modified or inverted traffic rules. It must inhibit its automatic habitual response and apply the new context-specific rule correctly.

| Property | Value |
|---|---|
| Items | 10 (3 easy, 4 medium, 3 hard) |
| Weight | 20% |
| Cognitive Load | 2–4 variables |

**Example scenario:** *"You're driving in a country where roundabout priority is REVERSED — entering vehicles have priority, not circulating ones. A car approaches to enter on your right. What do you do?"*

**What it reveals:** **Perseverative errors** — how often does the model apply habitual knowledge while ignoring explicit contextual instructions? This is a key indicator of true vs. apparent understanding.

---

### Task 4 — Multi-Variable Tracking `[Executive Function: Working Memory]`

The model must simultaneously monitor multiple dynamic variables (fuel levels, earnings, time, fleet status, passenger needs) and answer precise questions about each.

| Property | Value |
|---|---|
| Items | 10 (3 easy, 4 medium, 3 hard) |
| Weight | 15% |
| Cognitive Load | 3–6 variables |

**Example scenario:** *"You started with a full tank (50L), consuming 8L/hour for 3 hours. You completed 3 trips earning \$25K, \$18K, and \$32K. Your daily goal is 10 hours and it's now 9 AM. Answer: fuel remaining? total earnings? hours left to goal?"*

**What it reveals:** Performance degradation as cognitive load increases. At what point does the model start making calculation errors or forgetting previously stated information?

---

### Task 5 — Priority Conflict `[Executive Function: Cognitive Flexibility]`

The model faces genuinely conflicting objectives with no perfect solution. It must recognize the dilemma, establish an explicit priority hierarchy, and justify what it sacrifices.

| Property | Value |
|---|---|
| Items | 10 (3 easy, 4 medium, 3 hard) |
| Weight | 20% |
| Cognitive Load | 2–5 variables |

**Example scenario:** *"You have ONE vehicle available and TWO simultaneous urgent requests: 8 workers who will lose their daily wage if not transported in 45 min, vs. 1 executive with a \$500M contract meeting in 60 min. Taxis are available with 15-min wait."*

**What it reveals:** Can the model make value judgments under uncertainty and explicit tradeoffs, or does it only follow rigid rules? Does it find creative solutions that serve both parties?

---

## 📊 Dataset Summary

| Task | Total | Easy | Medium | Hard | Domain |
|---|---|---|---|---|---|
| Route Planning | 10 | 4 | 4 | 2 | Urban routing |
| Plan Disruption | 10 | 4 | 4 | 2 | Emergency adaptation |
| Rule Reversal | 10 | 3 | 4 | 3 | Regulatory contexts |
| Multi-Variable | 10 | 3 | 4 | 3 | Fleet coordination |
| Priority Conflict | 10 | 3 | 4 | 3 | Ethical transport decisions |
| **TOTAL** | **50** | **17** | **20** | **13** | **Colombian traffic** |

**Language:** Spanish (Colombian context)  
**Format:** Structured JSON with verifiable answers  
**Evaluation:** Semantic keyword matching + correct answer verification  
**Source:** Synthetically generated based on real Colombian traffic regulations and scenarios  

---

## 🏗️ Project Structure
trafficmind_benchmark/
│
├── benchmark.py # Main entry point & task definitions
├── utils.py # Evaluator, keyword maps & utilities
├── README.md # This file
│
├── tasks/
│ ├── init.py
│ ├── task1_planning.py # Route Planning task
│ ├── task2_disruption.py # Plan Disruption task
│ ├── task3_reversal.py # Rule Reversal task
│ ├── task4_tracking.py # Multi-Variable Tracking task
│ └── task5_priority.py # Priority Conflict task
│
└── data/
├── task1_route_planning.json # 10 items
├── task2_plan_disruption.json # 10 items
├── task3_rule_reversal.json # 10 items
├── task4_multi_variable.json # 10 items
└── task5_priority_conflict.json # 10 items

text

---

## 🚀 Usage

```bash
# Install dependencies
pip install kaggle-benchmarks python-dotenv pandas

# Validate locally (no API required — uses optimal answers as ground truth)
python benchmark.py --validate

# Run full benchmark with AI model
python benchmark.py --run

# Run with parallel jobs
python benchmark.py --run --jobs 4

# Run specific task only
python benchmark.py --run --task 1
Environment Setup
bash
# .env file required for API access
MODEL_PROXY_URL=https://generativelanguage.googleapis.com/v1beta/openai/
MODEL_PROXY_API_KEY=your_api_key_here
LLM_DEFAULT=gemini-2.0-flash
LLM_DEFAULT_JUDGE=gemini-2.0-flash
LLMS_AVAILABLE=gemini-2.0-flash
📈 Scoring System
text
Final Score = Σ (task_pass_rate × task_weight)

Task Weights:
  Route Planning      →  20%
  Plan Disruption     →  25%  (highest — most cognitively complex)
  Rule Reversal       →  20%
  Multi-Variable      →  15%
  Priority Conflict   →  20%

Item Pass Threshold:  ≥ 60% of expected elements covered
Discriminatory Power (Validated Locally)
Task	Optimal Answer Score	Expected Random Score	Gap
Route Planning	65% avg	~15%	50pp
Plan Disruption	63% avg	~12%	51pp
Rule Reversal	83% avg	~10%	73pp
Multi-Variable	97% avg	~20%	77pp
Priority Conflict	63% avg	~12%	51pp
Overall pass rate with optimal answers: 78%
✅ No task scores 0% or 100% → Strong discriminatory power between models

🔬 Research Hypotheses
Based on the benchmark structure, we predict the following model behavior patterns:

Models will perform best on Task 1 (static planning with complete information)
Task 3 will reveal the most perseverative errors — models applying habitual rules despite explicit instructions to the contrary
Task 4 performance will degrade linearly with cognitive_load — revealing working memory limits
Task 2 will show the largest variance between models — adaptation is where model quality diverges most
Task 5 will reveal whether models can make genuine value judgments vs. pattern-matching to "safe" responses
These hypotheses make TrafficMind useful not just as a benchmark but as a research tool for understanding model cognition.

🧪 Evaluation Methodology
Each item contains:

scenario: Detailed situation description with all relevant variables
question: Specific question requiring executive function engagement
expected_elements: List of cognitive elements the response must address
optimal_answer / correct_answer: Ground truth reference
cognitive_load: Integer 1–6 rating of variable complexity
difficulty: easy / medium / hard classification
Evaluation pipeline:

text
Model Response
      │
      ▼
Text Normalization (accent removal, punctuation cleaning)
      │
      ▼
Semantic Keyword Matching per expected_element
      │
      ▼
Coverage Score = matched_elements / total_elements
      │
      ▼
Pass/Fail: Coverage ≥ 0.60 → PASS



📚 References
Diamond, A. (2013). Executive Functions. Annual Review of Psychology, 64, 135–168.
Morris, M. R., et al. (2023). Levels of AGI: Operationalizing Progress on the Path to AGI. Google DeepMind.
Miyake, A., et al. (2000). The Unity and Diversity of Executive Functions. Cognitive Psychology, 41(1), 49–100.
Colombian National Traffic Code (Código Nacional de Tránsito Terrestre — Ley 769 de 2002).



👤 Author & Affiliation
TrafficMind Benchmark
Developed for the "Measuring Progress toward AGI — Cognitive Skills" Hackathon
Organized by Google DeepMind & Kaggle | March–June 2026
Developer: Jhon Tailor Alvarez 

Track: Executive Functions
Domain: Traffic & Transportation (Colombian context)
Country: Colombia 🇨🇴

