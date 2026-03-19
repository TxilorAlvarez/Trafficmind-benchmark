"""
╔══════════════════════════════════════════════════════════════════╗
║          TrafficMind Benchmark  ·  v2.0.0                        ║
║          Executive Functions for Dynamic Route Planning           ║
║                                                                   ║
║  Track   : Executive Functions                                    ║
║  Hackathon: Measuring Progress Toward AGI – Cognitive Abilities   ║
║  Organized: Google DeepMind × Kaggle  |  2026                    ║
║  Author  : Jhon Tailor Alvarez  🇨🇴                               ║
╚══════════════════════════════════════════════════════════════════╝

Uso:
  python benchmark.py --validate
  python benchmark.py --run --model gpt-4o-mini
  python benchmark.py --run --model gpt-4o
  python benchmark.py --run --model gpt-4.1
  python benchmark.py --compare
  python benchmark.py --list-models
"""

# ─────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────
import os
import re
import sys
import json
import glob
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # sin GUI — compatible con servidor/CI
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────
# DIRECTORIOS BASE
# ─────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).parent
DATA_DIR     = ROOT_DIR / "data"
RESULTS_DIR  = ROOT_DIR / "results"
ANALYSIS_DIR = ROOT_DIR / "analysis"
SDK_CACHE    = ROOT_DIR / "sdk_cache"

for _d in [DATA_DIR, RESULTS_DIR, ANALYSIS_DIR, SDK_CACHE]:
    _d.mkdir(exist_ok=True)

# Indicar al SDK dónde guardar outputs (si lo soporta)
os.environ.setdefault("KAGGLE_BENCHMARKS_OUTPUT_DIR", str(SDK_CACHE))

# ─────────────────────────────────────────────────────────────────
# LOGGING  (consola + archivo)
# ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT_DIR / "trafficmind.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# KAGGLE BENCHMARKS SDK + UTILIDADES
# ─────────────────────────────────────────────────────────────────
import kaggle_benchmarks as kb
from utils import (
    load_json_data,
    evaluate_expected_elements,
    get_answer_field,
)

# ─────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────
TASK_WEIGHTS = {
    "route_planning":    0.20,
    "plan_disruption":   0.25,
    "rule_reversal":     0.20,
    "multi_variable":    0.15,
    "priority_conflict": 0.20,
}

SUPPORTED_MODELS = [
    "gpt-4o-mini",        # Baseline económico
    "gpt-4o",             # Frontier principal
    "gpt-4.1",            # Más reciente
    # "gemini-2.0-flash", # Descomentar si disponible en Kaggle quota
    # "claude-3-haiku",   # Descomentar si disponible
]


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def configure_model(model_name: str) -> None:
    """
    Activa el modelo sin tocar código.
    El SDK de Kaggle respeta LLM_DEFAULT como modelo activo.
    """
    os.environ["LLM_DEFAULT"]       = model_name
    os.environ["LLM_DEFAULT_JUDGE"] = model_name

    logger.info("─" * 60)
    logger.info(f"  🔎 Modelo activo  : {model_name}")
    logger.info(f"  🕐 Timestamp      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("─" * 60)


def _cleanup_sdk_files() -> None:
    """
    Mueve los archivos *.run.json y *.task.json generados
    automáticamente por el SDK al directorio sdk_cache/.
    """
    moved = 0
    for pattern in ["*.run.json", "*.task.json"]:
        for f in ROOT_DIR.glob(pattern):
            dest = SDK_CACHE / f.name
            f.rename(dest)
            moved += 1
    if moved:
        logger.info(f"  🧹 {moved} archivos SDK → sdk_cache/")


# ═══════════════════════════════════════════════════════════════
# PREPARADORES DE DATAFRAMES
# Las columnas de cada DataFrame deben coincidir EXACTAMENTE
# con los parámetros de la task function correspondiente.
# ═══════════════════════════════════════════════════════════════

def prepare_df_task1() -> pd.DataFrame:
    """Route Planning — Planning."""
    data = load_json_data("task1_route_planning.json")
    rows = []
    for item in data:
        sc = item["scenario"]
        md = sc.get("map_data", {})
        rows.append({
            "task_id": item["task_id"],
            "scenario": (
                f"{sc.get('description', '')}\n"
                f"Origen: {md.get('origin', '')}\n"
                f"Destino: {md.get('destination', '')}\n"
                f"Restricciones: {sc.get('constraints', [])}"
            ),
            "question":           item["question"],
            "expected_elements":  item["expected_elements"],
            "difficulty":         item["difficulty"],
            "cognitive_load":     item.get("cognitive_load", 3),
        })
    return pd.DataFrame(rows)


def prepare_df_task2() -> pd.DataFrame:
    """Plan Disruption — Cognitive Flexibility."""
    data = load_json_data("task2_plan_disruption.json")
    rows = []
    for item in data:
        sc      = item["scenario"]
        plan    = sc.get("initial_plan", {})
        disrupt = sc.get("disruption", {})
        alts    = sc.get("alternatives", [])
        scenario_text = (
            f"PLAN INICIAL: {plan.get('description', '')}\n"
            f"Progreso: {plan.get('progress', '')}\n"
            f"Tiempo restante: {plan.get('remaining_time', '')}\n\n"
            f"EVENTO INESPERADO — {disrupt.get('type', '').upper()}:\n"
            f"{disrupt.get('description', '')}\n"
            f"Tu posición: {disrupt.get('your_position', '')}\n\n"
            "OPCIONES DISPONIBLES:\n" +
            "\n".join(
                f"  • {a.get('name', '')}: {a.get('description', '')} "
                f"(+{a.get('estimated_additional_time', '')})"
                for a in alts
            )
        )
        rows.append({
            "task_id":            item["task_id"],
            "scenario":           scenario_text,
            "question":           item["question"],
            "expected_elements":  item["expected_elements"],
            "difficulty":         item["difficulty"],
            "cognitive_load":     item.get("cognitive_load", 3),
        })
    return pd.DataFrame(rows)


def prepare_df_task3() -> pd.DataFrame:
    """Rule Reversal — Inhibitory Control."""
    data = load_json_data("task3_rule_reversal.json")
    rows = []
    for item in data:
        sc = item["scenario"]
        scenario_text = (
            f"Contexto: {sc.get('context', '')}\n"
            f"Regla normal: {sc.get('normal_rule', '')}\n"
            f"REGLA ESPECIAL (aplica ESTA): {sc.get('reversed_rule', '')}\n"
            f"Situación: {sc.get('situation', '')}"
        )
        rows.append({
            "task_id":            item["task_id"],
            "scenario":           scenario_text,
            "question":           item["question"],
            "options":            item.get("options", []),
            "correct_answer":     item["correct_answer"],
            "expected_elements":  item["expected_elements"],
            "difficulty":         item["difficulty"],
            "cognitive_load":     item.get("cognitive_load", 3),
        })
    return pd.DataFrame(rows)


def prepare_df_task4() -> pd.DataFrame:
    """Multi-Variable Tracking — Working Memory."""
    data = load_json_data("task4_multi_variable.json")
    rows = []
    for item in data:
        sc = item["scenario"]
        scenario_text = (
            f"{sc.get('description', '')}\n\n"
            "VARIABLES A RASTREAR:\n"
            f"{json.dumps(sc.get('variables_to_track', sc), ensure_ascii=False, indent=2)}"
        )
        rows.append({
            "task_id":            item["task_id"],
            "scenario":           scenario_text,
            "questions":          item.get("questions", []),
            "expected_elements":  item["expected_elements"],
            "difficulty":         item["difficulty"],
            "cognitive_load":     item.get("cognitive_load", 3),
        })
    return pd.DataFrame(rows)


def prepare_df_task5() -> pd.DataFrame:
    """Priority Conflict — Conflict Resolution."""
    data = load_json_data("task5_priority_conflict.json")
    rows = []
    for item in data:
        sc = item["scenario"]
        scenario_text = (
            f"{sc.get('description', '')}\n\n"
            f"{json.dumps({k: v for k, v in sc.items() if k != 'description'}, ensure_ascii=False, indent=2)}"
        )
        rows.append({
            "task_id":            item["task_id"],
            "scenario":           scenario_text,
            "question":           item["question"],
            "expected_elements":  item["expected_elements"],
            "difficulty":         item["difficulty"],
            "cognitive_load":     item.get("cognitive_load", 3),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# TASK FUNCTIONS
# IMPORTANTE: los parámetros deben coincidir con las columnas
# del DataFrame que se le pasa a task.evaluate(evaluation_data=df)
# ═══════════════════════════════════════════════════════════════

@kb.task(
    name="Route Planning",
    description=(
        "Evalúa planificación multi-paso con restricciones simultáneas: "
        "tiempo, combustible, zonas escolares, calidad de vías. "
        "Función ejecutiva: PLANIFICACIÓN."
    ),
    version=1,
)
def route_planning_task(
    task_id: str,
    scenario: str,
    question: str,
    expected_elements: list,
    difficulty: str,
    cognitive_load: int,
) -> bool:

    prompt = f"""Eres un experto en planificación de rutas de transporte colombiano.

## ESCENARIO
{scenario}

## PREGUNTA
{question}

## INSTRUCCIONES
1. Analiza TODAS las opciones disponibles
2. Considera TODAS las restricciones mencionadas
3. Selecciona la ruta óptima con justificación clara
4. Proporciona un plan paso a paso
5. Calcula tiempos de llegada cuando sea relevante

Responde en español de forma clara y estructurada."""

    response   = kb.llm.prompt(prompt)
    evaluation = evaluate_expected_elements(response, expected_elements)

    logger.info(
        f"  [{task_id}] Planning [{difficulty}] CL:{cognitive_load} "
        f"Score:{evaluation['score']:.2f} | "
        f"{'✅ PASS' if evaluation['passed'] else '❌ FAIL'}"
    )
    return evaluation["passed"]


@kb.task(
    name="Plan Disruption",
    description=(
        "El plan está en ejecución cuando ocurre un evento inesperado. "
        "El modelo debe detectar el fallo y generar un plan alternativo. "
        "Función ejecutiva: ADAPTACIÓN Y FLEXIBILIDAD."
    ),
    version=1,
)
def plan_disruption_task(
    task_id: str,
    scenario: str,
    question: str,
    expected_elements: list,
    difficulty: str,
    cognitive_load: int,
) -> bool:

    prompt = f"""Eres un conductor experto en rutas colombianas. Estás en medio
de un recorrido cuando ocurre algo inesperado que invalida tu plan original.

## SITUACIÓN ACTUAL
{scenario}

## PREGUNTA
{question}

## DEBES
1. Reconocer POR QUÉ el plan original ya NO es viable
2. Evaluar las alternativas disponibles
3. Proponer un plan alternativo concreto y viable
4. Calcular nuevo tiempo estimado de llegada
5. Comunicar la situación con calma y profesionalismo

Responde en español de forma clara y directa."""

    response   = kb.llm.prompt(prompt)
    evaluation = evaluate_expected_elements(response, expected_elements)

    logger.info(
        f"  [{task_id}] Disruption [{difficulty}] CL:{cognitive_load} "
        f"Score:{evaluation['score']:.2f} | "
        f"{'✅ PASS' if evaluation['passed'] else '❌ FAIL'}"
    )
    return evaluation["passed"]


@kb.task(
    name="Rule Reversal",
    description=(
        "El modelo enfrenta reglas de tránsito modificadas. "
        "Debe inhibir la respuesta habitual y aplicar la regla nueva. "
        "Función ejecutiva: CONTROL INHIBITORIO."
    ),
    version=1,
)
def rule_reversal_task(
    task_id: str,
    scenario: str,
    question: str,
    options: list,
    correct_answer: str,
    expected_elements: list,
    difficulty: str,
    cognitive_load: int,
) -> bool:

    options_text = "\n".join(options) if options else "N/A"

    prompt = f"""Estás conduciendo en un contexto donde las reglas de tránsito
NORMALES HAN SIDO MODIFICADAS para este escenario específico.

## ESCENARIO
{scenario}

## OPCIONES
{options_text}

## PREGUNTA
{question}

## INSTRUCCIONES CRÍTICAS
- Las reglas especiales de este escenario ANULAN las reglas habituales
- Selecciona la opción correcta según las reglas de ESTE contexto
- Explica qué regla especial aplicas y por qué ignoras la regla normal

Indica la letra de la opción elegida y justifica tu decisión."""

    response = kb.llm.prompt(prompt)

    # ── Detectar si seleccionó la opción correcta ────────────
    r = response.lower()
    ca = correct_answer.lower()
    correct_selected = any([
        f"{ca})" in r,
        f"opción {ca}" in r,
        f"opcion {ca}" in r,
        f"**{ca})" in r,
        f"letra {ca}" in r,
        r.strip().startswith(ca),
    ])

    evaluation     = evaluate_expected_elements(response, expected_elements)
    semantic_score = evaluation["score"]

    # Scoring balanceado: semántica alta → pasa; semántica media + correcto → pasa
    if semantic_score >= 0.80:
        final_passed = True
    elif semantic_score >= 0.60 and correct_selected:
        final_passed = True
    else:
        final_passed = False

    logger.info(
        f"  [{task_id}] Reversal [{difficulty}] CL:{cognitive_load} "
        f"Sem:{semantic_score:.2f} Correct:{correct_selected} | "
        f"{'✅ PASS' if final_passed else '❌ FAIL'}"
    )
    return final_passed


@kb.task(
    name="Multi Variable Tracking",
    description=(
        "El modelo monitorea múltiples variables simultáneas "
        "(combustible, tiempo, dinero, pasajeros) y calcula con precisión. "
        "Función ejecutiva: MEMORIA DE TRABAJO."
    ),
    version=1,
)
def multi_variable_task(
    task_id: str,
    scenario: str,
    questions: list,
    expected_elements: list,
    difficulty: str,
    cognitive_load: int,
) -> bool:

    questions_text = "\n".join(
        f"{i+1}. {q.get('q', str(q))}"
        for i, q in enumerate(questions)
    )

    prompt = f"""Eres un coordinador de transporte monitoreando múltiples
variables en tiempo real. Debes ser PRECISO con los números y cálculos.

## ESCENARIO Y VARIABLES
{scenario}

## PREGUNTAS (responde cada una por separado, con cálculo explícito)
{questions_text}

Responde en español mostrando cada operación matemática paso a paso."""

    response   = kb.llm.prompt(prompt)
    evaluation = evaluate_expected_elements(response, expected_elements)

    # ── Evaluación numérica estricta ─────────────────────────
    correct_answers = [
        q.get("correct", "") for q in questions if isinstance(q, dict)
    ]
    numeric_matches = 0
    for correct in correct_answers:
        nums_correct  = re.findall(r'\d+[\.,]?\d*', correct.replace(",", ""))
        nums_response = re.findall(r'\d+[\.,]?\d*', response.replace(",", ""))
        key_nums = [n for n in nums_correct if len(n) >= 2]
        if key_nums:
            hits = sum(1 for n in key_nums if n in nums_response)
            if hits / len(key_nums) >= 0.6:
                numeric_matches += 1

    numeric_acc  = numeric_matches / len(correct_answers) if correct_answers else 0
    combined     = (evaluation["score"] * 0.4) + (numeric_acc * 0.6)
    final_passed = combined >= 0.65

    logger.info(
        f"  [{task_id}] MultiVar [{difficulty}] CL:{cognitive_load} "
        f"Combined:{combined:.2f} Sem:{evaluation['score']:.2f} Num:{numeric_acc:.2f} | "
        f"{'✅ PASS' if final_passed else '❌ FAIL'}"
    )
    return final_passed


@kb.task(
    name="Priority Conflict",
    description=(
        "El modelo enfrenta objetivos simultáneos en conflicto directo. "
        "Debe reconocer el dilema, priorizar y justificar el sacrificio. "
        "Función ejecutiva: FLEXIBILIDAD COGNITIVA / RESOLUCIÓN DE CONFLICTOS."
    ),
    version=1,
)
def priority_conflict_task(
    task_id: str,
    scenario: str,
    question: str,
    expected_elements: list,
    difficulty: str,
    cognitive_load: int,
) -> bool:

    prompt = f"""Eres un tomador de decisiones en transporte colombiano
enfrentando objetivos que entran en conflicto directo entre sí.

## SITUACIÓN CON CONFLICTO DE PRIORIDADES
{scenario}

## PREGUNTA
{question}

## DEBES DEMOSTRAR
1. Reconocer EXPLÍCITAMENTE el conflicto entre objetivos
2. Listar los objetivos en tensión
3. Establecer una jerarquía de prioridades con criterio claro
4. Tomar una decisión definitiva y justificarla
5. Explicar qué sacrificas y por qué vale la pena
6. Explorar si existe solución creativa que sirva a ambas partes

Responde en español siendo explícito sobre el conflicto y tu razonamiento."""

    response   = kb.llm.prompt(prompt)
    evaluation = evaluate_expected_elements(response, expected_elements)

    logger.info(
        f"  [{task_id}] Conflict [{difficulty}] CL:{cognitive_load} "
        f"Score:{evaluation['score']:.2f} | "
        f"{'✅ PASS' if evaluation['passed'] else '❌ FAIL'}"
    )
    return evaluation["passed"]


# ═══════════════════════════════════════════════════════════════
# RUNNER PRINCIPAL
# ═══════════════════════════════════════════════════════════════

TASK_REGISTRY = [
    {
        "key":    "route_planning",
        "label":  "📍 Task 1: Route Planning",
        "fn":     route_planning_task,
        "df_fn":  prepare_df_task1,
        "weight": TASK_WEIGHTS["route_planning"],
    },
    {
        "key":    "plan_disruption",
        "label":  "🚧 Task 2: Plan Disruption",
        "fn":     plan_disruption_task,
        "df_fn":  prepare_df_task2,
        "weight": TASK_WEIGHTS["plan_disruption"],
    },
    {
        "key":    "rule_reversal",
        "label":  "🔄 Task 3: Rule Reversal",
        "fn":     rule_reversal_task,
        "df_fn":  prepare_df_task3,
        "weight": TASK_WEIGHTS["rule_reversal"],
    },
    {
        "key":    "multi_variable",
        "label":  "📊 Task 4: Multi-Variable",
        "fn":     multi_variable_task,
        "df_fn":  prepare_df_task4,
        "weight": TASK_WEIGHTS["multi_variable"],
    },
    {
        "key":    "priority_conflict",
        "label":  "⚖️  Task 5: Priority Conflict",
        "fn":     priority_conflict_task,
        "df_fn":  prepare_df_task5,
        "weight": TASK_WEIGHTS["priority_conflict"],
    },
]


def run_all_tasks(model_name: str = "gpt-4o-mini", n_jobs: int = 1) -> dict:
    """
    Ejecuta las 5 tasks contra el modelo indicado.
    Guarda:
      - results/{model}_summary_{ts}.csv   → pass_rate por task
      - results/{model}_detailed_{ts}.csv  → resultado por item
    Retorna dict con métricas completas.
    """
    configure_model(model_name)

    logger.info("=" * 60)
    logger.info("  🚗 TrafficMind Benchmark — Iniciando evaluación")
    logger.info("=" * 60)

    summary_rows  = []
    detailed_rows = []

    for task in TASK_REGISTRY:
        logger.info(f"\n{task['label']}")

        df = task["df_fn"]()
        logger.info(f"  📂 Cargados {len(df)} items")

        runs   = task["fn"].evaluate(evaluation_data=df, n_jobs=n_jobs)
        df_res = runs.as_dataframe()

        pass_rate = float(df_res["result"].mean()) if len(df_res) > 0 else 0.0

        # ── Resumen por task ──────────────────────────────────
        summary_rows.append({
            "model":     model_name,
            "task":      task["key"],
            "pass_rate": pass_rate,
            "weight":    task["weight"],
            "weighted":  pass_rate * task["weight"],
        })

        # ── Detalle por item ──────────────────────────────────
        for idx in range(len(df)):
            result_val = bool(df_res.iloc[idx]["result"]) if idx < len(df_res) else False
            detailed_rows.append({
                "model":         model_name,
                "task":          task["key"],
                "task_id":       df.iloc[idx]["task_id"],
                "difficulty":    df.iloc[idx]["difficulty"],
                "cognitive_load": df.iloc[idx]["cognitive_load"],
                "passed":        result_val,
            })

        logger.info(f"  ✅ Pass rate: {pass_rate:.2%}")

    # ── Score Agregado ────────────────────────────────────────
    aggregate = sum(r["weighted"] for r in summary_rows)

    # ── Persistir resultados ──────────────────────────────────
    ts         = datetime.now().strftime("%Y%m%d_%H%M")
    safe_model = model_name.replace("/", "_").replace(".", "_")

    summary_path  = RESULTS_DIR / f"{safe_model}_summary_{ts}.csv"
    detailed_path = RESULTS_DIR / f"{safe_model}_detailed_{ts}.csv"

    pd.DataFrame(summary_rows).to_csv(summary_path,  index=False)
    pd.DataFrame(detailed_rows).to_csv(detailed_path, index=False)

    # ── Reporte final ─────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  📊 RESULTADOS FINALES — TrafficMind Benchmark")
    logger.info("=" * 60)
    for r in summary_rows:
        bar = "█" * int(r["pass_rate"] * 20)
        logger.info(
            f"  {r['task']:<25} {bar:<22} "
            f"{r['pass_rate']:.2%}  (w:{r['weight']})"
        )
    logger.info(f"\n  🏆 AGGREGATE SCORE : {aggregate:.4f}")
    logger.info(f"  💾 Summary         : {summary_path}")
    logger.info(f"  💾 Detailed        : {detailed_path}")
    logger.info("=" * 60)

    # ── Limpiar archivos SDK ──────────────────────────────────
    _cleanup_sdk_files()

    return {
        "model":     model_name,
        "aggregate": aggregate,
        "summary":   summary_rows,
        "detailed":  detailed_rows,
    }


# ═══════════════════════════════════════════════════════════════
# VALIDACIÓN LOCAL  (sin API — usa optimal_answer de los JSON)
# ═══════════════════════════════════════════════════════════════

def validate_local() -> bool:
    """
    Valida el pipeline completo sin consumir cuota de API.
    Usa las respuestas óptimas de los JSON como respuesta mock.
    Confirma que el evaluador y el dataset son correctos.
    """
    logger.info("\n" + "=" * 60)
    logger.info("  🔍 VALIDACIÓN LOCAL — TrafficMind Benchmark")
    logger.info("=" * 60)

    tasks_info = [
        ("Task 1 – Route Planning",    "task1_route_planning.json"),
        ("Task 2 – Plan Disruption",   "task2_plan_disruption.json"),
        ("Task 3 – Rule Reversal",     "task3_rule_reversal.json"),
        ("Task 4 – Multi-Variable",    "task4_multi_variable.json"),
        ("Task 5 – Priority Conflict", "task5_priority_conflict.json"),
    ]

    all_ok      = True
    total_items = 0
    total_pass  = 0

    for task_name, filename in tasks_info:
        data   = load_json_data(filename)
        passed = 0
        scores = []

        for item in data:
            answer     = get_answer_field(item)
            evaluation = evaluate_expected_elements(
                response=answer,
                expected_elements=item["expected_elements"],
            )
            scores.append(evaluation["score"])
            if evaluation["passed"]:
                passed += 1

        avg_score = sum(scores) / len(scores) if scores else 0
        pass_rate = passed / len(data)
        ok        = avg_score >= 0.60

        if not ok:
            all_ok = False

        icon = "✅" if ok else "⚠️ "
        logger.info(
            f"  {icon} {task_name:<33} "
            f"avg={avg_score:.2f}  pass={pass_rate:.0%}  n={len(data)}"
        )

        total_items += len(data)
        total_pass  += passed

    overall = total_pass / total_items if total_items else 0
    logger.info(f"\n  📊 Total items    : {total_items}")
    logger.info(f"  📊 Overall pass   : {overall:.2%}")
    status = "✅ Listo para producción" if all_ok else "⚠️  Revisar tasks con warning"
    logger.info(f"  {status}")
    logger.info("=" * 60)

    return all_ok


# ═══════════════════════════════════════════════════════════════
# ANÁLISIS Y GRÁFICOS COMPARATIVOS
# ═══════════════════════════════════════════════════════════════

def _load_latest_summaries() -> pd.DataFrame:
    """Carga el CSV summary más reciente por cada modelo."""
    files  = sorted(glob.glob(str(RESULTS_DIR / "*_summary_*.csv")))
    latest: dict = {}
    for f in files:
        stem  = Path(f).stem                     # e.g. gpt-4o-mini_summary_20260319_0212
        parts = stem.split("_summary_")
        model = parts[0] if parts else stem
        latest[model] = f                        # sobrescribe con el más reciente

    frames = [pd.read_csv(p) for p in latest.values()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_latest_detailed() -> pd.DataFrame:
    """Carga el CSV detailed más reciente por cada modelo."""
    files  = sorted(glob.glob(str(RESULTS_DIR / "*_detailed_*.csv")))
    latest: dict = {}
    for f in files:
        stem  = Path(f).stem
        parts = stem.split("_detailed_")
        model = parts[0] if parts else stem
        latest[model] = f

    frames = [pd.read_csv(p) for p in latest.values()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def generate_comparison_charts() -> None:
    """
    Genera 3 gráficos científicos a partir de los resultados guardados:
      1. Radar chart comparativo por modelo
      2. Cognitive Load vs Accuracy (curva de degradación)
      3. Accuracy por nivel de dificultad
    """
    ANALYSIS_DIR.mkdir(exist_ok=True)

    df_sum = _load_latest_summaries()
    df_det = _load_latest_detailed()

    if df_sum.empty:
        logger.warning(
            "⚠️  No hay resultados. Ejecuta primero: "
            "python benchmark.py --run --model gpt-4o-mini"
        )
        return

    models = df_sum["model"].unique().tolist()
    logger.info(f"\n📊 Modelos detectados: {models}")

    COLORS    = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
    color_map = {m: COLORS[i % len(COLORS)] for i, m in enumerate(models)}

    TASK_LABELS = {
        "route_planning":    "Route\nPlanning",
        "plan_disruption":   "Plan\nDisruption",
        "rule_reversal":     "Rule\nReversal",
        "multi_variable":    "Multi\nVariable",
        "priority_conflict": "Priority\nConflict",
    }

    # ── 1. RADAR CHART ───────────────────────────────────────
    tasks  = list(TASK_LABELS.keys())
    labels = list(TASK_LABELS.values())
    N      = len(tasks)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    for model in models:
        mdf    = df_sum[df_sum["model"] == model]
        values = [
            float(mdf[mdf["task"] == t]["pass_rate"].values[0])
            if t in mdf["task"].values else 0.0
            for t in tasks
        ]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2,
                label=model, color=color_map[model])
        ax.fill(angles, values, alpha=0.15, color=color_map[model])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="white", size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], color="gray", size=8)
    ax.grid(color="gray", linestyle="--", alpha=0.4)
    ax.spines["polar"].set_color("gray")
    ax.set_title(
        "TrafficMind — Executive Function Profile\n",
        color="white", size=14, fontweight="bold", pad=20
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              facecolor="#161b22", labelcolor="white", fontsize=10)

    radar_path = ANALYSIS_DIR / "1_radar_comparison.png"
    plt.savefig(radar_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    logger.info(f"  ✅ Radar chart     → {radar_path}")

    # ── 2. COGNITIVE LOAD vs ACCURACY ────────────────────────
    if not df_det.empty and "cognitive_load" in df_det.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_facecolor("#0d1117")
        fig.patch.set_facecolor("#0d1117")

        for model in models:
            mdf   = df_det[df_det["model"] == model]
            curve = mdf.groupby("cognitive_load")["passed"].mean()
            ax.plot(
                curve.index, curve.values, "o-",
                linewidth=2.5, markersize=7,
                label=model, color=color_map[model]
            )
            ax.fill_between(
                curve.index, curve.values,
                alpha=0.08, color=color_map[model]
            )

        ax.set_xlabel("Cognitive Load (1–6)", color="white", fontsize=12)
        ax.set_ylabel("Accuracy (Pass Rate)", color="white", fontsize=12)
        ax.set_title(
            "Cognitive Load vs Model Accuracy\n"
            "Performance degradation under increasing executive demand",
            color="white", fontsize=13, fontweight="bold"
        )
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(1, 7))
        ax.tick_params(colors="gray")
        ax.grid(color="gray", linestyle="--", alpha=0.3)
        ax.legend(facecolor="#161b22", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("gray")

        load_path = ANALYSIS_DIR / "2_cognitive_load_curve.png"
        plt.savefig(load_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close()
        logger.info(f"  ✅ Load curve      → {load_path}")

    # ── 3. ACCURACY POR DIFICULTAD ───────────────────────────
    if not df_det.empty and "difficulty" in df_det.columns:
        difficulties = ["easy", "medium", "hard"]
        x = np.arange(len(difficulties))
        w = 0.8 / max(len(models), 1)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_facecolor("#0d1117")
        fig.patch.set_facecolor("#0d1117")

        for i, model in enumerate(models):
            mdf    = df_det[df_det["model"] == model]
            values = [
                float(mdf[mdf["difficulty"] == d]["passed"].mean())
                if d in mdf["difficulty"].values else 0.0
                for d in difficulties
            ]
            offset = (i - len(models) / 2 + 0.5) * w
            bars   = ax.bar(
                x + offset, values, w * 0.85,
                label=model, color=color_map[model], alpha=0.85
            )
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.0%}",
                    ha="center", va="bottom",
                    color="white", fontsize=8
                )

        ax.set_xlabel("Difficulty Level", color="white", fontsize=12)
        ax.set_ylabel("Accuracy (Pass Rate)", color="white", fontsize=12)
        ax.set_title(
            "Accuracy by Difficulty Level\n"
            "Easy → Medium → Hard performance gradient",
            color="white", fontsize=13, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(["Easy", "Medium", "Hard"], color="white")
        ax.set_ylim(0, 1.15)
        ax.tick_params(colors="gray")
        ax.grid(axis="y", color="gray", linestyle="--", alpha=0.3)
        ax.legend(facecolor="#161b22", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("gray")

        diff_path = ANALYSIS_DIR / "3_difficulty_accuracy.png"
        plt.savefig(diff_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close()
        logger.info(f"  ✅ Difficulty      → {diff_path}")

    # ── Tabla comparativa CSV ─────────────────────────────────
    pivot = df_sum.pivot_table(index="task", columns="model", values="pass_rate")
    table_path = ANALYSIS_DIR / "model_comparison_table.csv"
    pivot.to_csv(table_path)
    logger.info(f"  ✅ Tabla CSV       → {table_path}")

    # ── Reporte en consola ────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  📊 TABLA COMPARATIVA ENTRE MODELOS")
    logger.info("=" * 60)
    logger.info(f"\n{pivot.to_string()}\n")

    agg = df_sum.groupby("model").apply(
        lambda x: (x["pass_rate"] * x["weight"]).sum()
    ).rename("aggregate_score")

    logger.info("  🏆 AGGREGATE SCORES:")
    for model, score in agg.items():
        bar = "█" * int(score * 20)
        logger.info(f"     {model:<20} {bar:<20} {score:.4f}")
    logger.info("=" * 60)


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="TrafficMind Benchmark – Executive Functions for AGI Evaluation",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Ejemplos de uso:
  python benchmark.py --validate
  python benchmark.py --run --model gpt-4o-mini
  python benchmark.py --run --model gpt-4o
  python benchmark.py --run --model gpt-4.1 --jobs 2
  python benchmark.py --compare
  python benchmark.py --list-models
        """,
    )

    parser.add_argument(
        "--validate", action="store_true",
        help="Validar benchmark localmente sin API (usa optimal_answer de los JSON)"
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Ejecutar benchmark completo con modelo de IA"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        choices=SUPPORTED_MODELS,
        help="Modelo a usar (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Jobs paralelos para evaluación (default: 1)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Generar gráficos comparativos desde los CSVs guardados en results/"
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="Listar todos los modelos soportados"
    )

    args = parser.parse_args()

    # ── Listar modelos ────────────────────────────────────────
    if args.list_models:
        print("\n📋 Modelos disponibles en TrafficMind:")
        for m in SUPPORTED_MODELS:
            print(f"   • {m}")
        print()
        sys.exit(0)

    # ── Validación local ──────────────────────────────────────
    if args.validate:
        ok = validate_local()
        sys.exit(0 if ok else 1)

    # ── Ejecución con modelo ──────────────────────────────────
    if args.run:
        results = run_all_tasks(
            model_name=args.model,
            n_jobs=args.jobs,
        )
        logger.info(f"\n🏆 Aggregate Score: {results['aggregate']:.4f}")
        sys.exit(0)

    # ── Comparación y gráficos ────────────────────────────────
    if args.compare:
        logger.info("\n📊 Generando análisis comparativo...")
        generate_comparison_charts()
        sys.exit(0)

    # Default: mostrar ayuda
    parser.print_help()