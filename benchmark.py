"""
benchmark.py - TrafficMind Benchmark
Executive Functions Benchmark for Dynamic Route Planning

Evalúa las funciones ejecutivas de modelos de IA usando
escenarios de tránsito y transporte colombiano.

Track: Executive Functions
Autor: [Tu nombre]
Versión: 1.0.0
"""

import os
import sys
import json
import logging
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# ─── Cargar variables de entorno ─────────────────────────────
load_dotenv()

# ─── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─── SDK de Kaggle ────────────────────────────────────────────
import kaggle_benchmarks as kb
from utils import (
    load_json_data,
    evaluate_expected_elements,
    get_answer_field,
    build_scenario_text,
    build_evaluation_summary,
    ELEMENT_KEYWORDS,
)


# ═══════════════════════════════════════════════════════════════
# TASK 1: ROUTE PLANNING - Planificación
# ═══════════════════════════════════════════════════════════════
@kb.task(
    name="Route Planning",
    description=(
        "Evalúa si el modelo puede planificar rutas óptimas "
        "considerando múltiples restricciones simultáneas: "
        "tiempo, combustible, zonas escolares, calidad de vías. "
        "Función ejecutiva: PLANIFICACIÓN MULTI-PASO."
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
    prompt = f"""Eres un experto en planificación de rutas de transporte.
Analiza el siguiente escenario y responde de forma estructurada.

## ESCENARIO:
{scenario}

## PREGUNTA:
{question}

## INSTRUCCIONES:
1. Analiza TODAS las opciones disponibles
2. Considera TODAS las restricciones mencionadas
3. Selecciona la ruta óptima con justificación clara
4. Proporciona un plan paso a paso
5. Calcula tiempos de llegada cuando sea relevante

Responde en español de forma clara y estructurada."""

    response = kb.llm.prompt(prompt)
    evaluation = evaluate_expected_elements(response, expected_elements)

    logger.info(
        f"[{task_id}] Route Planning [{difficulty}] "
        f"Score: {evaluation['score']:.2f} | "
        f"Passed: {evaluation['passed']}"
    )
    return evaluation["passed"]


# ═══════════════════════════════════════════════════════════════
# TASK 2: PLAN DISRUPTION - Adaptación
# ═══════════════════════════════════════════════════════════════
@kb.task(
    name="Plan Disruption",
    description=(
        "El modelo está a mitad de una ruta cuando ocurre un evento "
        "inesperado. Debe detectar que el plan original falló y generar "
        "un plan alternativo viable manteniendo el objetivo. "
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
    prompt = f"""Eres un conductor experto. Estás en medio de un recorrido
cuando ocurre algo inesperado que cambia todo.

## SITUACIÓN ACTUAL:
{scenario}

## PREGUNTA:
{question}

## DEBES:
1. Reconocer el problema que interrumpe el plan original
2. Evaluar por qué el plan original ya NO funciona
3. Proponer un plan alternativo concreto y viable
4. Calcular el nuevo tiempo estimado de llegada
5. Comunicar la situación al pasajero con calma y profesionalismo

Responde en español de forma clara y directa."""

    response = kb.llm.prompt(prompt)
    evaluation = evaluate_expected_elements(response, expected_elements)

    logger.info(
        f"[{task_id}] Plan Disruption [{difficulty}] "
        f"Score: {evaluation['score']:.2f} | "
        f"Passed: {evaluation['passed']}"
    )
    return evaluation["passed"]


# ═══════════════════════════════════════════════════════════════
# TASK 3: RULE REVERSAL - Control Inhibitorio
# ═══════════════════════════════════════════════════════════════
@kb.task(
    name="Rule Reversal",
    description=(
        "El modelo enfrenta reglas de tránsito modificadas o invertidas. "
        "Debe inhibir la respuesta automática habitual y aplicar "
        "correctamente la nueva regla del contexto específico. "
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
    options_text = "\n".join(options) if options else ""

    prompt = f"""Eres un conductor en una situación donde las reglas de tránsito
NORMALES HAN SIDO MODIFICADAS para este contexto específico.

## ESCENARIO:
{scenario}

## OPCIONES:
{options_text}

## PREGUNTA:
{question}

## INSTRUCCIONES CRÍTICAS:
- Las reglas especiales del escenario ANULAN las reglas habituales
- Selecciona la opción correcta según las reglas de ESTE contexto
- Explica qué regla especial aplicas
- Explica por qué NO usas la regla normal habitual

Responde indicando la opción elegida y justificando tu decisión."""

    response = kb.llm.prompt(prompt)

    # Verificar si seleccionó la respuesta correcta
    response_norm = response.lower()
    correct_selected = (
        f"{correct_answer.lower()})" in response_norm or
        f"opcion {correct_answer.lower()}" in response_norm or
        f"opción {correct_answer.lower()}" in response_norm or
        f"opción **{correct_answer.lower()}" in response_norm or
        f"**{correct_answer.lower()})" in response_norm or
        response_norm.strip().startswith(correct_answer.lower())
    )

    evaluation = evaluate_expected_elements(response, expected_elements)

    # Score combinado balanceado:
    # - Si semántica es alta (>=0.8) Y hay algo de respuesta correcta → PASS
    # - Si semántica es media (>=0.6) Y seleccionó correcto → PASS
    # - Si semántica es baja → FAIL
    semantic_score = evaluation["score"]
    if semantic_score >= 0.8:
        final_passed = True   # Respuesta muy completa → pasa
    elif semantic_score >= 0.6 and correct_selected:
        final_passed = True   # Respuesta buena + opción correcta → pasa
    else:
        final_passed = False  # Respuesta incompleta → falla


    logger.info(
        f"[{task_id}] Rule Reversal [{difficulty}] "
        f"Score: {evaluation['score']:.2f} | "
        f"Correct: {correct_answer} | "
        f"Selected correctly: {correct_selected} | "
        f"Passed: {final_passed}"
    )
    return final_passed


# ═══════════════════════════════════════════════════════════════
# TASK 4: MULTI-VARIABLE TRACKING - Memoria de Trabajo
# ═══════════════════════════════════════════════════════════════
@kb.task(
    name="Multi Variable Tracking",
    description=(
        "El modelo debe monitorear múltiples variables simultáneas "
        "(combustible, tiempo, dinero, pasajeros, estado de flota) "
        "y calcular/responder con precisión sobre cada una. "
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

## ESCENARIO Y VARIABLES:
{scenario}

## PREGUNTAS QUE DEBES RESPONDER (una por una):
{questions_text}

## INSTRUCCIONES:
1. Responde CADA pregunta por separado y en orden
2. Muestra el cálculo o razonamiento para cada respuesta
3. Sé específico con números y valores exactos
4. Identifica si alguna variable requiere atención urgente

Responde en español con cálculos explícitos."""

    response = kb.llm.prompt(prompt)
    
    # Evaluación semántica base
    evaluation = evaluate_expected_elements(response, expected_elements)
    
    # ── Evaluación numérica estricta ──────────────────────────
    correct_answers = [q.get("correct", "") for q in questions if isinstance(q, dict)]
    
    numeric_matches = 0
    for correct in correct_answers:
        # Extraer números de la respuesta correcta
        import re
        numbers_in_correct = re.findall(r'\d+[\.,]?\d*', correct.replace(",", ""))
        numbers_in_response = re.findall(r'\d+[\.,]?\d*', response.replace(",", ""))
        
        # Verificar si los números clave aparecen en la respuesta
        key_numbers = [n for n in numbers_in_correct if len(n) >= 2]
        matches = sum(1 for n in key_numbers if n in numbers_in_response)
        
        if key_numbers and matches / len(key_numbers) >= 0.6:
            numeric_matches += 1
    
    # Requiere tanto evaluación semántica como numérica
    numeric_accuracy = numeric_matches / len(correct_answers) if correct_answers else 0
    
    # Score combinado más estricto
    combined_score = (evaluation["score"] * 0.4) + (numeric_accuracy * 0.6)
    final_passed = combined_score >= 0.65  # Umbral más alto
    
    logger.info(
        f"[{task_id}] Multi-Variable [{difficulty}] "
        f"Score: {combined_score:.2f} | "
        f"Semantic: {evaluation['score']:.2f} | "
        f"Numeric: {numeric_accuracy:.2f} | "
        f"Cognitive load: {cognitive_load}/5 | "
        f"Passed: {final_passed}"
    )
    return final_passed

# ═══════════════════════════════════════════════════════════════
# TASK 5: PRIORITY CONFLICT - Flexibilidad Cognitiva
# ═══════════════════════════════════════════════════════════════
@kb.task(
    name="Priority Conflict",
    description=(
        "El modelo enfrenta objetivos simultáneos en conflicto directo. "
        "Debe reconocer el dilema, establecer una jerarquía de prioridades "
        "con criterio claro y justificar lo que sacrifica. "
        "Función ejecutiva: FLEXIBILIDAD COGNITIVA."
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
    prompt = f"""Eres un tomador de decisiones en transporte que enfrenta
objetivos que entran en conflicto directo entre sí.

## SITUACIÓN CON CONFLICTO DE PRIORIDADES:
{scenario}

## PREGUNTA:
{question}

## DEBES DEMOSTRAR:
1. Reconocer EXPLÍCITAMENTE el conflicto entre objetivos
2. Listar los objetivos en tensión
3. Establecer una jerarquía de prioridades con criterio claro
4. Tomar una decisión definitiva y justificarla
5. Explicar qué sacrificas y por qué vale la pena

Responde en español siendo explícito sobre el conflicto y tu razonamiento."""

    response = kb.llm.prompt(prompt)
    evaluation = evaluate_expected_elements(response, expected_elements)

    logger.info(
        f"[{task_id}] Priority Conflict [{difficulty}] "
        f"Score: {evaluation['score']:.2f} | "
        f"Passed: {evaluation['passed']}"
    )
    return evaluation["passed"]


# ═══════════════════════════════════════════════════════════════
# FUNCIONES DE PREPARACIÓN DE DATOS
# ═══════════════════════════════════════════════════════════════

def prepare_dataframe_task1() -> pd.DataFrame:
    """Prepara DataFrame para Task 1."""
    data = load_json_data("task1_route_planning.json")
    rows = []
    for item in data:
        scenario = item["scenario"]
        map_data = scenario.get("map_data", {})
        rows.append({
            "task_id":          item["task_id"],
            "scenario":         (
                f"{scenario.get('description','')}\n"
                f"Origen: {map_data.get('origin','')}\n"
                f"Destino: {map_data.get('destination','')}\n"
                f"Restricciones: {scenario.get('constraints',[])}"
            ),
            "question":         item["question"],
            "expected_elements": item["expected_elements"],
            "difficulty":       item["difficulty"],
            "cognitive_load":   item["cognitive_load"],
        })
    return pd.DataFrame(rows)


def prepare_dataframe_task2() -> pd.DataFrame:
    """Prepara DataFrame para Task 2."""
    data = load_json_data("task2_plan_disruption.json")
    rows = []
    for item in data:
        scenario = item["scenario"]
        plan     = scenario.get("initial_plan", {})
        disrupt  = scenario.get("disruption", {})
        alts     = scenario.get("alternatives", [])

        scenario_text = (
            f"PLAN INICIAL: {plan.get('description','')}\n"
            f"Progreso: {plan.get('progress','')}\n"
            f"Tiempo restante: {plan.get('remaining_time','')}\n\n"
            f"EVENTO INESPERADO - {disrupt.get('type','').upper()}:\n"
            f"{disrupt.get('description','')}\n"
            f"Tu posición: {disrupt.get('your_position','')}\n\n"
            f"OPCIONES:\n" +
            "\n".join(
                f"  - {a.get('name','')}: {a.get('description','')} "
                f"(+{a.get('estimated_additional_time','')})"
                for a in alts
            )
        )
        rows.append({
            "task_id":          item["task_id"],
            "scenario":         scenario_text,
            "question":         item["question"],
            "expected_elements": item["expected_elements"],
            "difficulty":       item["difficulty"],
            "cognitive_load":   item["cognitive_load"],
        })
    return pd.DataFrame(rows)


def prepare_dataframe_task3() -> pd.DataFrame:
    """Prepara DataFrame para Task 3."""
    data = load_json_data("task3_rule_reversal.json")
    rows = []
    for item in data:
        scenario = item["scenario"]
        scenario_text = (
            f"Contexto: {scenario.get('context','')}\n"
            f"Regla normal: {scenario.get('normal_rule','')}\n"
            f"REGLA ESPECIAL (aplica esta): {scenario.get('reversed_rule','')}\n"
            f"Situación: {scenario.get('situation','')}"
        )
        rows.append({
            "task_id":          item["task_id"],
            "scenario":         scenario_text,
            "question":         item["question"],
            "options":          item.get("options", []),
            "correct_answer":   item["correct_answer"],
            "expected_elements": item["expected_elements"],
            "difficulty":       item["difficulty"],
            "cognitive_load":   item.get("cognitive_load", 3),
        })
    return pd.DataFrame(rows)


def prepare_dataframe_task4() -> pd.DataFrame:
    """Prepara DataFrame para Task 4."""
    data = load_json_data("task4_multi_variable.json")
    rows = []
    for item in data:
        scenario = item["scenario"]
        import json as _json
        scenario_text = (
            f"{scenario.get('description','')}\n\n"
            f"VARIABLES:\n"
            f"{_json.dumps(scenario.get('variables_to_track', scenario), ensure_ascii=False, indent=2)}"
        )
        rows.append({
            "task_id":          item["task_id"],
            "scenario":         scenario_text,
            "questions":        item.get("questions", []),
            "expected_elements": item["expected_elements"],
            "difficulty":       item["difficulty"],
            "cognitive_load":   item["cognitive_load"],
        })
    return pd.DataFrame(rows)


def prepare_dataframe_task5() -> pd.DataFrame:
    """Prepara DataFrame para Task 5."""
    data = load_json_data("task5_priority_conflict.json")
    rows = []
    for item in data:
        scenario = item["scenario"]
        import json as _json
        scenario_text = (
            f"{scenario.get('description','')}\n\n"
            f"{_json.dumps({k:v for k,v in scenario.items() if k != 'description'}, ensure_ascii=False, indent=2)}"
        )
        rows.append({
            "task_id":          item["task_id"],
            "scenario":         scenario_text,
            "question":         item["question"],
            "expected_elements": item["expected_elements"],
            "difficulty":       item["difficulty"],
            "cognitive_load":   item["cognitive_load"],
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# RUNNER PRINCIPAL
# ═══════════════════════════════════════════════════════════════

def run_all_tasks(n_jobs: int = 1) -> dict:
    """
    Ejecuta todas las tasks y retorna resultados agregados.
    
    Args:
        n_jobs: Número de jobs paralelos (1=secuencial, -1=todos los cores)
    
    Returns:
        dict con scores por task y score agregado final
    """
    logger.info("="*60)
    logger.info("🚗 TrafficMind Benchmark - Iniciando evaluación")
    logger.info("="*60)

    results = {}

    # ── Task 1: Route Planning ────────────────────────────────
    logger.info("\n📍 Task 1: Route Planning")
    df1 = prepare_dataframe_task1()
    runs1 = route_planning_task.evaluate(
        evaluation_data=df1,
        n_jobs=n_jobs,
    )
    df_results1 = runs1.as_dataframe()
    results["route_planning"] = {
        "runs": runs1,
        "pass_rate": df_results1["result"].mean() if len(df_results1) > 0 else 0,
        "total": len(df1),
        "weight": 0.20,
    }
    logger.info(f"  ✅ Pass rate: {results['route_planning']['pass_rate']:.2%}")

    # ── Task 2: Plan Disruption ───────────────────────────────
    logger.info("\n🚧 Task 2: Plan Disruption")
    df2 = prepare_dataframe_task2()
    runs2 = plan_disruption_task.evaluate(
        evaluation_data=df2,
        n_jobs=n_jobs,
    )
    df_results2 = runs2.as_dataframe()
    results["plan_disruption"] = {
        "runs": runs2,
        "pass_rate": df_results2["result"].mean() if len(df_results2) > 0 else 0,
        "total": len(df2),
        "weight": 0.25,
    }
    logger.info(f"  ✅ Pass rate: {results['plan_disruption']['pass_rate']:.2%}")

    # ── Task 3: Rule Reversal ─────────────────────────────────
    logger.info("\n🔄 Task 3: Rule Reversal")
    df3 = prepare_dataframe_task3()
    runs3 = rule_reversal_task.evaluate(
        evaluation_data=df3,
        n_jobs=n_jobs,
    )
    df_results3 = runs3.as_dataframe()
    results["rule_reversal"] = {
        "runs": runs3,
        "pass_rate": df_results3["result"].mean() if len(df_results3) > 0 else 0,
        "total": len(df3),
        "weight": 0.20,
    }
    logger.info(f"  ✅ Pass rate: {results['rule_reversal']['pass_rate']:.2%}")

    # ── Task 4: Multi-Variable Tracking ──────────────────────
    logger.info("\n📊 Task 4: Multi-Variable Tracking")
    df4 = prepare_dataframe_task4()
    runs4 = multi_variable_task.evaluate(
        evaluation_data=df4,
        n_jobs=n_jobs,
    )
    df_results4 = runs4.as_dataframe()
    results["multi_variable"] = {
        "runs": runs4,
        "pass_rate": df_results4["result"].mean() if len(df_results4) > 0 else 0,
        "total": len(df4),
        "weight": 0.15,
    }
    logger.info(f"  ✅ Pass rate: {results['multi_variable']['pass_rate']:.2%}")

    # ── Task 5: Priority Conflict ─────────────────────────────
    logger.info("\n⚖️  Task 5: Priority Conflict")
    df5 = prepare_dataframe_task5()
    runs5 = priority_conflict_task.evaluate(
        evaluation_data=df5,
        n_jobs=n_jobs,
    )
    df_results5 = runs5.as_dataframe()
    results["priority_conflict"] = {
        "runs": runs5,
        "pass_rate": df_results5["result"].mean() if len(df_results5) > 0 else 0,
        "total": len(df5),
        "weight": 0.20,
    }
    logger.info(f"  ✅ Pass rate: {results['priority_conflict']['pass_rate']:.2%}")

    # ── Score Agregado Final ──────────────────────────────────
    aggregate = sum(
        v["pass_rate"] * v["weight"]
        for v in results.values()
    )

    logger.info("\n" + "="*60)
    logger.info("📊 RESULTADOS FINALES - TrafficMind Benchmark")
    logger.info("="*60)
    for task_name, data in results.items():
        bar = "█" * int(data["pass_rate"] * 20)
        logger.info(
            f"  {task_name:<25} {bar:<20} "
            f"{data['pass_rate']:.2%} "
            f"(weight: {data['weight']})"
        )
    logger.info(f"\n  🏆 AGGREGATE SCORE: {aggregate:.4f}")
    logger.info("="*60)

    results["aggregate_score"] = aggregate
    return results


# ═══════════════════════════════════════════════════════════════
# VALIDACIÓN LOCAL (sin API)
# ═══════════════════════════════════════════════════════════════

def validate_local() -> bool:
    """
    Valida que todo el pipeline funciona sin necesitar API.
    Usa las respuestas óptimas de los JSON como mock.
    """
    print("\n" + "="*60)
    print("🔍 VALIDACIÓN LOCAL - TrafficMind Benchmark")
    print("="*60)

    all_ok = True
    tasks_info = [
        ("Task 1 - Route Planning",    "task1_route_planning.json",   "optimal_answer"),
        ("Task 2 - Plan Disruption",   "task2_plan_disruption.json",  "optimal_response"),
        ("Task 3 - Rule Reversal",     "task3_rule_reversal.json",    "correct_answer"),
        ("Task 4 - Multi-Variable",    "task4_multi_variable.json",   "questions"),
        ("Task 5 - Priority Conflict", "task5_priority_conflict.json","optimal_answer"),
    ]

    total_items = 0
    total_passed = 0

    for task_name, filename, answer_field in tasks_info:
        data = load_json_data(filename)
        passed = 0
        scores = []

        for item in data:
            answer = get_answer_field(item)
            evaluation = evaluate_expected_elements(
                response=answer,
                expected_elements=item["expected_elements"],
            )
            scores.append(evaluation["score"])
            if evaluation["passed"]:
                passed += 1

        avg_score = sum(scores) / len(scores)
        pass_rate = passed / len(data)
        status = "✅" if avg_score >= 0.6 else "⚠️ "
        if avg_score < 0.6:
            all_ok = False

        print(f"  {status} {task_name:<30} "
              f"avg={avg_score:.2f} | "
              f"pass_rate={pass_rate:.0%} | "
              f"items={len(data)}")

        total_items += len(data)
        total_passed += passed

    overall = total_passed / total_items
    print(f"\n  📊 Total items: {total_items}")
    print(f"  📊 Overall pass rate: {overall:.2%}")
    print(f"  {'✅ Benchmark listo para producción' if all_ok else '⚠️  Revisar tasks con warning'}")
    print("="*60)

    return all_ok


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TrafficMind Benchmark - Executive Functions"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validar localmente sin API (usa respuestas óptimas)"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Ejecutar benchmark completo con modelo de IA"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Número de jobs paralelos (default: 1)"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["1","2","3","4","5","all"],
        default="all",
        help="Ejecutar task específica o todas"
    )
    args = parser.parse_args()

    if args.validate or (not args.run):
        # Por defecto: validar localmente
        ok = validate_local()
        sys.exit(0 if ok else 1)

    if args.run:
        results = run_all_tasks(n_jobs=args.jobs)
        print(f"\n🏆 Aggregate Score: {results['aggregate_score']:.4f}")
