"""
Task 2: Plan Disruption - Adaptación a Cambios
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from dotenv import load_dotenv
load_dotenv()

import kaggle_benchmarks as kb
from utils import (load_json_data, evaluate_expected_elements,
                   get_answer_field, build_scenario_text)

_data = load_json_data("task2_plan_disruption.json")

@kb.task(
    name="Plan Disruption",
    description=(
        "El modelo está a mitad de una ruta cuando ocurre un evento "
        "inesperado (accidente, cierre, clima). Debe detectar que el "
        "plan original falló y generar un plan alternativo viable. "
        "Función ejecutiva: ADAPTACIÓN Y FLEXIBILIDAD."
    ),
    version=1,
)
def plan_disruption_task(
    scenario: str,
    question: str,
    expected_elements: list,
    optimal_response: str,
    difficulty: str,
    cognitive_load: int,
) -> bool:
    prompt = f"""Eres un conductor experto. Estás en medio de un recorrido
cuando ocurre algo inesperado.

## SITUACIÓN:
{scenario}

## PREGUNTA:
{question}

## DEBES:
1. Reconocer el problema que interrumpe el plan
2. Evaluar por qué el plan original ya NO funciona  
3. Proponer un plan alternativo concreto
4. Calcular el nuevo tiempo de llegada
5. Comunicar la situación al pasajero con calma

Responde en español de forma clara."""

    response = kb.llm.chat(prompt)
    evaluation = evaluate_expected_elements(response, expected_elements)

    print(f"  [{difficulty.upper()}] Score: {evaluation['score']:.2f} | "
          f"Cubiertos: {evaluation['covered_count']}/{evaluation['total_elements']}")
    return evaluation["passed"]


def _build_scenario_task2(item: dict) -> str:
    """Construye texto del escenario para task2."""
    scenario = item.get("scenario", {})
    parts = []

    plan = scenario.get("initial_plan", {})
    if plan:
        parts.append("PLAN INICIAL:")
        parts.append(f"  {plan.get('description', '')}")
        parts.append(f"  Progreso: {plan.get('progress', '')}")
        parts.append(f"  Tiempo restante: {plan.get('remaining_time', '')}")
        parts.append(f"  Margen hasta el vuelo: {plan.get('buffer_time', '')}")

    disruption = scenario.get("disruption", {})
    if disruption:
        parts.append(f"\nEVENTO INESPERADO - {disruption.get('type','').upper()}:")
        parts.append(f"  {disruption.get('description', '')}")
        parts.append(f"  Tu posición actual: {disruption.get('your_position', '')}")

    alternatives = scenario.get("alternatives", [])
    if alternatives:
        parts.append("\nOPCIONES DISPONIBLES:")
        for alt in alternatives:
            parts.append(f"  - {alt.get('name','')}: {alt.get('description','')} "
                        f"(+{alt.get('estimated_additional_time','')})")

    return "\n".join(parts)


if __name__ == "__main__":
    print("=== TEST Task 2: Plan Disruption (sin API) ===\n")
    data = load_json_data("task2_plan_disruption.json")

    scores = []
    for item in data:
        answer = get_answer_field(item)
        evaluation = evaluate_expected_elements(
            response=answer,
            expected_elements=item["expected_elements"],
        )
        scores.append(evaluation["score"])
        status = "✅" if evaluation["passed"] else "❌"
        print(f"{status} {item['task_id']} [{item['difficulty']}] "
              f"Score: {evaluation['score']:.2f} | "
              f"Cubiertos: {evaluation['covered_count']}/{evaluation['total_elements']}")
        if evaluation["missing"]:
            print(f"   Faltantes: {evaluation['missing']}")

    avg = sum(scores) / len(scores)
    print(f"\n📊 Score promedio: {avg:.2f}")
    print("✅ Task 2 lista" if avg >= 0.6 else "⚠️ Revisar evaluador")
