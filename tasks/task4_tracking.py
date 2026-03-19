"""
Task 4: Multi-Variable Tracking - Memoria de Trabajo
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

import kaggle_benchmarks as kb
from utils import (load_json_data, evaluate_expected_elements,
                   get_answer_field)

_data = load_json_data("task4_multi_variable.json")

# Keywords específicas para task4
TASK4_KEYWORDS = {
    "calcula_combustible_restante": [
        "26", "litros", "combustible", "queda",
        "restante", "consumio", "24 litros"
    ],
    "suma_recaudacion": [
        "75", "75.000", "75,000", "total",
        "recaudado", "suma"
    ],
    "calcula_tiempo_restante": [
        "7 horas", "siete", "faltan", "restante",
        "meta", "completar", "horas"
    ],
    "trackea_todas_variables": [
        "combustible", "dinero", "tiempo", "variables",
        "todos", "factores"
    ],
    "estado_cada_variable": [
        "nivel", "estado", "actual", "disponible",
        "restante", "queda"
    ],
    "decision_integrada": [
        "considerando", "integrando", "todo",
        "combinando", "conjunto"
    ],
    "identifica_variable_critica": [
        "critico", "prioritario", "importante",
        "urgente", "limitante", "primero"
    ],
    "plan_multicriterio": [
        "plan", "accion", "estrategia",
        "considerando", "balance"
    ],
    "monitorea_combustible": [
        "combustible", "litros", "tanque",
        "gasolina", "lleno", "consumo"
    ],
    "monitorea_tiempo": [
        "horas", "tiempo", "turno", "meta",
        "am", "pm", "trabajado"
    ],
    "monitorea_dinero": [
        "dinero", "recaudado", "pesos",
        "viaje", "total", "ganado"
    ],
    "responde_preguntas_especificas": [
        "26", "75", "7 horas", "respuesta",
        "son", "quedan", "tengo"
    ],
}


@kb.task(
    name="Multi Variable Tracking",
    description=(
        "El modelo debe monitorear múltiples variables simultáneas "
        "(combustible, dinero, tiempo, pasajeros) y calcular/responder "
        "preguntas sobre cada una con precisión. "
        "Función ejecutiva: MEMORIA DE TRABAJO."
    ),
    version=1,
)
def multi_variable_task(
    scenario: str,
    questions: list,
    expected_elements: list,
    difficulty: str,
    cognitive_load: int,
) -> bool:
    questions_text = "\n".join(
        f"{i+1}. {q.get('q', q)}"
        for i, q in enumerate(questions)
    ) if questions else ""

    prompt = f"""Eres un coordinador de transporte monitoreando múltiples
variables en tiempo real.

## ESCENARIO Y VARIABLES A MONITOREAR:
{scenario}

## PREGUNTAS QUE DEBES RESPONDER:
{questions_text}

## LO QUE DEBES DEMOSTRAR:
1. Calcular el estado ACTUAL de CADA variable
2. Mostrar el proceso de cálculo paso a paso
3. Responder cada pregunta con precisión numérica
4. Identificar cuál variable es más crítica

Responde en español con cálculos explícitos."""

    response = kb.llm.chat(prompt)
    evaluation = evaluate_expected_elements(
        response, expected_elements, TASK4_KEYWORDS
    )

    print(f"  [{difficulty.upper()}] Score: {evaluation['score']:.2f} | "
          f"Carga cognitiva: {cognitive_load}/5 | "
          f"Cubiertos: {evaluation['covered_count']}/{evaluation['total_elements']}")
    return evaluation["passed"]


def _build_scenario_task4(item: dict) -> str:
    """Construye texto del escenario para task4."""
    scenario = item.get("scenario", {})
    parts = []

    if "description" in scenario:
        parts.append(scenario["description"])

    variables = scenario.get("variables_to_track", {})
    if variables:
        parts.append("\nVARIABLES A MONITOREAR:")
        parts.append(json.dumps(variables, ensure_ascii=False, indent=2))

    return "\n".join(parts)


if __name__ == "__main__":
    print("=== TEST Task 4: Multi-Variable Tracking (sin API) ===\n")
    data = load_json_data("task4_multi_variable.json")

    scores = []
    for item in data:
        # Construir respuesta combinando los "correct" de cada pregunta
        answer = get_answer_field(item)
        evaluation = evaluate_expected_elements(
            response=answer,
            expected_elements=item["expected_elements"],
            element_keywords=TASK4_KEYWORDS,
        )
        scores.append(evaluation["score"])
        status = "✅" if evaluation["passed"] else "❌"

        # Mostrar las respuestas correctas esperadas
        correctas = [q.get("correct", "") for q in item.get("questions", [])]
        print(f"{status} {item['task_id']} [{item['difficulty']}] "
              f"Score: {evaluation['score']:.2f} | "
              f"Carga: {item['cognitive_load']}/5")
        print(f"   Respuestas esperadas: {correctas}")
        if evaluation["missing"]:
            print(f"   Faltantes: {evaluation['missing']}")

    avg = sum(scores) / len(scores)
    print(f"\n📊 Score promedio: {avg:.2f}")
    print("✅ Task 4 lista" if avg >= 0.6 else "⚠️ Revisar evaluador")
