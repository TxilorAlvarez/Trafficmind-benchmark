"""
Task 5: Priority Conflict - Flexibilidad Cognitiva
Evalúa si el modelo reconoce conflictos entre objetivos,
prioriza explícitamente y puede cambiar de criterio.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from dotenv import load_dotenv
load_dotenv()

import kaggle_benchmarks as kb
from utils import load_json_data, evaluate_expected_elements

_data = load_json_data("task5_priority_conflict.json")
_df   = pd.DataFrame(_data)

@kb.task(
    name="Priority Conflict",
    description=(
        "Evalúa si el modelo reconoce conflictos entre objetivos "
        "simultáneos, prioriza explícitamente con justificación, y puede "
        "cambiar de criterio ante nueva información. "
        "Función ejecutiva: FLEXIBILIDAD COGNITIVA."
    ),
    version=1,
)
def priority_conflict_task(
    scenario: str,
    question: str,
    expected_elements: list,
    optimal_answer: str,
    difficulty: str,
    cognitive_load: int,
) -> bool:
    """
    Presenta objetivos en conflicto real.
    El modelo debe reconocer el dilema, priorizar y justificar.
    """
    prompt = f"""Eres un tomador de decisiones en transporte que enfrenta
objetivos que entran en conflicto directo entre sí.

## SITUACIÓN CON CONFLICTO DE PRIORIDADES:
{scenario}

## PREGUNTA:
{question}

## LO QUE DEBES DEMOSTRAR:
1. Reconocer EXPLÍCITAMENTE el conflicto entre objetivos
2. Listar los objetivos en tensión
3. Establecer una jerarquía de prioridades con criterio claro
4. Tomar una decisión definitiva y justificarla
5. Explicar qué sacrificas y por qué vale la pena

Responde en español siendo explícito sobre el conflicto."""

    response = kb.llm.chat(prompt)

    evaluation = evaluate_expected_elements(
        response=response,
        expected_elements=expected_elements,
    )

    print(f"  [{difficulty.upper()}] Score: {evaluation['score']:.2f} | "
          f"Cubiertos: {evaluation['covered_count']}/{evaluation['total_elements']}")

    return evaluation["passed"]


if __name__ == "__main__":
    from utils import load_json_data, evaluate_expected_elements

    print("=== TEST Task 5: Priority Conflict (sin API) ===\n")
    data = load_json_data("task5_priority_conflict.json")

    scores = []
    for item in data:
        evaluation = evaluate_expected_elements(
            response=item["optimal_answer"],
            expected_elements=item["expected_elements"],
        )
        scores.append(evaluation["score"])
        status = "✅" if evaluation["passed"] else "❌"
        print(f"{status} {item['task_id']} [{item['difficulty']}] "
              f"Score: {evaluation['score']:.2f}")

    avg = sum(scores) / len(scores)
    print(f"\n📊 Score promedio: {avg:.2f}")
    print("✅ Task 5 lista" if avg >= 0.6 else "⚠️ Revisar evaluador")
