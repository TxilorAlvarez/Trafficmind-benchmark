"""
Task 1: Route Planning - Planificación de Rutas
Evalúa la capacidad de planificación multi-paso con restricciones.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from dotenv import load_dotenv
load_dotenv()

import kaggle_benchmarks as kb
from utils import load_json_data, evaluate_expected_elements

# ─── Cargar datos ─────────────────────────────────────────────
_data = load_json_data("task1_route_planning.json")
_df   = pd.DataFrame(_data)

# ─── Definir la Task con decorador real del SDK ───────────────
@kb.task(
    name="Route Planning",
    description=(
        "Evalúa si el modelo puede planificar rutas óptimas "
        "considerando múltiples restricciones: tiempo, combustible, "
        "zonas escolares, calidad de vías y carga. "
        "Función ejecutiva: PLANIFICACIÓN MULTI-PASO."
    ),
    version=1,
)
def route_planning_task(
    scenario: str,
    question: str,
    expected_elements: list,
    optimal_answer: str,
    difficulty: str,
    cognitive_load: int,
) -> bool:
    """
    Recibe un escenario de tránsito y evalúa si el modelo
    genera un plan de ruta válido que cumpla las restricciones.
    """
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

    # Llamar al modelo via SDK
    response = kb.llm.chat(prompt)

    # Evaluar respuesta
    evaluation = evaluate_expected_elements(
        response=response,
        expected_elements=expected_elements,
    )

    # Registrar resultado para análisis
    print(f"  [{difficulty.upper()}] Score: {evaluation['score']:.2f} | "
          f"Cubiertos: {evaluation['covered_count']}/{evaluation['total_elements']}")

    return evaluation["passed"]


# ─── Test local sin SDK ───────────────────────────────────────
if __name__ == "__main__":
    from utils import load_json_data, evaluate_expected_elements

    print("=== TEST Task 1: Route Planning (sin API) ===\n")
    data = load_json_data("task1_route_planning.json")

    scores = []
    for item in data:
        scenario_text = (
            f"{item['scenario']['description']}\n"
            f"Origen: {item['scenario']['map_data']['origin']}\n"
            f"Destino: {item['scenario']['map_data']['destination']}\n"
            f"Restricciones: {item['scenario']['constraints']}"
        )

        # Simulamos la respuesta óptima para testear el evaluador
        evaluation = evaluate_expected_elements(
            response=item["optimal_answer"],
            expected_elements=item["expected_elements"],
        )
        scores.append(evaluation["score"])
        status = "✅" if evaluation["passed"] else "❌"
        print(f"{status} {item['task_id']} [{item['difficulty']}] "
              f"Score: {evaluation['score']:.2f} | "
              f"Cubiertos: {evaluation['covered_count']}/{evaluation['total_elements']}")

    avg = sum(scores) / len(scores)
    print(f"\n📊 Score promedio con respuesta óptima: {avg:.2f}")
    print("✅ Task 1 lista para producción" if avg >= 0.6 else "⚠️ Revisar evaluador")
