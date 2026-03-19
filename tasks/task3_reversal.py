"""
Task 3: Rule Reversal - Inhibición de Respuestas Habituales
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from dotenv import load_dotenv
load_dotenv()

import kaggle_benchmarks as kb
from utils import (load_json_data, evaluate_expected_elements,
                   get_answer_field, ELEMENT_KEYWORDS)

_data = load_json_data("task3_rule_reversal.json")

# Keywords específicas para task3
TASK3_KEYWORDS = {
    **ELEMENT_KEYWORDS,
    "reconoce_regla_invertida": [
        "invertida", "opuesta", "diferente", "especial",
        "este pais", "contexto", "regla nueva", "modificada"
    ],
    "aplica_regla_nueva_no_habitual": [
        "cedo", "detengo", "ceder", "detener", "parar",
        "opcion b", "b)", "respuesta b", "correcto es b"
    ],
    "justifica_segun_contexto": [
        "porque", "en este pais", "segun", "indica",
        "regla establece", "ya que", "contexto"
    ],
    "evalua_opciones": [
        "opcion", "alternativa", "puede", "podria",
        "a)", "b)", "c)", "d)"
    ],
    "selecciona_mejor_alternativa": [
        "mejor", "correcta", "adecuada", "recomiendo",
        "selecciono", "es la"
    ],
    "calcula_nuevo_tiempo": [
        "tiempo", "minutos", "llegada", "estimado",
        "aproximadamente", "tardara"
    ],
    "comunica_al_pasajero": [
        "pasajero", "señor", "senora", "informo",
        "tranquilo", "no se preocupe", "comunicar"
    ],
    "mantiene_calma": [
        "calma", "tranquilo", "no preocupe", "controlado",
        "seguro", "calmado"
    ],
    # task4 específicos
    "calcula_combustible_restante": [
        "26", "litros", "combustible", "queda",
        "restante", "24 litros", "consumio"
    ],
    "suma_recaudacion": [
        "75", "75,000", "75000", "total", "recaudado",
        "suma", "dinero"
    ],
    "calcula_tiempo_restante": [
        "7 horas", "siete horas", "faltan", "restante",
        "meta", "completar"
    ],
}


@kb.task(
    name="Rule Reversal",
    description=(
        "El modelo enfrenta reglas de tránsito modificadas o invertidas. "
        "Debe inhibir la respuesta automática (regla habitual) y aplicar "
        "correctamente la nueva regla del contexto. "
        "Función ejecutiva: CONTROL INHIBITORIO."
    ),
    version=1,
)
def rule_reversal_task(
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
- Explica qué regla especial aplicas y por qué NO usas la regla normal

Responde en español indicando la opción y justificando."""

    response = kb.llm.chat(prompt)
    evaluation = evaluate_expected_elements(
        response, expected_elements, TASK3_KEYWORDS
    )

    print(f"  [{difficulty.upper()}] Score: {evaluation['score']:.2f} | "
          f"Correcto: {correct_answer} | "
          f"Cubiertos: {evaluation['covered_count']}/{evaluation['total_elements']}")
    return evaluation["passed"]


def _build_scenario_task3(item: dict) -> str:
    """Construye texto del escenario para task3."""
    scenario = item.get("scenario", {})
    parts = []
    if "context" in scenario:
        parts.append(f"Contexto: {scenario['context']}")
    if "normal_rule" in scenario:
        parts.append(f"Regla normal: {scenario['normal_rule']}")
    if "reversed_rule" in scenario:
        parts.append(f"REGLA ESPECIAL (aplicar esta): {scenario['reversed_rule']}")
    if "situation" in scenario:
        parts.append(f"Situación: {scenario['situation']}")
    return "\n".join(parts)


if __name__ == "__main__":
    print("=== TEST Task 3: Rule Reversal (sin API) ===\n")
    data = load_json_data("task3_rule_reversal.json")

    scores = []
    for item in data:
        # Construir respuesta de prueba combinando correct_answer + explanation
        answer = get_answer_field(item)
        evaluation = evaluate_expected_elements(
            response=answer,
            expected_elements=item["expected_elements"],
            element_keywords=TASK3_KEYWORDS,
        )
        scores.append(evaluation["score"])
        status = "✅" if evaluation["passed"] else "❌"
        print(f"{status} {item['task_id']} [{item['difficulty']}] "
              f"Score: {evaluation['score']:.2f} | "
              f"Correcto: {item['correct_answer']} | "
              f"Trampa: {item.get('cognitive_trap','')[:40]}...")
        if evaluation["missing"]:
            print(f"   Faltantes: {evaluation['missing']}")

    avg = sum(scores) / len(scores)
    print(f"\n📊 Score promedio: {avg:.2f}")
    print("✅ Task 3 lista" if avg >= 0.6 else "⚠️ Revisar evaluador")
