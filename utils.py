"""
utils.py - TrafficMind Benchmark
Utilidades compartidas para todas las tasks.
"""

import json
import os
import re
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Rutas base ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"


def load_json_data(filename: str) -> list[dict]:
    """Carga un archivo JSON desde la carpeta data/."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Cargados {len(data)} items desde {filename}")
    return data


def filter_by_difficulty(
    data: list[dict],
    difficulty: str | None = None
) -> list[dict]:
    """Filtra items por dificultad."""
    if difficulty is None:
        return data
    return [item for item in data if item.get("difficulty") == difficulty]


def normalize_text(text: str) -> str:
    """Normaliza texto para comparación."""
    text = text.lower()
    replacements = {
        'á':'a','à':'a','ä':'a','â':'a',
        'é':'e','è':'e','ë':'e','ê':'e',
        'í':'i','ì':'i','ï':'i','î':'i',
        'ó':'o','ò':'o','ö':'o','ô':'o',
        'ú':'u','ù':'u','ü':'u','û':'u',
        'ñ':'n'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def check_keyword_presence(
    response: str,
    keywords: list[str],
    threshold: float = 0.4
) -> bool:
    """Verifica si suficientes keywords están en la respuesta."""
    response_norm = normalize_text(response)
    matches = sum(
        1 for kw in keywords
        if normalize_text(kw) in response_norm
    )
    return matches >= max(1, len(keywords) * threshold)


# ─── Mapa de keywords por elemento esperado ──────────────────
ELEMENT_KEYWORDS: dict[str, list[str]] = {
    # Task 1 - Route Planning
    "selecciona_ruta_especifica": [
        "ruta", "selecciono", "elijo", "recomiendo", "tomo", "mejor", "opcion"
    ],
    "considera_tiempo_disponible": [
        "minutos", "tiempo", "llegada", "antes", "horas", "margen", "plazo"
    ],
    "justifica_eleccion": [
        "porque", "debido", "ya que", "razon", "justific", "mejor",
        "aunque", "a pesar", "ventaja", "beneficio", "compensa"
    ],
    "plan_paso_a_paso": [
        "paso", "primero", "luego", "despues", "finalmente", "primero",
        "segundo", "tercero", "entonces", "a continuacion"
    ],
    "analiza_todas_las_rutas": [
        "ruta a", "ruta b", "ruta c", "analisis", "opciones",
        "alternativas", "comparando", "todas"
    ],
    "considera_hora_pico": [
        "hora pico", "congestion", "trafico", "rush", "atasco",
        "viernes", "tarde", "matutino"
    ],
    "considera_condiciones_clima": [
        "lluvia", "clima", "condiciones", "tiempo", "mojado",
        "neblina", "visibilidad", "meteorolog"
    ],
    "considera_obras_via": [
        "obras", "construccion", "bloqueo", "desvio", "km 15",
        "trabajos", "cierre"
    ],
    "calcula_tiempo_llegada": [
        "llegaria", "llegar", "estimado", "aproximadamente",
        "am", "pm", "minutos", "horas"
    ],
    "selecciona_ruta_viable": [
        "viable", "posible", "funciona", "cumple", "factible",
        "recomiendo", "elijo"
    ],
    "justifica_con_margen_seguridad": [
        "margen", "seguridad", "tiempo extra", "holgura",
        "suficiente", "antes de", "con tiempo"
    ],
    "prioriza_comodidad_paciente": [
        "comodidad", "paciente", "comodo", "dolor", "suave",
        "vibracion", "post-cirugia", "quirurgico"
    ],
    "analiza_calidad_vias": [
        "calidad", "pavimento", "estado", "via", "superficie",
        "irregular", "baches", "excelente", "mala"
    ],
    "considera_numero_topes": [
        "topes", "reductor", "speed", "bache", "0 topes",
        "2 topes", "12 topes", "15 topes"
    ],
    "considera_tramos_sin_pavimentar": [
        "sin pavimentar", "sin pavimento", "destapado", "trocha",
        "rural", "km sin", "0 km"
    ],
    "descarta_rutas_inadecuadas": [
        "descartada", "descarto", "no es viable", "no aplica",
        "no puede", "elimino", "rechaz"
    ],
    "justifica_con_criterio_medico": [
        "medico", "quirurgico", "cirugia", "recuperacion",
        "clinico", "rodilla", "postoperatorio"
    ],
    "determina_orden_recogida": [
        "primero", "segundo", "tercero", "cuarto", "orden",
        "recog", "primero a", "primero b"
    ],
    "optimiza_ruta": [
        "optimo", "eficiente", "menor", "mejor", "circuito",
        "minimizando", "ahorra"
    ],
    "considera_tiempo_espera_estudiantes": [
        "espera", "tiempo", "estudiantes", "minutos", "20 minutos",
        "limite", "excede"
    ],
    "calcula_tiempo_total": [
        "total", "minutos", "tiempo", "estimado", "45", "50",
        "recorrido"
    ],
    "considera_fragilidad_equipos": [
        "fragil", "delicado", "cuidado", "equipo", "sonido",
        "danio", "costoso", "proteger"
    ],
    "considera_evento_especial": [
        "concierto", "evento", "estadio", "trafico adicional",
        "congestion adicional"
    ],
    "balancea_tiempo_vs_seguridad_carga": [
        "balance", "seguridad", "tiempo", "carga",
        "fragilidad", "prioridad"
    ],
    "plan_con_margen": [
        "margen", "tiempo extra", "seguridad", "minutos antes",
        "suficiente"
    ],
    "verifica_limite_peso_cada_ruta": [
        "toneladas", "peso", "limite", "carga",
        "10 toneladas", "20 toneladas", "15 toneladas"
    ],
    "identifica_ruta_A_ilegal": [
        "ilegal", "prohibido", "multa", "no puede",
        "excede", "supera", "ruta a"
    ],
    "analiza_ruta_B_vs_C": [
        "ruta b", "ruta c", "comparacion", "versus",
        "ambas", "entre b y c"
    ],
    "considera_horario_restriccion": [
        "horario", "restriccion", "permitido", "horas",
        "6am", "8pm", "4:00 pm"
    ],
    "selecciona_ruta_legal_optima": [
        "legal", "permitida", "viable", "cumple",
        "ruta c", "via industrial"
    ],
    "verifica_capacidad_bus": [
        "capacidad", "estudiantes", "bus", "cupo",
        "26", "40", "total"
    ],
    "considera_condiciones_lluvia": [
        "lluvia", "clima", "precaucion", "velocidad",
        "reducir", "mojado", "frenado"
    ],
    "considera_seguridad_estudiantes": [
        "seguridad", "cuidado", "precaucion", "ninos",
        "menores", "escolares"
    ],
    "plan_con_horarios_cada_parada": [
        "8:15", "8:25", "8:35", "parada",
        "horario", "am"
    ],
    "considera_peaje": [
        "peaje", "costo", "pago", "sin peaje",
        "gratis", "cobro"
    ],
    "considera_preferencias_familia": [
        "familia", "preferencia", "disfruta", "paseo",
        "escenica", "bonita", "paisaje"
    ],
    "considera_comodidad_abuela": [
        "abuela", "comodidad", "despacio", "tiempo",
        "camina", "mayor"
    ],
    "considera_tiempo_suficiente": [
        "tiempo", "minutos", "llegada", "margen",
        "suficiente", "holgura"
    ],
    "considera_evento_mercado": [
        "mercado", "peatones", "domingo", "evento",
        "precaucion", "despacio"
    ],
    "selecciona_ruta": [
        "ruta", "camino", "via", "recorrido",
        "tomo", "elijo"
    ],
    "considera_urgencia": [
        "urgente", "rapido", "tiempo", "limite",
        "plazo", "deadline"
    ],

    # Task 2 - Plan Disruption
    "reconoce_problema": [
        "problema", "accidente", "bloqueo", "cierre",
        "imprevisto", "cambio", "situacion"
    ],
    "descarta_plan_original": [
        "plan original", "ruta original", "ya no",
        "no funciona", "imposible", "bloqueada"
    ],
    "propone_alternativa": [
        "alternativa", "nueva ruta", "desvio", "opcion",
        "en su lugar", "cambio a"
    ],
    "mantiene_objetivo": [
        "objetivo", "destino", "llegar", "cumplir",
        "igual", "mismo"
    ],
    "justifica_nuevo_plan": [
        "porque", "debido", "razon", "ya que",
        "permite", "viable"
    ],
    "evalua_tiempo_nuevo_plan": [
        "minutos", "tiempo", "llegada", "estimado",
        "tardara", "demora"
    ],
    "considera_impacto_pasajeros": [
        "pasajeros", "cliente", "informo", "aviso",
        "comunico", "afecta"
    ],

    # Task 3 - Rule Reversal
    "aplica_regla_nueva": [
        "regla", "nueva", "especial", "indica",
        "segun", "establece", "modificada"
    ],
    "no_aplica_regla_habitual": [
        "habitual", "normal", "usualmente", "pero",
        "aunque normalmente", "en este caso"
    ],
    "identifica_regla_especial": [
        "regla especial", "excepcion", "modificacion",
        "invertida", "diferente", "especifica"
    ],
    "explica_aplicacion": [
        "aplico", "significa", "por lo tanto",
        "entonces", "en consecuencia"
    ],
    "evita_error_perseverativo": [
        "no confundo", "a pesar", "aunque",
        "diferente a lo normal", "caso especial"
    ],

    # Task 4 - Multi-Variable
    "trackea_todas_variables": [
        "combustible", "tiempo", "trafico", "pasajeros",
        "variables", "todos los factores"
    ],
    "estado_cada_variable": [
        "nivel", "estado", "actual", "disponible",
        "restante", "queda"
    ],
    "decision_integrada": [
        "considerando", "integrando", "todo",
        "combinando", "en conjunto"
    ],
    "identifica_variable_critica": [
        "critico", "prioritario", "mas importante",
        "urgente", "limitante"
    ],
    "plan_multicriterio": [
        "plan", "accion", "estrategia",
        "considerando todo", "balance"
    ],

    # Task 5 - Priority Conflict
    "reconoce_conflicto": [
        "conflicto", "dilema", "tension", "versus",
        "contrapone", "choca", "incompatible"
    ],
    "prioriza_explicitamente": [
        "priorizo", "primero", "prioritario", "mas importante",
        "sobre", "antes que"
    ],
    "justifica_decision": [
        "porque", "razon", "justific", "ya que",
        "debido", "argumento"
    ],
    "considera_consecuencias": [
        "consecuencia", "resultado", "impacto",
        "efecto", "si no", "riesgo"
    ],
    "identifica_conflicto_tiempo_seguridad": [
        "tiempo", "seguridad", "rapidez", "riesgo",
        "conflicto", "versus"
    ],
    "prioriza_seguridad_ninos": [
        "seguridad", "ninos", "estudiantes", "menores",
        "proteger", "riesgo"
    ],
    "verifica_tiempo_disponible": [
        "tiempo", "margen", "minutos", "llegada",
        "suficiente", "a tiempo"
    ],
    "reconoce_dilema_etico": [
        "dilema", "etico", "moral", "decision",
        "dificil", "obligacion"
    ],
    "evalua_severidad_ambas_situaciones": [
        "severidad", "grave", "urgente", "critico",
        "ambas", "comparando"
    ],
    "considera_disponibilidad_otros_recursos": [
        "otros recursos", "otra unidad", "alternativa",
        "disponible", "central"
    ],
    "toma_decision_justificada": [
        "decision", "elijo", "opto", "determino",
        "concluyo", "finalmente"
    ],
    "propone_comunicacion_con_central": [
        "central", "comunico", "reporto", "informo",
        "aviso", "contacto"
    ],
    "analiza_todas_entregas": [
        "cliente a", "cliente b", "cliente c",
        "entregas", "todas", "analizo"
    ],
    "prioriza_por_impacto_y_tiempo": [
        "impacto", "tiempo", "plazo", "consecuencia",
        "critico", "prioridad"
    ],
    "calcula_viabilidad_ruta": [
        "viable", "posible", "tiempo", "llegar",
        "km", "minutos"
    ],
    "orden_de_entrega_justificado": [
        "orden", "primero", "segundo", "tercero",
        "a, b, c", "secuencia"
    ],
    "considera_consecuencias_retraso": [
        "retraso", "tarde", "consecuencia",
        "si no llego", "clinica", "firma"
    ],
    "identifica_multiples_objetivos_conflictivos": [
        "objetivos", "multiples", "conflicto",
        "a la vez", "simultane"
    ],
    "prioriza_seguridad_inmediata": [
        "seguridad", "inmediato", "primero",
        "antes", "urgente"
    ],
    "decide_momento_correccion": [
        "momento", "cuando", "despues",
        "espero", "luego de"
    ],
    "explica_secuencia_de_acciones": [
        "secuencia", "pasos", "primero", "luego",
        "despues", "finalmente"
    ],
    "considera_perfil_estudiante": [
        "principiante", "estudiante", "nivel",
        "experiencia", "aprendiz"
    ],
    "identifica_conflicto_instruccion_vs_realidad": [
        "instruccion", "jefe", "realidad", "pero",
        "sin embargo", "conflicto"
    ],
    "evalua_impacto_zona_escolar": [
        "zona escolar", "escolar", "ninos",
        "2:30", "salida", "congestion"
    ],
    "propone_solucion_y_comunicacion": [
        "comunico", "informo", "jefe", "explico",
        "soluccion", "propongo"
    ],
    "reconoce_conflicto_valores": [
        "valores", "principios", "reglamento",
        "servicio", "seguridad", "conflicto"
    ],
    "aplica_reglamento": [
        "reglamento", "norma", "regla", "prohibido",
        "no autorizado", "cumplir"
    ],
    "considera_todos_pasajeros": [
        "pasajeros", "todos", "25", "28",
        "resto", "demas"
    ],
    "comunica_decision_respetuosamente": [
        "informo", "explico", "amablemente",
        "respetuoso", "disculpe", "comprendo"
    ],
    "analiza_ambas_solicitudes_profundamente": [
        "solicitud a", "solicitud b", "ambas",
        "trabajadores", "ejecutivo", "analizo"
    ],
    "considera_alternativas_disponibles": [
        "taxi", "alternativa", "otra opcion",
        "disponible", "15 minutos"
    ],
    "decide_con_criterio_etico_y_practico": [
        "etico", "practico", "criterio",
        "decision", "razon"
    ],
    "propone_solucion_para_ambas_partes": [
        "ambas partes", "los dos", "soluccion",
        "taxi para", "vehiculo para"
    ],
    "justifica_decision_final": [
        "decision final", "concluyo", "porque",
        "razon", "justific"
    ],
    "reconoce_tension_grupo_vs_responsabilidad": [
        "grupo", "responsabilidad", "tension",
        "quieren", "pero", "mi deber"
    ],
    "evalua_riesgos_ruta_desconocida": [
        "desconocida", "riesgo", "no conozco",
        "verificada", "peligro"
    ],
    "considera_impacto_en_grupo_completo": [
        "grupo", "todos", "12", "completo",
        "cena", "reserva"
    ],
    "comunica_decision_diplomaticamente": [
        "diplomaticamente", "amable", "entiendo",
        "comprendo", "pero", "sin embargo"
    ],
    "propone_alternativa_si_existe": [
        "alternativa", "propongo", "podria",
        "proxima vez", "consultar"
    ],
    "identifica_conflicto_politica_vs_seguridad": [
        "politica", "seguridad", "conflicto",
        "empresa", "versus", "retraso"
    ],
    "evalua_riesgo_real_neblina_con_carga_peligrosa": [
        "neblina", "carga peligrosa", "riesgo",
        "visibilidad", "50m", "clase 3"
    ],
    "toma_decision_priorizando_seguridad": [
        "seguridad", "priorizo", "desvio",
        "no puedo", "debo"
    ],
    "planifica_comunicacion_con_empresa": [
        "comunico", "empresa", "reporto",
        "informo", "aviso"
    ],
    "justifica_con_marco_legal_o_etico": [
        "legal", "etico", "normativa", "codigo",
        "ley", "reglamento", "derecho"
    ],
}


def evaluate_expected_elements(
    response: str,
    expected_elements: list[str],
    element_keywords: dict[str, list[str]] | None = None
) -> dict[str, Any]:
    """
    Evalúa cuántos expected_elements están cubiertos en la respuesta.
    """
    keywords_map = element_keywords or ELEMENT_KEYWORDS

    covered = []
    missing = []

    for element in expected_elements:
        keywords = keywords_map.get(
            element,
            element.replace("_", " ").split()
        )
        if check_keyword_presence(response, keywords, threshold=0.3):
            covered.append(element)
        else:
            missing.append(element)

    score = len(covered) / len(expected_elements) if expected_elements else 0.0

    return {
        "score": round(score, 3),
        "covered": covered,
        "missing": missing,
        "total_elements": len(expected_elements),
        "covered_count": len(covered),
        "passed": score >= 0.6
    }


def get_answer_field(item: dict) -> str:
    """
    Obtiene el campo de respuesta óptima del item,
    manejando diferentes nombres de campo en los JSON.
    """
    # Task 1 y Task 5: optimal_answer
    if "optimal_answer" in item:
        return item["optimal_answer"]
    
    # Task 2: optimal_response (es un dict)
    if "optimal_response" in item:
        resp = item["optimal_response"]
        if isinstance(resp, dict):
            parts = []
            if "action" in resp:
                parts.append(f"Acción: {resp['action']}")
            if "reasoning" in resp:
                parts.append(f"Razonamiento: {resp['reasoning']}")
            if "communication" in resp:
                parts.append(f"Comunicación: {resp['communication']}")
            return " | ".join(parts) if parts else str(resp)
        return str(resp)
    
    # Task 3: explanation
    if "explanation" in item:
        return item["explanation"]
    
    # Task 4: construir respuesta desde questions
    if "questions" in item:
        questions = item["questions"]
        parts = []
        for q_item in questions:
            q_text = q_item.get("q", "")
            correct = q_item.get("correct", "")
            parts.append(f"{q_text} Respuesta: {correct}")
        return " | ".join(parts)
    
    # Fallbacks
    for field in ["answer", "correct_answer", "solution",
                  "respuesta_optima", "referencia"]:
        if field in item:
            return str(item[field])
    
    # Último recurso
    return item.get("scenario", {}).get("description", "Sin respuesta encontrada")


def build_scenario_text(item: dict) -> str:
    """
    Construye el texto del escenario desde un item del JSON,
    manejando diferentes estructuras posibles.
    """
    scenario = item.get("scenario", {})

    if isinstance(scenario, str):
        return scenario

    parts = []

    if "description" in scenario:
        parts.append(scenario["description"])

    # Mapa y condiciones
    map_data = scenario.get("map_data", {})
    if map_data:
        if "origin" in map_data:
            parts.append(f"Origen: {map_data['origin']}")
        if "destination" in map_data:
            parts.append(f"Destino: {map_data['destination']}")
        if "available_routes" in map_data:
            routes = map_data["available_routes"]
            parts.append("Rutas disponibles:")
            for r in routes:
                parts.append(f"  - {json.dumps(r, ensure_ascii=False)}")
        if "current_conditions" in map_data:
            parts.append(f"Condiciones: {map_data['current_conditions']}")

    # Restricciones
    constraints = scenario.get("constraints", [])
    if constraints:
        parts.append("Restricciones:")
        for c in constraints:
            parts.append(f"  - {c}")

    # Otros campos del escenario
    for key in ["situation", "disruption", "new_rule",
                "variables", "conflict", "passengers",
                "deliveries", "routes", "options"]:
        if key in scenario:
            val = scenario[key]
            parts.append(f"{key.title()}: {json.dumps(val, ensure_ascii=False)}")

    return "\n".join(parts) if parts else json.dumps(scenario, ensure_ascii=False)


def build_evaluation_summary(
    task_name: str,
    results_list: list[dict]
) -> dict[str, Any]:
    """Genera resumen estadístico de los resultados."""
    if not results_list:
        return {"task": task_name, "error": "Sin resultados"}

    scores = [r.get("score", 0) for r in results_list]
    passed = [r.get("passed", False) for r in results_list]

    return {
        "task": task_name,
        "total_items": len(results_list),
        "passed": sum(passed),
        "failed": len(passed) - sum(passed),
        "pass_rate": round(sum(passed) / len(passed), 3),
        "avg_score": round(sum(scores) / len(scores), 3),
        "min_score": round(min(scores), 3),
        "max_score": round(max(scores), 3),
    }


# ─── Test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== TEST utils.py ===\n")

    # Test 1: Cargar datos
    data = load_json_data("task1_route_planning.json")
    print(f"✅ Cargados {len(data)} items de task1")

    # Test 2: Filtrar por dificultad
    easy   = filter_by_difficulty(data, "easy")
    medium = filter_by_difficulty(data, "medium")
    hard   = filter_by_difficulty(data, "hard")
    print(f"✅ Distribución: easy={len(easy)}, "
          f"medium={len(medium)}, hard={len(hard)}")

    # Test 3: Verificar get_answer_field en todos los JSON
    import glob
    print("\n✅ Verificando campos de respuesta en todos los JSON:")
    for filepath in sorted(glob.glob("data/*.json")):
        try:
            d = json.load(open(filepath))
            if d:
                answer = get_answer_field(d[0])
                print(f"   {filepath}: campo encontrado "
                      f"({len(answer)} chars)")
        except Exception as e:
            print(f"   ❌ {filepath}: {e}")

    # Test 4: Evaluar respuesta de muestra
    sample_item = data[0]
    evaluation = evaluate_expected_elements(
        response=get_answer_field(sample_item),
        expected_elements=sample_item["expected_elements"],
    )
    print(f"\n✅ Evaluación con respuesta óptima:")
    print(f"   Score: {evaluation['score']}")
    print(f"   Pasó: {evaluation['passed']}")
    print(f"   Cubiertos: {evaluation['covered']}")
    print(f"   Faltantes: {evaluation['missing']}")
    print("\n✅ utils.py OK")


# ─── PATCH: Reemplazar keywords por palabras sueltas ─────────
# Necesario porque las frases multi-palabra no matchean bien
KEYWORDS_PATCH = {
    # Task 2 - Plan Disruption
    "reconoce_problema":                  ["desvio","alerta","accidente","bloqueo","imprevisto"],
    "reconoce_problema_critico":          ["deficit","autonomia","arriesgar","varado","critico"],
    "reconoce_urgencia_medica":           ["medica","urgencia","infarto","hospital","sintoma"],
    "reconoce_gravedad":                  ["codigo","rojo","sirena","activar","luces"],
    "reconoce_riesgo":                    ["taller","tarde","limite","riesgo","peligro"],
    "evalua_opciones":                    ["opcion","opciones","alternativa","podria","analizo"],
    "evalua_opciones_disponibles":        ["opcion","alternativa","posible","buscar","regresar"],
    "evalua_opciones_tiempo":             ["minutos","tiempo","horas","tardaria","rapido"],
    "evalua_opciones_medicas":            ["paramedico","hospital","medico","atencion","urgencias"],
    "evalua_urgencia":                    ["urgente","manana","medicina","farmacia","esperar"],
    "evalua_riesgo_mecanico":             ["motor","falla","sobrecalentando","mecanico","continuar"],
    "evalua_gravedad_sintomas":           ["dolor","pecho","infarto","anos","puede","sintoma"],
    "evalua_impacto_cambio":              ["urgencia","cirugia","prioridad","invertir","orden"],
    "evalua_si_llega_a_tiempo":           ["tarde","partido","tiempo","minutos","llegara"],
    "evalua_nivel_estudiante":            ["principiante","clase","listo","trafico","experiencia"],
    "selecciona_mejor_alternativa":       ["opcion","b","desvio","mejor","tomar","recomiendo"],
    "calcula_nuevo_tiempo":               ["minutos","pm","am","adicionales","estimado","llegaria"],
    "recalcula_tiempos":                  ["hospital","salida","llegada","pm","minutos","techcorp"],
    "comunica_al_pasajero":               ["senor","tranquilo","preocupe","llegaremos","aviso"],
    "comunica_a_pasajeros":               ["pasajera","pasajeros","regresar","hospital","informo"],
    "comunica_con_taller":                ["taller","llama","llamar","esperar","preguntar"],
    "comunica_profesionalmente":          ["detenerse","gestionar","mejor","opcion","taxi"],
    "considera_comunicacion_techcorp":    ["techcorp","pm","llegada","tarde","hospital"],
    "mantiene_calma":                     ["quizas","pequeno","llegaremos","bien","puede"],
    "mantiene_calma_estudiante":          ["gradualmente","encontraras","normal","construye","aprender"],
    "toma_decision":                      ["tomar","ir","regresar","desvio","opcion","decido"],
    "toma_decision_priorizada":           ["primero","hospital","urgencia","invertir","prioridad"],
    "toma_decision_practica":             ["farmacia","regreso","supermercado","primero","luego"],
    "toma_decision_informada":            ["decision","medica","paramedico","input","consultar"],
    "toma_decision_balanceada":           ["mejor","perder","minutos","taxi","reemplazo","detenerse"],
    "toma_decision_segura":               ["regresar","conocida","arriesgar","no","segura"],
    "toma_accion_inmediata_salud":        ["regresar","honda","inmediata","accion","hospital"],
    "prioriza_emergencia_medica":         ["emergencia","medica","infarto","prioridad","ignorar"],
    "contacta_empresa":                   ["contactar","empresa","llamo","reporto","informo"],
    "consulta_paramédico":                ["paramedico","consultar","inmediatamente","puede","sobrevivir"],
    "ejecuta_protocolo_emergencia":       ["protocolo","emergencia","sirena","luces","codigo"],
    "plan_detallado":                     ["regresar","logistica","despues","primero","luego"],
    "luego_resuelve_logistica":           ["logistica","despues","empresa","arreglo","luego"],
    "busca_alternativas":                 ["buscar","alternativa","estacion","regresar","ultima"],
    "considera_opciones_ruta":            ["opciones","farmacia","supermercado","opcion","ruta"],
    "considera_seguridad":                ["seguro","segura","riesgo","trafico","listo"],
    "considera_seguridad_pasajera":       ["pasajera","segura","varada","mejor","detenerse"],
    "considera_tiempo_vs_capacidad":      ["tiempos","consultar","paramedico","sobrevivir","45"],
    "optimiza_recorrido":                 ["farmacia","regreso","supermercado","vuelta","primero"],
    "propone_solucion_proactiva":         ["llama","taller","preguntar","esperar","minutos"],
    "tiene_plan_B":                       ["alternativa","buscar","otra","opcion","si"],
    "ofrece_solucion":                    ["taxi","gestionar","solucion","reemplazo"],
    "maneja_expectativas_hijo":           ["hijo","quizas","pequeno","primeros","partido"],
    "no_arriesga_familia":                ["arriesgar","familia","varado","regresar","no"],
    "adapta_metodologia":                 ["avenida","alternativa","secundarias","calles","adaptar"],
    "convierte_problema_en_oportunidad":  ["gradualmente","encontraras","aprender","experiencia","construye"],
    "justifica_decision":                 ["porque","urgencia","prioridad","techcorp","mejor"],

    # Task 3 - Rule Reversal
    "reconoce_regla_invertida":               ["invertida","invertido","opuesta","diferente","contraria"],
    "identifica_regla_invertida":             ["invertido","pico","placa","invertida","regla"],
    "aplica_regla_nueva_no_habitual":         ["ceder","detenerte","detenerse","cedo","nueva"],
    "aplica_regla_invertida":                 ["invertida","invertido","invierte","regla","placa"],
    "aplica_inversion_correctamente":         ["invertido","si","puede","circular","placa","par"],
    "aplica_protocolo_correctamente":         ["precaucion","pero","no","sentido","contrario"],
    "aplica_tecnica_contraintuitiva":         ["acelerar","esquivas","instinto","pero","natural"],
    "justifica_segun_contexto":               ["pais","invertida","porque","regla","habitual"],
    "no_sigue_automatismo":                   ["invertidas","verde","pare","significa","pero"],
    "no_sigue_regla_habitual":                ["habitual","pero","invertido","normalmente","placa"],
    "no_aplica_rutina_automatica":            ["espejos","extra","cauteloso","evitar","visibilidad"],
    "no_sobrepasa_limites_emergencia":        ["pero","no","sentido","contrario","emergencia"],
    "inhibe_respuesta_instintiva_frenar":     ["instinto","frenar","pero","mojado","control"],
    "inhibe_respuesta_verbal_automatica":     ["instinto","gritar","inutil","sordo","freno"],
    "inhibe_impulso_de_adelantar":            ["adelantar","carga","sobredimensionada","mantener","no"],
    "comprende_sistema_izquierdo":            ["izquierda","izquierdo","paises","conduccion","carril"],
    "espera_trafico_contrario":               ["esperar","trafico","frente","luego","cruzar"],
    "identifica_horario_especial":            ["horario","6","9","am","7","30","dentro"],
    "identifica_que_derecha_es_giro_amplio":  ["derecha","equivalente","giro","amplio","izquierda"],
    "prioriza_seguridad":                     ["cauteloso","seguridad","evitar","visibilidad","cuidado"],
    "prioriza_seguridad_inmediata":           ["freno","auxiliar","inmediatamente","instructor","usar"],
    "prioriza_seguridad_sobre_tecnicismo":    ["ninos","presentes","proposito","proteger","independientemente"],
    "reconoce_espiritu_de_la_ley":           ["proposito","proteger","ninos","independientemente","objetivo"],
    "reconoce_limitacion_del_vehiculo":       ["posicion","visibilidad","limitada","vehiculo","cabina"],
    "recuerda_instruccion_especifica":        ["instructor","dijo","diste","clases","indicado"],
    "recuerda_restriccion_especifica":        ["sobredimensionada","nunca","no","carga","adelantar"],
    "acepta_limitacion_operativa":            ["adelantar","mantener","distancia","esperar","carga"],
    "adapta_comportamiento_a_contexto":       ["ninos","presentes","independientemente","dia","contexto"],
    "busca_solucion_adaptada":                ["espejos","adicionales","extra","cauteloso","visibilidad"],
    "distingue_reglas_que_si_cambian":        ["permite","ciertas","excepciones","rojo","precaucion"],
    "distingue_reglas_que_no_cambian":        ["pero","no","sentido","contrario","nunca","carga"],
    "selecciona_intervencion_efectiva":       ["freno","auxiliar","instructor","inmediatamente","usar"],
    "considera_discapacidad_estudiante":      ["sordo","estudiante","gritar","inutil","freno"],
    "sigue_protocolo_carga_especial":         ["sobredimensionada","nunca","carga","adelantar","protocolo"],

    # Task 4 - Multi Variable
    "calcula_combustible_restante":    ["26","litros","50","24","horas","consumidos"],
    "suma_recaudacion":                ["75","000","total","25","18","32"],
    "calcula_tiempo_restante":         ["7","horas","meta","10","faltan","diaria"],
    "calcula_meta":                    ["meta","objetivo","horas","faltan","completar"],
    "calcula_hora_llegada":            ["10","am","6","30","40","minutos","paradas"],
    "compara_tiempos":                 ["18","disponibles","estimados","sin","margen"],
    "compara_disponibilidad":          ["disponible","esperando","carga","combustible","80"],
    "compara_autonomia_vs_distancia":  ["autonomia","500","320","km","alcanza"],
    "anticipa_problemas":              ["mantenimiento","20","minutos","debe","completarse"],
    "asigna_recursos_por_tiempo":      ["pista","libre","3","minutos","vuelo"],
    "asocia_estudiante_con_debilidad": ["espejos","reforzar","carlos","debilidad"],
    "cuenta_clases_completadas":       ["2","ana","carlos","clases","completadas"],
    "cuenta_entregas_completadas":     ["2","p1","p3","entregas","pendientes","1"],
    "cuenta_operativos":               ["3","vehiculo","1","2","4","taller"],
    "cuenta_pasajeros":                ["4","2","adultos","ninos","pasajeros"],
    "cuenta_por_categoria":            ["2","camion","alimentos","medicamentos","refrigerados"],
    "cuenta_recursos_disponibles":     ["2","pista","a","b","c","cerrada"],
    "distingue_normal_vs_alerta":      ["normal","alerta","llanta","temperatura","limite"],
    "evalua_oportunidad_tiempo_limitado": ["2","minutos","aeropuerto","45","000","cerca"],
    "identifica_conflictos":           ["mantenimiento","20","minutos","operacion","antes"],
    "identifica_emergencias":          ["llanta","baja","buscando","taller","miguel"],
    "identifica_necesidades_criticas": ["combustible","20","estacion","antes","ana"],
    "identifica_problema_secundario":  ["combustible","bajo","25","luz","debe"],
    "identifica_proximo_estudiante":   ["ana","manana","8","00","am","examen"],
    "identifica_urgentes_pendientes":  ["2","p1","p3","limite","3","pm"],
    "identifica_urgencia":             ["2","minutos","decidir","aeropuerto","limite"],
    "monitorea_condiciones_criticas":  ["temperatura","5","limite","medicamentos","monitorearse"],
    "prioriza_correctamente":          ["reunion","prioridad","tiempo","combustible","alcanza"],
    "prioriza_por_deadline":           ["limite","3","pm","farmacia","30","minutos"],
    "proyecta_ingresos":               ["140","000","45","15","12","212"],
    "recuerda_condiciones_especiales": ["refrigeracion","temperatura","4","medicamentos","verificar"],
    "recuerda_incidente":              ["discusion","estudiantes","parada","2","resuelta"],
    "recuerda_notas_especificas":      ["espejos","ana","examen","licencia","aprendizaje"],
    "recuerda_tiempos":                ["2","horas","vehiculos","taller","esperando"],
    "suma_resta_estudiantes":          ["28","8","12","3","6","5","estudiantes"],
    "verifica_capacidad":              ["40","28","espacios","disponibles","capacidad"],
    "verifica_cumplimiento_legal":     ["7","horas","8","permitidas","limite","roberto"],
    "verifica_requisitos":             ["licencia","aprendizaje","primera","clase","traiga"],
}

# Aplicar patch: sobrescribir entradas en ELEMENT_KEYWORDS
ELEMENT_KEYWORDS.update(KEYWORDS_PATCH)
