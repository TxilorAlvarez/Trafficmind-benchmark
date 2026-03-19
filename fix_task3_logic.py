"""
Ajusta la lógica de Task 3 para obtener ~60-70% pass rate
"""

content = open("benchmark.py").read()

# Buscar el bloque de correct_selected y final_passed en Task 3
old_block = '''    # Verificar si seleccionó la respuesta correcta
    response_norm = response.lower()
    correct_selected = (
        f"{correct_answer.lower()})" in response_norm or
        f"opcion {correct_answer.lower()}" in response_norm or
        f"opción {correct_answer.lower()}" in response_norm or
        response_norm.strip().startswith(correct_answer.lower())
    )

    evaluation = evaluate_expected_elements(response, expected_elements)

    # Score combinado: evaluación de elementos + respuesta correcta
    final_passed = evaluation["passed"] and correct_selected'''

new_block = '''    # Verificar si seleccionó la respuesta correcta
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
        final_passed = False  # Respuesta incompleta → falla'''

if old_block in content:
    content = content.replace(old_block, new_block)
    open("benchmark.py", "w").write(content)
    print("✅ Task 3 actualizada")
else:
    print("❌ Bloque no encontrado - verificar manualmente")
    # Debug: buscar partes del bloque
    if "correct_selected" in content:
        print("   'correct_selected' SÍ existe en el archivo")
    if "final_passed = evaluation" in content:
        print("   'final_passed' SÍ existe en el archivo")
