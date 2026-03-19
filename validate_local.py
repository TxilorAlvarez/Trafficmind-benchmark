import json
import re
from utils import evaluate_expected_elements, get_answer_field

def test_task3():
    print("\n" + "="*60)
    print("TASK 3: Rule Reversal - Nueva logica balanceada")
    print("="*60)
    data = json.load(open("data/task3_rule_reversal.json"))
    passed = 0
    for item in data:
        response = item.get("explanation", "")
        correct_answer = item.get("correct_answer", "B")
        expected = item.get("expected_elements", [])
        ev = evaluate_expected_elements(response, expected)
        rn = response.lower()
        correct_selected = (
            f"{correct_answer.lower()})" in rn or
            f"opcion {correct_answer.lower()}" in rn or
            f"opción {correct_answer.lower()}" in rn or
            rn.strip().startswith(correct_answer.lower())
        )
        # Nueva lógica balanceada
        semantic_score = ev["score"]
        if semantic_score >= 0.8:
            final_passed = True
        elif semantic_score >= 0.6 and correct_selected:
            final_passed = True
        else:
            final_passed = False
        if final_passed: passed += 1
        print(f"  [{item['task_id']}] [{item['difficulty']}] "
              f"sem={semantic_score:.2f} correct={correct_selected} "
              f"passed={final_passed}")
    print(f"\n  RESULTADO: {passed}/10 = {passed*10}%")
    return passed

def test_task4():
    print("\n" + "="*60)
    print("TASK 4: Multi-Variable - Evaluacion numerica")
    print("="*60)
    data = json.load(open("data/task4_multi_variable.json"))
    passed = 0
    for item in data:
        expected = item.get("expected_elements", [])
        questions = item.get("questions", [])
        correct_answers = [q.get("correct","") for q in questions]
        simulated = " | ".join(correct_answers)
        ev = evaluate_expected_elements(simulated, expected)
        numeric_matches = 0
        for correct in correct_answers:
            nc = re.findall(r'\d+', correct.replace(",",""))
            nr = re.findall(r'\d+', simulated.replace(",",""))
            key = [n for n in nc if len(n) >= 2]
            if key:
                hits = sum(1 for n in key if n in nr)
                if hits/len(key) >= 0.5:
                    numeric_matches += 1
        num_acc = numeric_matches/len(correct_answers) if correct_answers else 0
        combined = (ev["score"]*0.4) + (num_acc*0.6)
        new_pass = combined >= 0.65
        if new_pass: passed += 1
        print(f"  [{item['task_id']}] [{item['difficulty']}] "
              f"sem={ev['score']:.2f} num={num_acc:.2f} "
              f"comb={combined:.2f} passed={new_pass}")
    print(f"\n  RESULTADO: {passed}/10 = {passed*10}%")
    return passed

def test_otras():
    print("\n" + "="*60)
    print("TASKS 1, 2, 5 - Con respuestas optimas")
    print("="*60)
    tasks = [
        ("task1_route_planning.json",    "Task1 Route Planning   "),
        ("task2_plan_disruption.json",   "Task2 Plan Disruption  "),
        ("task5_priority_conflict.json", "Task5 Priority Conflict"),
    ]
    results = {}
    for fname, name in tasks:
        data = json.load(open(f"data/{fname}"))
        passed = 0
        scores = []
        for item in data:
            resp = get_answer_field(item)
            ev = evaluate_expected_elements(resp, item.get("expected_elements",[]))
            if ev["passed"]: passed += 1
            scores.append(ev["score"])
        avg = sum(scores)/len(scores)
        results[name] = passed
        print(f"  {name}: {passed}/10 = {passed*10}% | avg={avg:.2f}")
    return results

def resumen(t3, t4, otras):
    print("\n" + "="*60)
    print("RESUMEN FINAL ESPERADO CON GPT")
    print("="*60)
    t1 = otras.get("Task1 Route Planning   ", 6)
    t2 = otras.get("Task2 Plan Disruption  ", 9)
    t5 = otras.get("Task5 Priority Conflict", 8)
    scores = {
        "Task1 Route Planning  ": (t1/10, 0.20),
        "Task2 Plan Disruption ": (t2/10, 0.25),
        "Task3 Rule Reversal   ": (t3/10, 0.20),
        "Task4 Multi-Variable  ": (t4/10, 0.15),
        "Task5 Priority Conflict": (t5/10, 0.20),
    }
    aggregate = 0
    for name, (rate, weight) in scores.items():
        bar = "█" * int(rate*20)
        print(f"  {name} {bar:<20} {rate*100:.0f}% (w:{weight})")
        aggregate += rate * weight
    print(f"\n  AGGREGATE SCORE: {aggregate:.4f} ({aggregate*100:.1f}%)")
    zona = "IDEAL ✅" if 0.55 <= aggregate <= 0.80 else "REVISAR ⚠️"
    print(f"  Zona discriminatoria: {zona}")

if __name__ == "__main__":
    print("VALIDACION LOCAL SIN API - TrafficMind Benchmark")
    t3 = test_task3()
    t4 = test_task4()
    otras = test_otras()
    resumen(t3, t4, otras)
