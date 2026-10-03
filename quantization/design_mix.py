#!/usr/bin/env python3
"""Diseño de mezclas de cuantización por tensor guiadas por la imatrix (objetivo: V100 32 GB, ~Q4 de tamaño).

Entrada: q/tensors_B.json (tipos y tamaños de la versión B) y q/imatrix_stats.txt (importancia por tensor).
Salida:  q/<nombre>.args con las reglas --tensor-type para llama-quantize, y el tamaño estimado.

    python3 quantization/design_mix.py --budget 18.4
"""
import argparse, json, re

BYTES = {"Q3_K": 110 / 256, "Q4_K": 144 / 256, "Q5_K": 176 / 256, "Q6_K": 210 / 256, "Q8_0": 34 / 32, "F32": 4.0}


def load():
    tensors = {t["name"]: t for t in json.load(open("q/tensors_B.json"))}
    imp = {}
    for line in open("q/imatrix_stats.txt"):
        m = re.match(r".*?I\s+(\d+)\t(\S+)\s*\t\s*([0-9.]+)\t", line)
        if m:
            imp[f"blk.{m[1]}.{m[2]}.weight"] = float(m[3])
    return tensors, imp


def size(tensors, plan):
    return sum(t["n"] * BYTES[plan.get(name, t["type"])] for name, t in tensors.items()) / 1e9


def rank_layers(imp, kind, n_layers=64):
    """Capas ordenadas por importancia (Σ Act²) para un tipo de tensor; de más a menos."""
    vals = [(imp.get(f"blk.{l}.{kind}.weight", 0), l) for l in range(n_layers)]
    return [l for _, l in sorted(vals, reverse=True)]


SENSIBLES = r"ffn_down|attn_qkv|ssm_out|attn_output|attn_v"   # salidas de bloque y tensores delicados


def base_upgrades(tensors, pattern=SENSIBLES):
    """Q6_K EXPLÍCITO en todos los tensores sensibles, también los que Q4_K_M ya tenía en Q6_K:
    al añadir reglas, la heurística interna de Q4_K_M se desplaza y puede bajar a Q4 los que no
    tengan regla propia (comprobado en G: 34 tensores Q6_K -> Q4_K)."""
    plan = {}
    for name, t in tensors.items():
        if re.search(rf"\.({pattern})\.weight$", name) or name == "output.weight":
            plan[name] = "Q6_K"
    return plan


def design(tensors, imp, variant, budget):
    up_rank = rank_layers(imp, "ffn_up")  # gate y up comparten entrada: misma importancia
    if variant == "G":
        plan = base_upgrades(tensors)
    else:
        plan = base_upgrades(tensors, SENSIBLES + "|attn_q|attn_k")
        # Σ(Act²) crece con la profundidad y no distingue bien entre capas intermedias: se bajan a
        # Q3_K los ffn_up/gate de capas ALTERNAS de la franja central (17, 19, ..., 47), para no
        # degradar un tramo seguido; lo ahorrado paga attn_q/k a Q6 y más ffn a Q5 en las últimas capas
        low = list(range(17, 48, 2))
        plan["token_embd.weight"] = "Q6_K"
        for l in low:
            for k in ("ffn_up", "ffn_gate"):
                plan[f"blk.{l}.{k}.weight"] = "Q3_K"
    # con el presupuesto restante, subir ffn_up/gate a Q5_K de las capas más importantes hacia abajo
    for l in up_rank:
        names = [f"blk.{l}.{k}.weight" for k in ("ffn_up", "ffn_gate")]
        if any(plan.get(n) == "Q3_K" for n in names):
            continue
        trial = dict(plan, **{n: "Q5_K" for n in names})
        if size(tensors, trial) > budget:
            break
        plan = trial
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=18.4, help="GB máximos del fichero")
    a = ap.parse_args()
    tensors, imp = load()
    print(f"B (Q4_K_M + imatrix): {size(tensors, {}):.2f} GB")
    for variant, fname in (("G", "I-MIX-v100-g2"),):
        plan = design(tensors, imp, variant, a.budget)
        counts = {}
        for name, q in plan.items():
            counts[q] = counts.get(q, 0) + 1
        q5 = sorted({int(n.split(".")[1]) for n, q in plan.items() if q == "Q5_K"})
        q3 = sorted({int(n.split(".")[1]) for n, q in plan.items() if q == "Q3_K"})
        print(f"{fname}: {size(tensors, plan):.2f} GB | cambios {counts} | ffn Q5 en capas {q5}"
              + (f" | ffn Q3 en capas {q3}" if q3 else ""))
        with open(f"q/{fname}.args", "w") as f:
            for name, q in sorted(plan.items()):
                f.write(f"--tensor-type\n{re.escape(name)}={q.lower()}\n")


if __name__ == "__main__":
    main()
