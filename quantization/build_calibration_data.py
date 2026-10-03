#!/usr/bin/env python3
"""Textos para cuantizar y comparar versiones GGUF del modelo.

- quant_data/calib.txt: calibración de la imatrix (qué pesos importan). Mezcla de dominios para que
  la cuantización no sacrifique nada de lo que se va a usar.
- quant_data/eval/<dominio>.txt: textos DISTINTOS a los de calibración para medir, por dominio,
  cuánto se aleja cada versión cuantizada del modelo en bf16 (KL-divergencia y perplejidad).

Dominios y peso en la calibración:
  solidity 50% (ejemplos de train; eval sale de test) · foundry 15% (tests de Foundry con PoCs de exploits)
  blockchain 10% (estándares ERC, CC0) · python 10% (librería estándar) · code 5% (Go/Rust)
  es 10% (Wikipedia en español, CC BY-SA)

    python3 quantization/build_calibration_data.py --data data --foundry-dir <pocs> --erc-dir <ERCS> \\
        --code-dir <go_rust_src> --es-dir <eswiki_txt>
"""
import argparse, hashlib, json, random, re
from pathlib import Path

OUT = Path("quant_data")
CALIB_CHARS = 560_000          # ~160k tokens: ~300 bloques de 512 para llama-imatrix
EVAL_CHARS = 14_000            # ~4k tokens por dominio (ficheros de logits de ~2 GB por dominio)
SHARE = {"solidity": .50, "foundry": .15, "blockchain": .10, "python": .10, "code": .05, "es": .10}
SNIPPET = 3_000
rnd = random.Random(1234)


def in_eval(name):  # partición determinista y estable de ficheros
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % 5 == 0


def snippets(text, n=SNIPPET):
    """Trozos de ~n caracteres cortados en salto de línea."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    out, i = [], 0
    while i < len(text):
        j = text.rfind("\n", i, i + n)
        j = i + n if j <= i else j
        chunk = text[i:j].strip()
        if len(chunk) > 400:
            out.append(chunk)
        i = j + 1
    return out


def take(pool, budget):
    rnd.shuffle(pool)
    out, total = [], 0
    for s in pool:
        if total >= budget:
            break
        out.append(s)
        total += len(s)
    return out


def chat_text(row):
    """Ejemplo de chat como texto plano: código, razonamiento (si hay) e informe."""
    m = row["messages"]
    parts = [m[1]["content"]]
    if m[-1].get("reasoning_content"):
        parts.append(m[-1]["reasoning_content"])
    parts.append(m[-1]["content"])
    return "\n\n".join(parts)


def files_text(paths, limit=200_000):
    return [p.read_text(errors="replace")[:limit] for p in paths]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="carpeta con train_mixed.jsonl y test.jsonl")
    ap.add_argument("--foundry-dir", required=True, help="tests de Foundry (*_exp.sol); los de 2026-* van a evaluación")
    ap.add_argument("--erc-dir", required=True, help="carpeta con los ERC en Markdown")
    ap.add_argument("--python-dir", default="/usr/lib/python3.13")
    ap.add_argument("--code-dir", required=True, help="código Go/Rust de ejemplo")
    ap.add_argument("--es-dir", required=True, help="artículos de Wikipedia en español (.txt)")
    a = ap.parse_args()
    data = Path(a.data)
    (OUT / "eval").mkdir(parents=True, exist_ok=True)
    pools = {}

    # solidity: calibración con train (con razonamientos), evaluación con test
    train = [json.loads(l) for l in open(data / "train_mixed.jsonl")]
    test = [json.loads(l) for l in open(data / "test.jsonl")]
    pools["solidity"] = (sum((snippets(chat_text(r)) for r in train), []),
                         sum((snippets(chat_text(r)) for r in test), []))

    # foundry: PoCs de exploits (tests de Foundry); evaluación con los de 2026
    pocs = sorted(Path(a.foundry_dir).rglob("*_exp.sol"))
    pools["foundry"] = (sum((snippets(t) for t in files_text([p for p in pocs if "/2026-" not in str(p)])), []),
                        sum((snippets(t) for t in files_text([p for p in pocs if "/2026-" in str(p)])), []))

    # blockchain: estándares ERC (prosa técnica + especificaciones)
    ercs = sorted(Path(a.erc_dir).glob("*.md"))
    pools["blockchain"] = (sum((snippets(t) for t in files_text([p for p in ercs if not in_eval(p.name)])), []),
                           sum((snippets(t) for t in files_text([p for p in ercs if in_eval(p.name)])), []))

    # python: librería estándar
    py = sorted(Path(a.python_dir).glob("*.py"))
    pools["python"] = (sum((snippets(t) for t in files_text([p for p in py if not in_eval(p.name)])), []),
                       sum((snippets(t) for t in files_text([p for p in py if in_eval(p.name)])), []))

    # code: Go y Rust (otros lenguajes)
    code = sorted(list(Path(a.code_dir).rglob("*.go")) + list(Path(a.code_dir).rglob("*.rs")))
    code = [p for p in code if p.is_file() and "_test" not in p.name]
    pools["code"] = (sum((snippets(t) for t in files_text([p for p in code if not in_eval(p.name)])), []),
                     sum((snippets(t) for t in files_text([p for p in code if in_eval(p.name)])), []))

    # español: Wikipedia
    wiki = sorted(Path(a.es_dir).glob("*.txt"))
    docs = []
    ev_wiki = wiki[::4]  # pocos artículos: se reservan 1 de cada 4 para evaluación
    pools["es"] = (sum((snippets(t) for t in files_text([p for p in wiki if p not in ev_wiki] + docs)), []),
                   sum((snippets(t) for t in files_text(ev_wiki)), []))

    calib, stats = [], {}
    for dom, (cal_pool, ev_pool) in pools.items():
        cal = take(list(cal_pool), int(CALIB_CHARS * SHARE[dom]))
        ev = take(list(ev_pool), EVAL_CHARS)
        calib += cal
        (OUT / "eval" / f"{dom}.txt").write_text("\n\n".join(ev) + "\n")
        stats[dom] = {"calib_chars": sum(map(len, cal)), "eval_chars": sum(map(len, ev)),
                      "pool_calib": len(cal_pool), "pool_eval": len(ev_pool)}
    rnd.shuffle(calib)
    (OUT / "calib.txt").write_text("\n\n".join(calib) + "\n")
    stats["calib_total_chars"] = sum(map(len, calib))
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
