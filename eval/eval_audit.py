#!/usr/bin/env python3
"""Evaluar un modelo sobre un test.jsonl a través de un servidor compatible con OpenAI
(llama.cpp `llama-server`, vLLM, Ollama...). Sirve igual para el base y para el nuestro.

    python3 eval/eval_audit.py --test data/test.jsonl --run base-q4
    python3 eval/eval_audit.py --test data/test.jsonl --run ft-q4

Métricas (sin juez LLM, deterministas):
- negativos: % en los que el modelo reporta algún hallazgo H/M (falsos positivos). Menor es mejor.
- positivos: % de ficheros con al menos un hallazgo reportado (detección) y % de hallazgos de
  referencia cuya función aparece en la respuesta (recall por ubicación). Mayor es mejor.
- nº medio de hallazgos reportados por fichero (para ver si el modelo "dispara a todo").
Se puede relanzar: continúa donde lo dejó.
"""
import argparse, json, os, re, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path



# Se añade al system prompt en la evaluación para que el modelo base use el mismo formato
FORMAT_HINT = (
    "\n\nAnswer format:\n## Findings\n\n### [H-1] <title>\n\n**Severity:** High|Medium  \n"
    "**Location:** `Contract.function()` in `<file>`\n\n<root cause, attack path, impact, recommendation>\n\n"
    "(one ### section per finding). If there are none, answer exactly: "
    "'## Findings\n\nNo High or Medium severity vulnerabilities were identified in `<file>`.'"
)
NO_FINDINGS = re.compile(r"no high or medium severity vulnerabilit", re.I)
FINDING_HEAD = re.compile(r"^#{2,4}\s*\[?[HM]-\d+\]?", re.M)
SEVERITY = re.compile(r"\*\*Severity:?\*\*:?\s*(High|Medium|Critical)", re.I)
LOC_FUNCS = re.compile(r"`(?:\w+\.)?(\w+)\(\)`")
REFUSAL = re.compile(r"\b(I can(?:no|')t (?:help|assist|provide)|I(?: am|'m) (?:not able|unable) to (?:help|assist|provide)|"
                     r"I won't (?:help|provide)|cannot assist with|against my (?:guidelines|policy))", re.I)


def ask(base_url, model, messages, max_tokens, api_key, thinking=False, providers=""):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            # con thinking se usa muestreo suave (el razonamiento en greedy tiende a repetirse)
            **({"temperature": 0.6, "top_p": 0.95} if thinking else {"temperature": 0})}
    if "openrouter.ai" in base_url:
        body["usage"] = {"include": True}
        if providers:
            body["provider"] = {"order": providers.split(","), "allow_fallbacks": False}
    else:
        body["chat_template_kwargs"] = {"enable_thinking": thinking}
    req = urllib.request.Request(f"{base_url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.load(r)
    if "error" in j:
        raise RuntimeError(j["error"])
    text = j["choices"][0]["message"]["content"] or ""
    # el razonamiento puede venir aparte (reasoning_content) o dentro del texto: se evalúa solo la respuesta
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.split("</think>")[-1].strip(), float((j.get("usage") or {}).get("cost") or 0)


def n_findings(text):
    if NO_FINDINGS.search(text) and not SEVERITY.search(text):
        return 0
    return max(len(FINDING_HEAD.findall(text)), len(SEVERITY.findall(text)))


def reference(row):
    """Hallazgos de referencia: lista de conjuntos de funciones de su **Location:**."""
    ans = row["messages"][2]["content"]
    refs = []
    for part in ans.split("\n---\n"):
        m = re.search(r"\*\*Location:\*\*(.*)", part)
        if m:
            refs.append(set(LOC_FUNCS.findall(m.group(1))))
    return refs


def score(row, pred):
    neg = bool(row["meta"].get("negative"))
    k = n_findings(pred)
    out = {"negative": neg, "predicted_findings": k, "block": row["meta"]["block"],
           "refusal": bool(REFUSAL.search(pred))}
    # funciones que el modelo señala en sus líneas **Location:** (lo que de verdad "acusa")
    loc_fns = set(re.findall(r"(\w+)\(\)", " ".join(re.findall(r"\*\*Location:?\*\*:?(.*)", pred))))
    out["loc_pred"] = len(loc_fns)
    if not neg:
        refs = [fs for fs in reference(row) if fs]  # sin función en la ubicación: no medible
        hit = [any(re.search(rf"\b{re.escape(f)}\b", pred) for f in fs) for fs in refs]
        allref = set().union(*refs) if refs else set()
        out.update(ref_findings=len(refs), matched=sum(hit),
                   matched_strict=sum(any(f in loc_fns for f in fs) for fs in refs),
                   loc_hit=len(loc_fns & allref))
    return out


def summarize(results):
    def agg(rs):
        neg = [r for r in rs if r["negative"]]
        pos = [r for r in rs if not r["negative"]]
        ref = sum(r["ref_findings"] for r in pos)
        return {
            "n": len(rs),
            "false_positive_rate_on_negatives": round(sum(r["predicted_findings"] > 0 for r in neg) / len(neg), 3) if neg else None,
            "detection_rate_on_positives": round(sum(r["predicted_findings"] > 0 for r in pos) / len(pos), 3) if pos else None,
            "location_recall": round(sum(r["matched"] for r in pos) / ref, 3) if ref else None,
            # estrictas: solo cuenta lo que el modelo pone en **Location:** (no menciones sueltas en el texto)
            "location_recall_strict": round(sum(r.get("matched_strict", 0) for r in pos) / ref, 3) if ref else None,
            "location_precision": round(sum(r.get("loc_hit", 0) for r in pos) / max(1, sum(r.get("loc_pred", 0) for r in pos)), 3) if pos else None,
            "avg_predicted_findings": round(sum(r["predicted_findings"] for r in rs) / len(rs), 2) if rs else None,
            "refusal_rate": round(sum(r.get("refusal", False) for r in rs) / len(rs), 3) if rs else None,
        }
    blocks = sorted({r["block"] for r in results})
    return {"all": agg(results), **{b: agg([r for r in results if r["block"] == b]) for b in blocks}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--model", default="local")
    ap.add_argument("--api-key", default=None, help="por defecto la variable de entorno API_KEY")
    ap.add_argument("--providers", default="", help="OpenRouter: lista de proveedores fijos (opcional)")
    ap.add_argument("--run", required=True, help="nombre de la ejecución, p.ej. base-q4")
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--out-dir", default="eval", help="carpeta de resultados (relativa al directorio actual)")
    ap.add_argument("--workers", type=int, default=8, help="peticiones en paralelo (llama-server -np)")
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--thinking", action="store_true", help="evaluar con enable_thinking=true")
    ap.add_argument("--rescore", action="store_true", help="recalcular el resumen de un run ya hecho, sin llamar al modelo")
    a = ap.parse_args()
    if a.api_key is None:
        a.api_key = os.environ.get("API_KEY", "none")

    rows = [json.loads(l) for l in open(a.test)]
    if a.limit:
        rows = rows[:a.limit]
    out_dir = Path(a.out_dir)
    out_dir.mkdir(exist_ok=True)
    pred_file = out_dir / f"{a.run}.jsonl"
    done = {}
    if pred_file.exists():
        for l in open(pred_file):
            d = json.loads(l)
            done[d["i"]] = d

    def work(i):
        row = rows[i]
        msgs = [{"role": "system", "content": row["messages"][0]["content"] + FORMAT_HINT},
                {"role": "user", "content": row["messages"][1]["content"]}]
        try:
            pred, cost = ask(a.base_url, a.model, msgs, a.max_tokens, a.api_key, a.thinking, a.providers)
        except Exception as e:
            return {"i": i, "error": str(e)}
        return {"i": i, "meta": row["meta"], "prediction": pred, "cost": cost, "score": score(row, pred)}

    # recalcular siempre la puntuación de lo ya guardado (las métricas pueden haber cambiado)
    for d in done.values():
        if "prediction" in d and d["i"] < len(rows):
            d["score"] = score(rows[d["i"]], d["prediction"])
    todo = [] if a.rescore else [i for i in range(len(rows)) if i not in done or "error" in done[i]]
    print(f"{len(rows)} ejemplos, {len(todo)} pendientes")
    with open(pred_file, "a") as f, ThreadPoolExecutor(a.workers) as ex:
        for n, res in enumerate(ex.map(work, todo), 1):
            done[res["i"]] = res
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            if n % 10 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
    results = [d["score"] for d in done.values() if "score" in d]
    errors = sum("error" in d for d in done.values())
    cost = sum(d.get("cost", 0) for d in done.values())
    summary = {"run": a.run, "errors": errors, **({"cost_usd": round(cost, 3)} if cost else {}), **summarize(results)}
    (out_dir / f"{a.run}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
