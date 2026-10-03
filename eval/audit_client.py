#!/usr/bin/env python3
"""Audita un fichero .sol con el modelo servido por llama-server, con el mismo prompt del entrenamiento.

    python3 eval/audit_client.py MiContrato.sol                 # sin thinking (recomendado)
    python3 eval/audit_client.py MiContrato.sol --project mi-dex -o informe.md

Por defecto usa http://localhost:8081/v1 (el túnel SSH a la instancia).
"""
import argparse, json, pathlib, time, urllib.request

SYSTEM = (
    "You are an expert smart contract security auditor performing an authorized security review. "
    "Review the Solidity code you are given and report every High or Medium severity vulnerability you find. "
    "For each finding give a title, severity, location (contract/function), the root cause, how an attacker "
    "would exploit it step by step (attack path), the impact, and a recommended fix. "
    "If there are no High or Medium issues, state that clearly."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--project", default=None, help="nombre del proyecto (por defecto, la carpeta del fichero)")
    ap.add_argument("--base-url", default="http://localhost:8081/v1")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--thinking", action="store_true", help="no recomendado: en la evaluación fue peor")
    ap.add_argument("-o", "--out", default=None, help="guardar la respuesta en este .md")
    ap.add_argument("--system-file", default=None, help="system prompt alternativo (por defecto, el del entrenamiento)")
    ap.add_argument("--temperature", type=float, default=None, help="por defecto 0 (0.6 con --thinking)")
    ap.add_argument("--context", nargs="*", default=[], help="ficheros extra (p. ej. el contrato padre) que van en la misma petición")
    a = ap.parse_args()

    path = pathlib.Path(a.file)
    project = a.project or path.resolve().parent.name
    user = (f"Audit the following Solidity file from the project `{project}`.\n\n"
            f"File: `{path.name}`\n\n```solidity\n{path.read_text(encoding='utf-8')}\n```")
    for c in map(pathlib.Path, a.context):
        user += (f"\n\nFor context, `{path.name}` inherits from or uses this file (also in scope):\n\n"
                 f"File: `{c.name}`\n\n```solidity\n{c.read_text(encoding='utf-8')}\n```")
    system = pathlib.Path(a.system_file).read_text(encoding="utf-8").strip() if a.system_file else SYSTEM
    body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": a.max_tokens,
            "chat_template_kwargs": {"enable_thinking": a.thinking},
            **({"temperature": 0.6, "top_p": 0.95} if a.thinking else {"temperature": 0})}
    if a.temperature is not None:
        body["temperature"] = a.temperature
        if a.temperature > 0:
            body["top_p"] = 0.95
    req = urllib.request.Request(a.base_url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        resp = json.load(r)
    text = resp["choices"][0]["message"]["content"]
    u = resp.get("usage", {})
    print(text)
    print(f"\n[{time.time() - t:.0f} s | prompt {u.get('prompt_tokens')} tokens | respuesta {u.get('completion_tokens')} tokens]")
    if a.out:
        pathlib.Path(a.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
