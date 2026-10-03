#!/usr/bin/env python3
"""Prueba de regresión general: ¿el entrenamiento de auditor ha estropeado otras habilidades?

Se pasa igual al modelo base y al entrenado (servidor compatible con OpenAI: llama-server, vLLM...):
    python3 13_general_eval.py --run base-q4
    python3 13_general_eval.py --run ft-q4
    python3 13_general_eval.py --selftest          # comprueba que los tests aceptan las soluciones de referencia

- Python (16): el modelo escribe una función y se ejecutan tests sobre ella (pasa / no pasa).
- Español (16): gramática, cultura general, comprensión, traducción y razonamiento; se puntúa por
  palabras clave y además se mide si responde en español.
System prompt genérico (no el de auditor) y sin thinking. Salida: eval/general_<run>.jsonl y _summary.json
"""
import argparse, json, re, subprocess, sys, tempfile, unicodedata, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SYSTEM = "Eres un asistente útil. Responde siempre en español, salvo el código."

# ----------------------------------------------------------------------------- Python
PY = [
 ("is_palindrome", "Escribe una función `is_palindrome(s: str) -> bool` que diga si un texto es palíndromo ignorando mayúsculas, espacios y signos de puntuación.",
  "assert is_palindrome('Anita lava la tina')\nassert is_palindrome('A man, a plan, a canal: Panama')\nassert not is_palindrome('hola')\nassert is_palindrome('')",
  "def is_palindrome(s):\n    t=[c.lower() for c in s if c.isalnum()]\n    return t==t[::-1]"),
 ("fizzbuzz", "Escribe `fizzbuzz(n: int) -> list[str]` que devuelva la lista del 1 al n donde los múltiplos de 3 son 'Fizz', los de 5 'Buzz', los de ambos 'FizzBuzz' y el resto el número como texto.",
  "assert fizzbuzz(5)==['1','2','Fizz','4','Buzz']\nassert fizzbuzz(15)[-1]=='FizzBuzz'\nassert fizzbuzz(0)==[]",
  "def fizzbuzz(n):\n    return ['FizzBuzz' if i%15==0 else 'Fizz' if i%3==0 else 'Buzz' if i%5==0 else str(i) for i in range(1,n+1)]"),
 ("merge_intervals", "Escribe `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` que fusione intervalos solapados (cerrados) y devuelva el resultado ordenado por inicio.",
  "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,4],[4,5]])==[[1,5]]\nassert merge_intervals([])==[]\nassert merge_intervals([[5,7],[1,2]])==[[1,2],[5,7]]",
  "def merge_intervals(iv):\n    out=[]\n    for a,b in sorted(iv):\n        if out and a<=out[-1][1]: out[-1][1]=max(out[-1][1],b)\n        else: out.append([a,b])\n    return out"),
 ("flatten", "Escribe `flatten(x: list) -> list` que aplane una lista con listas anidadas a cualquier profundidad, conservando el orden.",
  "assert flatten([1,[2,[3,[4]],5]])==[1,2,3,4,5]\nassert flatten([])==[]\nassert flatten([[[]]])==[]\nassert flatten(['a',['b']])==['a','b']",
  "def flatten(x):\n    out=[]\n    for e in x:\n        out.extend(flatten(e)) if isinstance(e,list) else out.append(e)\n    return out"),
 ("word_count", "Escribe `word_count(text: str) -> dict[str, int]` que cuente cuántas veces aparece cada palabra, en minúsculas e ignorando signos de puntuación. Las palabras pueden llevar tildes y ñ.",
  "assert word_count('Hola, hola mundo.')=={'hola':2,'mundo':1}\nassert word_count('')=={}\nassert word_count('Año año AÑO!')=={'año':3}",
  "import re\ndef word_count(text):\n    d={}\n    for w in re.findall(r'\\w+', text.lower()):\n        d[w]=d.get(w,0)+1\n    return d"),
 ("rotate_matrix", "Escribe `rotate_matrix(m: list[list[int]]) -> list[list[int]]` que devuelva la matriz rotada 90 grados en sentido horario (sin modificar la original).",
  "assert rotate_matrix([[1,2],[3,4]])==[[3,1],[4,2]]\nassert rotate_matrix([[1,2,3]])==[[1],[2],[3]]\nm=[[1,2],[3,4]]; rotate_matrix(m); assert m==[[1,2],[3,4]]",
  "def rotate_matrix(m):\n    return [list(r) for r in zip(*m[::-1])]"),
 ("roman_to_int", "Escribe `roman_to_int(s: str) -> int` que convierta un número romano válido (I, V, X, L, C, D, M) a entero.",
  "assert roman_to_int('III')==3\nassert roman_to_int('IV')==4\nassert roman_to_int('MCMXCIV')==1994\nassert roman_to_int('LVIII')==58",
  "def roman_to_int(s):\n    v={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n    t=0\n    for i,c in enumerate(s):\n        t+= -v[c] if i+1<len(s) and v[c]<v[s[i+1]] else v[c]\n    return t"),
 ("top_k_frequent", "Escribe `top_k_frequent(nums: list[int], k: int) -> list[int]` que devuelva los k números más frecuentes, ordenados por frecuencia descendente y, en caso de empate, por valor ascendente.",
  "assert top_k_frequent([1,1,1,2,2,3],2)==[1,2]\nassert top_k_frequent([4,4,5,5,6],2)==[4,5]\nassert top_k_frequent([7],1)==[7]",
  "from collections import Counter\ndef top_k_frequent(nums,k):\n    c=Counter(nums)\n    return sorted(c,key=lambda x:(-c[x],x))[:k]"),
 ("binary_search", "Escribe `binary_search(arr: list[int], x: int) -> int` que busque x en una lista ordenada y devuelva su índice, o -1 si no está. Debe ser O(log n).",
  "assert binary_search([1,3,5,7,9],7)==3\nassert binary_search([1,3,5],4)==-1\nassert binary_search([],1)==-1\nassert binary_search([2],2)==0",
  "def binary_search(a,x):\n    lo,hi=0,len(a)-1\n    while lo<=hi:\n        m=(lo+hi)//2\n        if a[m]==x: return m\n        lo,hi=(m+1,hi) if a[m]<x else (lo,m-1)\n    return -1"),
 ("valid_brackets", "Escribe `valid_brackets(s: str) -> bool` que compruebe si los paréntesis (), corchetes [] y llaves {} de un texto están bien balanceados y anidados. Ignora los demás caracteres.",
  "assert valid_brackets('a(b[c]{d})')\nassert not valid_brackets('(]')\nassert not valid_brackets('((')\nassert valid_brackets('')\nassert not valid_brackets(')(')",
  "def valid_brackets(s):\n    st=[];p={')':'(',']':'[','}':'{'}\n    for c in s:\n        if c in '([{': st.append(c)\n        elif c in p:\n            if not st or st.pop()!=p[c]: return False\n    return not st"),
 ("LRUCache", "Escribe una clase `LRUCache` con `__init__(self, capacity: int)`, `get(self, key) -> int` (devuelve -1 si no existe) y `put(self, key, value)`. Al superar la capacidad se elimina el elemento usado hace más tiempo; tanto get como put cuentan como uso.",
  "c=LRUCache(2); c.put(1,1); c.put(2,2); assert c.get(1)==1; c.put(3,3); assert c.get(2)==-1; c.put(4,4); assert c.get(1)==-1; assert c.get(3)==3; assert c.get(4)==4",
  "from collections import OrderedDict\nclass LRUCache:\n    def __init__(self,capacity): self.c=capacity; self.d=OrderedDict()\n    def get(self,k):\n        if k not in self.d: return -1\n        self.d.move_to_end(k); return self.d[k]\n    def put(self,k,v):\n        self.d[k]=v; self.d.move_to_end(k)\n        if len(self.d)>self.c: self.d.popitem(last=False)"),
 ("chunk", "Escribe `chunk(lst: list, n: int) -> list[list]` que divida una lista en trozos consecutivos de tamaño n (el último puede ser más corto).",
  "assert chunk([1,2,3,4,5],2)==[[1,2],[3,4],[5]]\nassert chunk([],3)==[]\nassert chunk([1,2],5)==[[1,2]]",
  "def chunk(lst,n):\n    return [lst[i:i+n] for i in range(0,len(lst),n)]"),
 ("to_snake_case", "Escribe `to_snake_case(s: str) -> str` que convierta un identificador camelCase o PascalCase a snake_case. Las siglas seguidas se tratan como una palabra (por ejemplo 'parseHTTPResponse' -> 'parse_http_response').",
  "assert to_snake_case('camelCase')=='camel_case'\nassert to_snake_case('PascalCase')=='pascal_case'\nassert to_snake_case('parseHTTPResponse')=='parse_http_response'\nassert to_snake_case('simple')=='simple'",
  "import re\ndef to_snake_case(s):\n    s=re.sub(r'([A-Z]+)([A-Z][a-z])', r'\\1_\\2', s)\n    s=re.sub(r'([a-z0-9])([A-Z])', r'\\1_\\2', s)\n    return s.lower()"),
 ("second_largest", "Escribe `second_largest(nums: list[int])` que devuelva el segundo valor distinto más grande, o None si no existe.",
  "assert second_largest([3,1,4,4,2])==3\nassert second_largest([5,5])is None\nassert second_largest([])is None\nassert second_largest([-1,-2])==-2",
  "def second_largest(nums):\n    s=sorted(set(nums),reverse=True)\n    return s[1] if len(s)>1 else None"),
 ("parse_kv", "Escribe `parse_kv(s: str) -> dict` que convierta un texto como 'a=1;b=hola;c=3' en un diccionario. Los valores que sean enteros se convierten a int; el resto se dejan como texto. Ignora los pares vacíos.",
  "assert parse_kv('a=1;b=hola;c=3')=={'a':1,'b':'hola','c':3}\nassert parse_kv('')=={}\nassert parse_kv('x=-5;;y=z')=={'x':-5,'y':'z'}",
  "def parse_kv(s):\n    d={}\n    for p in s.split(';'):\n        if not p: continue\n        k,v=p.split('=',1)\n        try: d[k]=int(v)\n        except ValueError: d[k]=v\n    return d"),
 ("format_units", "Escribe `format_units(value: int, decimals: int) -> str` que convierta una cantidad entera de un token (por ejemplo wei) a texto decimal exacto, sin ceros sobrantes al final ni usar float. Ejemplo: format_units(1500000000000000000, 18) == '1.5'.",
  "assert format_units(1500000000000000000,18)=='1.5'\nassert format_units(10**18,18)=='1'\nassert format_units(1,18)=='0.000000000000000001'\nassert format_units(0,18)=='0'\nassert format_units(123456,6)=='0.123456'\nassert format_units(-2500000,6)=='-2.5'",
  "def format_units(value,decimals):\n    sign='-' if value<0 else ''\n    q,r=divmod(abs(value),10**decimals)\n    frac=str(r).rjust(decimals,'0').rstrip('0') if decimals else ''\n    return sign+str(q)+('.'+frac if frac else '')"),
]
PY_SUFFIX = "\n\nResponde solo con el código Python completo en un bloque ```python, sin ejemplos de uso."

# ----------------------------------------------------------------------------- Español
# (pregunta, grupos de palabras clave: cada grupo es una lista de alternativas y deben aparecer todos)
ES = [
 ("¿Cuál es la capital de Perú? Responde con una sola palabra.", [["lima"]]),
 ("¿Quién escribió 'Cien años de soledad'?", [["garcia marquez", "gabo"]]),
 ("¿Cuántos lados tiene un hexágono? Responde con un número.", [["6", "seis"]]),
 ("Escribe el plural de la palabra 'lápiz'.", [["lapices"]]),
 ("Conjuga el verbo 'tener' en pretérito perfecto simple, primera persona del singular. Responde solo con la forma verbal.", [["tuve"]]),
 ("¿Cuál es el femenino de 'actor'?", [["actriz"]]),
 ("Corrige la ortografía de esta frase y escríbela bien: 'Haber si vienes mañana'.", [["a ver"]]),
 ("Si tengo 3 manzanas y compro el doble de las que tengo, ¿cuántas manzanas tengo en total? Responde con un número.", [["9", "nueve"]]),
 ("Lee el texto y responde: 'María salió de casa a las ocho. Tardó veinte minutos en llegar a la oficina.' ¿A qué hora llegó María a la oficina?", [["8:20", "ocho y veinte", "20:20"]]),
 ("Traduce al español: 'The contract is vulnerable to a reentrancy attack.'", [["contrato"], ["vulnerable"], ["reentrada", "reentrancia", "reentrancy"], ["ataque"]]),
 ("Responde únicamente 'sí' o 'no': ¿el agua hierve a 100 grados Celsius a nivel del mar?", [["si"]]),
 ("Explica en una o dos frases qué es una función hash.", [["hash"], ["entrada", "datos", "mensaje", "texto"], ["salida", "valor", "resumen", "huella", "longitud fija", "tamano fijo"]]),
 ("¿Qué palabra es un sinónimo de 'rápido': lento, veloz o pesado?", [["veloz"]]),
 ("Ordena alfabéticamente estas palabras y escríbelas separadas por comas: pera, manzana, uva, cereza.", [["cereza, manzana, pera, uva", "cereza,manzana,pera,uva"]]),
 ("¿En qué continente está Marruecos?", [["africa"]]),
 ("Escribe una frase en español que use correctamente la palabra 'sino'.", [["sino"]]),
]

ES_STOP = set("el la los las de que y en un una por con para no es se lo al del como pero más sus le ya o este esta son está muy también fue ha hay".split())
EN_STOP = set("the of and to in is that it for on with as are this be by was not or an have from at which you".split())


def norm(t):
    t = unicodedata.normalize("NFKD", t.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def spanish_ratio(text):
    words = re.findall(r"[a-záéíóúñü]+", text.lower())
    es = sum(w in ES_STOP for w in words); en = sum(w in EN_STOP for w in words)
    return 1.0 if es + en == 0 else es / (es + en)


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else text


def run_tests(code, tests, timeout=10):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + tests + "\nprint('OK')\n")
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0 and r.stdout.strip().endswith("OK"), (r.stderr or "")[-300:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        Path(path).unlink(missing_ok=True)


def ask(a, prompt):
    body = {"model": a.model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "max_tokens": a.max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": a.thinking}}
    req = urllib.request.Request(f"{a.base_url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        text = json.load(r)["choices"][0]["message"]["content"] or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).split("</think>")[-1].strip()


def selftest():
    bad = 0
    for name, _, tests, ref in PY:
        ok, err = run_tests(ref, tests)
        print(("OK  " if ok else "FALLA ") + name, "" if ok else err)
        bad += not ok
    print("python: referencias que no pasan sus tests:", bad)
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8081/v1")
    ap.add_argument("--model", default="local")
    ap.add_argument("--run")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default="eval")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.run:
        ap.error("--run es obligatorio")

    jobs = [("python", i) for i in range(len(PY))] + [("es", i) for i in range(len(ES))]

    def work(job):
        kind, i = job
        try:
            if kind == "python":
                name, prompt, tests, _ = PY[i]
                ans = ask(a, prompt + PY_SUFFIX)
                ok, err = run_tests(extract_code(ans), tests)
                return {"kind": kind, "id": name, "ok": ok, "spanish": None, "answer": ans, "error": err}
            prompt, groups = ES[i]
            ans = ask(a, prompt)
            n = norm(ans)
            ok = all(any(norm(alt) in n for alt in g) for g in groups)
            return {"kind": kind, "id": i, "ok": ok, "spanish": spanish_ratio(ans) >= 0.6, "answer": ans}
        except Exception as e:
            return {"kind": kind, "id": i, "ok": False, "spanish": None, "answer": "", "error": f"petición: {e}"}

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(work, jobs))
    out = Path(a.out_dir); out.mkdir(exist_ok=True)
    with open(out / f"general_{a.run}.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    py = [r for r in res if r["kind"] == "python"]; es = [r for r in res if r["kind"] == "es"]
    summary = {"run": a.run, "thinking": a.thinking,
               "python_pass": f"{sum(r['ok'] for r in py)}/{len(py)}",
               "es_correct": f"{sum(r['ok'] for r in es)}/{len(es)}",
               "es_answers_in_spanish": f"{sum(bool(r['spanish']) for r in es)}/{len(es)}",
               "python_failed": [r["id"] for r in py if not r["ok"]],
               "es_failed": [ES[r["id"]][0][:60] for r in es if not r["ok"] and isinstance(r["id"], int)]}
    (out / f"general_{a.run}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
