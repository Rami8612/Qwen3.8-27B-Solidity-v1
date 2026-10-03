#!/usr/bin/env bash
# Cuantizaciones del modelo entrenado + comparación por dominio.
# Se ejecuta en la H100 DESPUÉS del entrenamiento (necesita out/merged o un GGUF BF16 en out/gguf).
#
#   cd /workspace/bundle && nohup bash quantize.sh > quantize.log 2>&1 &
#   CLEAN=1 bash quantize.sh   # además borra la copia duplicada out/gguf/*.safetensors y la caché del modelo base (libera ~105 GB)
#
# Genera en q/:
#   A  Q4_K_M sin imatrix          (el que exportó Unsloth)
#   B  Q4_K_M + imatrix            (misma talla, calibrado con nuestros dominios)
#   C  MIX: base Q3_K_M + imatrix, subiendo a Q5/Q6 los tensores más sensibles (ver MIX_RULES)
#   D  Q5_K_M + imatrix
#   E  Q6_K + imatrix              (alta calidad)
#   F  Q8_0                        (casi sin pérdida: el techo de la comparación)
#   G… mezclas propias guiadas por la imatrix (paso 4b), se añaden a q/ con la letra G, H...
# y para cada una mide, en 6 dominios (solidity, foundry, blockchain, python, code, es), la
# divergencia KL frente al modelo BF16 y la perplejidad -> q/summary.md
set -euo pipefail
cd "$(dirname "$0")"
LC=./llama.cpp
B=$LC/build/bin
mkdir -p q q/kld

# Tensores que se protegen en la versión MIX. Se pueden ajustar tras ver q/imatrix_stats.txt.
MIX_BASE=Q3_K_M
MIX_RULES=(--output-tensor-type q6_k --token-embedding-type q5_k
           --tensor-type "attn_v=q6_k" --tensor-type "ffn_down=q5_k")

step() { echo; echo "=== $(date +%H:%M:%S) $* ==="; df -h /workspace | tail -1; }

step "1. llama.cpp (quantize, imatrix, perplexity, server)"
[ -d $LC ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp $LC
which cmake >/dev/null || (apt-get update && apt-get install -y cmake build-essential)
cmake -S $LC -B $LC/build -DGGML_CUDA=ON >/dev/null
cmake --build $LC/build -j 12 --target llama-quantize llama-imatrix llama-perplexity llama-server

step "2. GGUF BF16 (fuente de todas las versiones)"
BF16=$(ls out/gguf_gguf/*[Bb][Ff]16*.gguf out/gguf/*[Bb][Ff]16*.gguf 2>/dev/null | grep -v mmproj | head -1 || true)
if [ -z "$BF16" ]; then
  BF16=q/model-BF16.gguf
  if [ ! -f $BF16 ]; then
    # gguf-py del propio llama.cpp, sin tocar torch/transformers del entorno de entrenamiento
    pip install -q --no-deps -e $LC/gguf-py
    pip install -q sentencepiece protobuf
    python3 $LC/convert_hf_to_gguf.py out/merged --outtype bf16 --outfile $BF16
  fi
fi
echo "BF16: $BF16 ($(du -h "$BF16" | cut -f1))"
if [ "${CLEAN:-0}" = 1 ]; then
  echo "CLEAN=1: borrando la copia duplicada en out/gguf y la caché del modelo base (out/merged y out/lora se conservan)"
  rm -rf out/gguf/*.safetensors /workspace/.hf_home/hub/models--unsloth--Qwen3.8-27B /workspace/.hf_home/hub/blobs  # blobs: descargador rápido de Unsloth
fi

step "3. imatrix con la calibración mixta"
if [ ! -f q/imatrix.gguf ]; then
  $B/llama-imatrix -m "$BF16" -f quant_data/calib.txt -o q/imatrix.gguf -ngl 99 -c 512 -b 512
fi
$B/llama-imatrix --in-file q/imatrix.gguf --show-statistics > q/imatrix_stats.txt 2>&1 || true
echo "estadísticas de importancia por tensor en q/imatrix_stats.txt"

step "4. cuantizaciones"
A=$(ls out/gguf_gguf/*Q4_K_M*.gguf out/gguf/*Q4_K_M*.gguf 2>/dev/null | head -1 || true)
[ -n "$A" ] && ln -sf "$(realpath "$A")" q/A-Q4_K_M.gguf
quant() { [ -f "$2" ] || $B/llama-quantize "${@:3}" "$BF16" "$2" "$1"; }
[ -f q/A-Q4_K_M.gguf ] || quant Q4_K_M q/A-Q4_K_M.gguf
quant Q4_K_M  q/B-Q4_K_M-imat.gguf --imatrix q/imatrix.gguf
quant $MIX_BASE q/C-MIX.gguf       --imatrix q/imatrix.gguf "${MIX_RULES[@]}"
quant Q5_K_M  q/D-Q5_K_M-imat.gguf --imatrix q/imatrix.gguf
quant Q6_K    q/E-Q6_K-imat.gguf   --imatrix q/imatrix.gguf
quant Q8_0    q/F-Q8_0.gguf                                  # casi sin pérdida; techo de la comparación
ls -laL q/*.gguf

step "5. referencia BF16 por dominio (logits para la KL)"
DOMS=(solidity foundry blockchain python code es)
for d in "${DOMS[@]}"; do
  [ -f q/kld/$d.kld ] || $B/llama-perplexity -m "$BF16" -f quant_data/eval/$d.txt -c 512 -ngl 99 \
      --kl-divergence-base q/kld/$d.kld > q/kld/$d.base.log 2>&1
  grep -o "Final estimate: PPL = [0-9.]*" q/kld/$d.base.log | sed "s/^/  $d BF16 /" || true
done

step "6. comparar cada versión con BF16"
for m in q/[A-Z]-*.gguf; do   # A-E + cualquier mezcla propia (F-..., G-...)
  n=$(basename "$m" .gguf)
  for d in "${DOMS[@]}"; do
    [ -f q/kld/$n.$d.log ] || $B/llama-perplexity -m "$m" -f quant_data/eval/$d.txt -c 512 -ngl 99 \
        --kl-divergence-base q/kld/$d.kld --kl-divergence > q/kld/$n.$d.log 2>&1
  done
done

step "7. resumen"
python3 - <<'EOF'
import glob, os, re
doms = ["solidity", "foundry", "blockchain", "python", "code", "es"]
def grab(path, pat):
    try:
        m = re.search(pat, open(path, errors="replace").read())
        return float(m.group(1)) if m else None
    except FileNotFoundError:
        return None
rows = []
for m in sorted(glob.glob("q/[A-Z]-*.gguf")):
    n = os.path.basename(m)[:-5]
    gb = os.path.getsize(os.path.realpath(m)) / 1e9
    kld = [grab(f"q/kld/{n}.{d}.log", r"Mean\s+KLD:\s+([0-9.]+)") for d in doms]
    top = [grab(f"q/kld/{n}.{d}.log", r"Same top p:\s+([0-9.]+)") for d in doms]
    rows.append((n, gb, kld, top))
f = lambda v, fmt: (fmt % v) if v is not None else "?"
out = ["# Comparación de cuantizaciones (frente a BF16; KLD menor = más fiel)", "",
       "| versión | GB | " + " | ".join(f"KLD {d}" for d in doms) + " | KLD media | mismo top-1 medio |",
       "|---|---|" + "---|" * (len(doms) + 2)]
for n, gb, kld, top in rows:
    k = [x for x in kld if x is not None]; t = [x for x in top if x is not None]
    out.append(f"| {n} | {gb:.1f} | " + " | ".join(f(x, "%.4f") for x in kld)
               + f" | {f(sum(k)/len(k) if k else None, '%.4f')} | {f(sum(t)/len(t) if t else None, '%.1f')}% |")
open("q/summary.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
EOF
step "fin"
