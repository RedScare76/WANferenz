# WANferenz

[WANferenz.de](https://wanferenz.de)

# Übersicht

## Die Problemstellung

Je größer ein Sprachmodell ist, desto mehr Speicher und Rechenleistung benötigt es. Für sehr große Modelle reicht die lokal verfügbare Grafikkarte deshalb häufig nicht aus. 

Eine Möglichkeit besteht darin, das Modell auf mehrere Rechner mit Grafikkarten zu verteilen. So lassen sich deren Speicher und Rechenleistung gemeinsam nutzen. Stehen diese Rechner an unterschiedlichen Orten und sind über das Internet verbunden, entsteht jedoch ein weiteres Problem: Netzwerklatenz. Jede Übertragung braucht Zeit. Da Sprachmodelle ihre Ausgabe schrittweise in Tokens erzeugen, können sich zusätzliche Wartezeiten bei jedem Generierungsschritt wiederholen und die gesamte Ausgabe deutlich verlangsamen.

## Lösungsansatz WANferenz

Deshalb habe ich **WANferenz** entwickelt, eine Inferenz-Engine, die speziell auf Ausführung über mehrere Rechner bei hoher Netzwerklatenz ausgerichtet ist. AMein Ziel ist es, große Sprachmodelle auch dann praktisch nutzbar zu machen, wenn keine einzelne ausreichend große GPU zur Verfügung steht und die beteiligten Rechner nicht über ein schnelles lokales Netzwerk verbunden sind. Das mrderne Open-Weight Modell DeepSeek-V4-Flash bot sich an, da es über Architekturmerkmale verfügt, die eine Verteilung über mehrere Knoten erleichtern.

Bei einer zentral gesteuerten RPC-Verteilung, wie sie [llama.cpp](https://llama.app/) unterstützt, läuft die Kommunikation mit entfernten Rechenknoten über ein zentrales Backend. Zusätzliche Hin- und Rückwege fallen in einem lokalen Netzwerk mit sehr niedriger Latenz wenig ins Gewicht. Bei einer Verbindung mit hoher Latenz können sie dagegen einen wesentlichen Teil der Laufzeit ausmachen. WANferenz setzt deshalb bei der Frage an: Wie lässt sich die notwendige Kommunikation zwischen den Rechnern verringern?

Dazu organisiert WANferenz die Rechenknoten als direkte Kette. Jeder Knoten berechnet einen zusammenhängenden Abschnitt der Modellschichten und gibt die Zwischenergebnisse an den nächsten Knoten weiter. Die Attention-Zustände, mit denen das Modell den bisherigen Kontext verarbeitet, bleiben auf dem jeweils zuständigen Rechner. Die Auswahl der nächsten Ausgabetokens findet direkt auf dem letzten Knoten statt. Ein Koordinator nimmt weiterhin Anfragen entgegen und empfängt die Ergebnisse; er muss jedoch nicht jede einzelne Rechenoperation auf den entfernten GPUs steuern.

Bei der Verteilung hilft die Spezialisierung auf DeepSeek-V4-Flash: Der spekulative Decoder DSpark schlägt kommende Tokens vor, die anschließend durch das Modell überprüft werden. Dafür nutzt er Zwischenrepräsentationen der letzten drei Hauptmodellschichten. WANferenz legt diese Schichten zusammen mit dem Decoder auf den letzten Knoten. So stehen die benötigten Informationen dort lokal bereit. Die Modi _WANferenz greedy_ und _WANferenz DSpark_ werden auf identische Ausgabetokens geprüft.

## Was die Messungen zeigen

Die gespeicherten Experimente untersuchen diesen Ansatz auf zwei DGX Sparks. Zusätzliche, kontrolliert eingestellte Netzwerklatenz bildet dabei eine Eigenschaft von Internetverbindungen nach. Bei einem Zielwert von 95 ms Anwendungs-Roundtrip-Zeit erreichte WANferenz bis zu 60 % mehr duchsatz als llama.cpp. Dieser Vergleich bezieht sich auf die gesamte gemessene Generierung einschließlich Prompt-Verarbeitung; die reine Decode-Geschwindigkeit wird separat ausgewiesen. Versuchsbedingungen, Einzelmessungen und weitere Latenzwerte stehen im [Benchmarkbericht](benchmark-results/).

Dieses Repository enthält die Engine, den Benchmark-Runner, eine feste Benchmark-Eingabe und gespeicherte Messergebnisse. Die folgende Anleitung beschreibt die Einrichtung und Ausführung von _WANferenz greedy_ und _WANferenz DSpark_ auf zwei DGX Sparks. Die Modellgewichte werden separat heruntergeladen.

# Durchführung Benchmark

## 1. Prepare both Sparks

Use two NVIDIA DGX Sparks with 128 GB unified memory each, Ubuntu 24.04/AArch64, working NVIDIA drivers, and their regular Ethernet connection. The recorded GPU environment used driver 580.173.02, PyTorch 2.11.0+cu130, CUDA 13.0 and TileLang 0.1.9. The container below supplies the Python/CUDA runtime; the host still needs a compatible driver.

These examples use Spark 1 at `192.168.10.152` and Spark 2 at `192.168.10.154`, connected through `enP7s7`. Use the LAN addresses, not QSFP. Allow SSH and TCP port 29610 from Spark 1 to Spark 2. Reserve roughly 320 GB of free disk space on Spark 1 for source and converted weights, and 160 GB on Spark 2 for converted weights, plus space for containers and caches.

On **both Sparks**:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip openssh-client openssh-server rsync iputils-ping socat
sudo systemctl enable --now ssh
nvidia-smi
```

Install [Docker Engine for Ubuntu](https://docs.docker.com/engine/install/ubuntu/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo usermod -aG docker "$USER"
```

Log out and back in after changing group membership. The benchmark launches Docker locally and over SSH, so your account must be able to run `docker ps` without `sudo` on both machines.

On both Sparks, pull the recorded runtime and assign the tag expected by the runner:

```bash
export WANFERENZ_IMAGE=ghcr.io/anemll/dspark-vllm-gx10:0.1.1
docker pull ghcr.io/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8
docker tag ghcr.io/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8 "$WANFERENZ_IMAGE"
docker run --rm --gpus all --entrypoint python3 "$WANFERENZ_IMAGE" \
  -c 'import torch, tilelang; print(torch.__version__, torch.version.cuda, tilelang.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
```

The runner uses `--pull never`; prepare the image on both machines before starting it.

## 2. Set up the source and virtual environment

Place this repository at `~/wanferenz` on Spark 1.  

```bash
cd ~/wanferenz
export WANFERENZ_REPO="$PWD"
export WANFERENZ_MODEL="$HOME/v4-wanferenz"
export WANFERENZ_SOURCE="$HOME/v4-source"
export WANFERENZ_REMOTE=192.168.10.154
export WANFERENZ_SSH_USER="$USER"
export WANFERENZ_KEY="$HOME/.ssh/id_ed25519"
export WANFERENZ_IMAGE=ghcr.io/anemll/dspark-vllm-gx10:0.1.1
```
Create the **host virtual environment on Spark 1**:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . huggingface_hub
wanferenz-bench --help
```

The host environment runs the benchmark controller and optional CPU checks. GPU inference runs inside the prepared container. Spark 2 needs the source and container, but no host Python virtual environment.

## 3. Download and convert the model

Download the recorded [DeepSeek-V4-Flash-0731 checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) on Spark 1 using the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```bash
hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --revision 9e165c30e2704aec5d9d593cce3eebd58bbef1cb \
  --include '*.safetensors' 'tokenizer.json' 'tokenizer_config.json' 'inference/config.json' \
  --local-dir "$WANFERENZ_SOURCE"
mkdir -p "$WANFERENZ_MODEL"
docker run --rm --user "$(id -u):$(id -g)" --entrypoint python3 \
  -e PYTHONPATH=/workspace/wanferenz \
  -v "$WANFERENZ_REPO:/workspace/wanferenz:ro" \
  -v "$WANFERENZ_SOURCE:/source:ro" \
  -v "$WANFERENZ_MODEL:/model" \
  -w /workspace/wanferenz "$WANFERENZ_IMAGE" \
  -m wanferenz.model.conversion \
  --hf-ckpt-path /source --save-path /model --expert-dtype fp4
```

Use an empty output directory. The converter refuses existing converted shards. It processes source shards separately, preserves FP4 experts and produces `model00000-mp1.safetensors`, subsequent shards, `config.json` and tokenizer files. The recorded checkpoint produces 48 shards totaling approximately 157 GB. Both Sparks need the complete converted directory, including all three MTP/DSpark blocks.

Copy source and converted weights to Spark 2. These commands assume paths without spaces:

```bash
ssh -i "$WANFERENZ_KEY" "$WANFERENZ_SSH_USER@$WANFERENZ_REMOTE" \
  "mkdir -p '$WANFERENZ_REPO' '$WANFERENZ_MODEL'"
rsync -a -e "ssh -i $WANFERENZ_KEY" \
  --exclude .git --exclude .venv --exclude __pycache__ --exclude '*.egg-info' \
  --exclude .pytest_cache --exclude .agents --exclude .codex --exclude benchmark-results \
  "$WANFERENZ_REPO/" "$WANFERENZ_SSH_USER@$WANFERENZ_REMOTE:$WANFERENZ_REPO/"
rsync -a --info=progress2 -e "ssh -i $WANFERENZ_KEY" \
  "$WANFERENZ_MODEL/" "$WANFERENZ_SSH_USER@$WANFERENZ_REMOTE:$WANFERENZ_MODEL/"
```

## 4. Run the benchmark

From the repository on Spark 1, with `.venv` active:

```bash
export WANFERENZ_RUN="$WANFERENZ_REPO/benchmark-results/lan-$(date -u +%Y%m%dT%H%M%SZ)"
wanferenz-bench \
  --repo "$WANFERENZ_REPO" --model "$WANFERENZ_MODEL" \
  --remote "$WANFERENZ_REMOTE" --user "$WANFERENZ_SSH_USER" --key "$WANFERENZ_KEY" \
  --profile moe-graph --dspark --dspark-moe --dspark-graph --dspark-full-graph \
  --lazy-draft --draft-width 1 --spec-depth 2 --refill-floor 1 \
  --split 23 --snapshot-mode full \
  --workload benchmark-workloads/v4-flash-128.json \
  --output "$WANFERENZ_RUN"
```

The runner starts the tail on Spark 2 and the head on Spark 1, loads layers 0–22 / 23–42, and waits for readiness. First startup includes kernel compilation and graph capture. It runs one full 128-token warmup per mode, then four measured repetitions of each mode in alternating order. The hashed workload fixes 26 prompt tokens, 128 output tokens and a 512-position context. It also checks the checkpoint's tokenizer hash.

The process returns success only if the measured token streams match across WANferenz greedy and WANferenz DSpark, requested lengths and timings are present, and the complete drafter graph captured. Internal mode names in JSON are `greedy` and `spec`. For subsequent runs with the same workload, add `--reference /absolute/path/to/prior/summary.json` to also check against a saved token stream. Use a fresh output directory for every run.

The runner stops its two stage containers when it finishes or encounters an ordinary error. It leaves stopped containers available for inspection. Their exact names are recorded in `config.json`; after saving the logs, remove only those named containers with `docker rm` on their respective hosts. Following a forced termination or lost connection, inspect those names on both machines and stop any remaining run-owned containers before retrying.

## 5. Read the results

The compact competition results are in [benchmark-results/0-overview/](benchmark-results/0-overview/), alongside the consolidated CSVs, workload and graphs. New benchmark runs go into separate dated directories, which are ignored by Git.

Each run saves:

| File | Contents |
| --- | --- |
| `summary.json` | Success/parity flags, every accepted job, token IDs, per-mode medians and detailed timings |
| `config.json` | Model path, container names, layer split, controls and complete workload |
| `jobs.jsonl` | Submitted jobs, including warmups |
| `coordinator.log` | Generation events, timing records and any rejected latency attempts |
| `stage0.log`, `stage1.log` | Worker startup, graph capture and runtime logs |
| `network-rtt.txt` | ICMP probe before loading |


## 6. Optional: measure higher network latency

The runner can use [Toxiproxy](https://github.com/Shopify/toxiproxy) to add delay without changing interface qdiscs. This measures application-echo RTT through equally configured proxies, not physical WAN or ICMP RTT. Use the same workload and engine options as the LAN run.

On **Spark 2**, in a separate terminal, start an echo helper and leave it running:

```bash
socat TCP-LISTEN:29630,bind=192.168.10.154,reuseaddr,fork,nodelay EXEC:/bin/cat
```

Allow Spark 1 to connect to TCP 29630. On **Spark 1**, check that ports 8474 and 29621–29623 are unused (`ss -ltn`), then start a dedicated proxy instance:

```bash
export WANFERENZ_PROXY="wanferenz-latency-$(date -u +%Y%m%dT%H%M%SZ)"
docker pull ghcr.io/shopify/toxiproxy:latest
docker run -d --rm --name "$WANFERENZ_PROXY" --network host \
  ghcr.io/shopify/toxiproxy:latest -host 127.0.0.1
export WANFERENZ_SWEEP="$WANFERENZ_REPO/benchmark-results/latency-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$WANFERENZ_SWEEP"
docker inspect "$WANFERENZ_PROXY" > "$WANFERENZ_SWEEP/proxy-container.json"
```

Create three proxies and a sweep plan in that fresh directory:

```bash
python - <<'PY'
import json, os
from pathlib import Path
from wanferenz.benchmark.latency import api
base = 'http://127.0.0.1:8474'
remote = os.environ['WANFERENZ_REMOTE']
names = ['wanferenz-forward', 'wanferenz-return', 'wanferenz-echo']
for name, local, upstream in zip(names, [29621, 29622, 29623], [29610, 29610, 29630]):
    api(base, '/proxies', {'name': name, 'listen': f'127.0.0.1:{local}',
                         'upstream': f'{remote}:{upstream}', 'enabled': True})
    for stream in ['upstream', 'downstream']:
        api(base, f'/proxies/{name}/toxics', {'name': stream, 'type': 'latency',
            'stream': stream, 'toxicity': 1, 'attributes': {'latency': 0, 'jitter': 0}})
plan = {'api': base, 'proxyNames': names, 'echoAddress': '127.0.0.1:29623',
        'forwardAddress': '127.0.0.1:29621', 'returnAddress': '127.0.0.1:29622',
        'targetsMs': [5, 20, 40, 80, 100], 'toleranceMs': 1.5}
path = Path(os.environ['WANFERENZ_SWEEP']) / 'sweep-plan.json'
with path.open('x') as output:
    json.dump(plan, output, indent=2)
PY
wanferenz-bench \
  --repo "$WANFERENZ_REPO" --model "$WANFERENZ_MODEL" \
  --remote "$WANFERENZ_REMOTE" --user "$WANFERENZ_SSH_USER" --key "$WANFERENZ_KEY" \
  --profile moe-graph --dspark --dspark-moe --dspark-graph --dspark-full-graph \
  --lazy-draft --draft-width 1 --spec-depth 2 --refill-floor 1 \
  --split 23 --snapshot-mode full \
  --workload benchmark-workloads/v4-flash-128.json \
  --sweep-plan "$WANFERENZ_SWEEP/sweep-plan.json" \
  --output "$WANFERENZ_SWEEP/run"
```

Each target gets full warmups and four interleaved repetitions. Delay is split between directions, calibrated against the echo helper and checked before and after each job within ±1.5 ms. Drifted jobs retain their rejected record and retry up to five times; persistent drift fails the run. Choose targets above your actual baseline RTT. The summary separates modes by target, such as `greedy-r040` and `spec-r040`.

After the run, including a failed run, save proxy logs and stop this run's proxy container:

```bash
docker logs "$WANFERENZ_PROXY" > "$WANFERENZ_SWEEP/proxy.log" 2>&1
docker stop "$WANFERENZ_PROXY"
```

Stop the echo helper with Ctrl-C in its terminal on Spark 2. Preserve all measured and rejected records. Reuse the same proxy image digest for matched experiments; `proxy-container.json` records the image used.

## Correctness checks and source

The main hardware validation is the benchmark in section 4: it loads the real checkpoint on both Sparks and compares generated tokens across WANferenz greedy and WANferenz DSpark. Use `--reference` to compare later runs against a saved result from the same workload.

For a quick development check, install the test extra in the host virtual environment and run the small smoke/parity suite:

```bash
python -m pip install -e '.[test]'
OMP_NUM_THREADS=1 python -m pytest -q tests/
```

Use PyTorch 2.11 for the CPU baseline and allow loopback sockets. These checks cover packaging, workload identity, benchmark accounting and token parity across miniature CPU chains, including speculative rejection and resets. They do not validate GB10 kernels, CUDA graphs, full-checkpoint correctness or performance on the Sparks.

The `wanferenz/` package separates serving, protocol, model loading/conversion, kernels, decoding, CUDA graphs and benchmark measurement. The fixed input is in `benchmark-workloads/`. For lower-level stage/coordinator commands, run `wanferenz --help`.


