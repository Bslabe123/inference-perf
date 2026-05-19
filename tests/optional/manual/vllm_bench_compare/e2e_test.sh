#!/bin/bash
#
# Compares inference-perf vs `vllm bench serve` head-to-head against fresh,
# identically-configured vLLM deployments.
#
# For each case under cases/<name>/, two namespaces are created in parallel:
#   <prefix>-<case>-ip   inference-perf + its own vLLM
#   <prefix>-<case>-vb   vllm bench serve + its own vLLM
# Both vLLMs come up cold from the same vllm.yaml so neither tool benefits
# from warm KV caches or compilation artifacts left by a prior run. Once both
# servers are Ready, the two client Jobs run concurrently; results are
# scraped from pod logs, written to cases/<name>/output/, and a side-by-side
# comparison is rendered by render_comparison.py.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASES_DIR="$SCRIPT_DIR/cases"

NAMESPACE_PREFIX="vllm-bench-compare"
IP_IMAGE="quay.io/inference-perf/inference-perf:latest"
CLEANUP=false
ROLLOUT_TIMEOUT="30m"
JOB_TIMEOUT="60m"
ONLY_CASE=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [-n <namespace_prefix>] [-i <ip_image>] [-t <rollout_timeout>] [-j <job_timeout>] [-k <case>] [-c] [-h]

Runs every case under $CASES_DIR (or just one with -k). For each case, brings
up a fresh vLLM in two sibling namespaces and runs inference-perf in one and
\`vllm bench serve\` in the other against the same workload config.

  -n <prefix>             Namespace prefix (default: $NAMESPACE_PREFIX)
  -i <image>              inference-perf image (default: $IP_IMAGE)
  -t <duration>           vLLM rollout timeout per namespace (default: $ROLLOUT_TIMEOUT)
  -j <duration>           Client Job timeout (default: $JOB_TIMEOUT)
  -k <case>               Only run a single case by name
  -c                      Delete namespaces after the run completes
  -h                      Show this help
EOF
}

while getopts ":n:i:t:j:k:ch" opt; do
  case "$opt" in
    n) NAMESPACE_PREFIX="$OPTARG" ;;
    i) IP_IMAGE="$OPTARG" ;;
    t) ROLLOUT_TIMEOUT="$OPTARG" ;;
    j) JOB_TIMEOUT="$OPTARG" ;;
    k) ONLY_CASE="$OPTARG" ;;
    c) CLEANUP=true ;;
    h) usage; exit 0 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage; exit 2 ;;
    :)  echo "Option -$OPTARG requires an argument" >&2; usage; exit 2 ;;
  esac
done

if [ ! -d "$CASES_DIR" ]; then
  echo "Error: cases directory not found at $CASES_DIR" >&2
  exit 1
fi

IP_CONFIGMAP="inference-perf-config"
IP_JOB="inference-perf"
VB_JOB="vllm-bench"

# Locate inference-perf Job manifest template (same logic as the multimodal test).
TEMPLATE_MANIFEST="$SCRIPT_DIR/../../../../deploy/manifests.yaml"
if [ ! -f "$TEMPLATE_MANIFEST" ]; then
  TEMPLATE_MANIFEST="$SCRIPT_DIR/../../../deploy/manifests.yaml"
fi
if [ ! -f "$TEMPLATE_MANIFEST" ]; then
  echo "Error: deploy/manifests.yaml template not found" >&2
  exit 1
fi

copy_hf_secret() {
  local ns="$1"
  if kubectl get secret hf-secret -n default >/dev/null 2>&1; then
    kubectl get secret hf-secret -n default -o yaml \
      | grep -v -E "namespace:|resourceVersion:|uid:|creationTimestamp:" \
      | kubectl apply -n "$ns" -f -
  else
    echo "Warning: hf-secret not found in 'default' namespace; gated models will fail to load." >&2
  fi
}

ensure_namespace() {
  local ns="$1"
  kubectl create namespace "$ns" --dry-run=client -o yaml | kubectl apply -f -
  copy_hf_secret "$ns"
}

bring_up_vllm() {
  local ns="$1"
  local vllm_yaml="$2"
  kubectl apply -f "$vllm_yaml" -n "$ns"
  kubectl rollout status deployment/vllm-server -n "$ns" --timeout="$ROLLOUT_TIMEOUT"
}

run_inference_perf() {
  local ns="$1"
  local config_file="$2"

  kubectl create configmap "$IP_CONFIGMAP" \
    --from-file=config.yml="$config_file" \
    -n "$ns" --dry-run=client -o yaml | kubectl apply -f -

  kubectl delete job "$IP_JOB" -n "$ns" --ignore-not-found
  sed "s|quay.io/inference-perf/inference-perf:latest|$IP_IMAGE|g" "$TEMPLATE_MANIFEST" \
    | kubectl apply -f - -n "$ns"

  kubectl wait --for=condition=complete "job/$IP_JOB" -n "$ns" --timeout="$JOB_TIMEOUT"
}

run_vllm_bench() {
  local ns="$1"
  local job_yaml="$2"

  kubectl delete job "$VB_JOB" -n "$ns" --ignore-not-found
  kubectl apply -f "$job_yaml" -n "$ns"

  kubectl wait --for=condition=complete "job/$VB_JOB" -n "$ns" --timeout="$JOB_TIMEOUT"
}

extract_inference_perf_results() {
  local ns="$1"
  local out_dir="$2"
  local log_file="$out_dir/inference_perf_logs.txt"

  kubectl logs "job/$IP_JOB" -n "$ns" > "$log_file"
  awk '/=== START_SUMMARY ===/{flag=1;next}/=== END_SUMMARY ===/{flag=0}flag' \
    "$log_file" > "$out_dir/inference_perf.json"

  if [ ! -s "$out_dir/inference_perf.json" ]; then
    echo "Error: empty inference-perf summary extracted from $log_file" >&2
    return 1
  fi
  rm -f "$log_file"
}

extract_vllm_bench_results() {
  local ns="$1"
  local out_dir="$2"
  local log_file="$out_dir/vllm_bench_logs.txt"

  kubectl logs "job/$VB_JOB" -n "$ns" > "$log_file"
  awk '/=== START_VLLM_BENCH ===/{flag=1;next}/=== END_VLLM_BENCH ===/{flag=0}flag' \
    "$log_file" > "$out_dir/vllm_bench.json"

  if [ ! -s "$out_dir/vllm_bench.json" ]; then
    echo "Error: empty vllm bench result extracted from $log_file" >&2
    return 1
  fi
  rm -f "$log_file"
}

run_case() {
  local case_dir="$1"
  local case_name="$2"
  local ns_ip="$3"
  local ns_vb="$4"

  local vllm_yaml="$case_dir/vllm.yaml"
  local ip_config="$case_dir/inference_perf_config.yml"
  local vb_job="$case_dir/vllm_bench_job.yaml"
  local out_dir="$case_dir/output"

  for f in "$vllm_yaml" "$ip_config" "$vb_job"; do
    if [ ! -f "$f" ]; then
      echo "Skipping $case_name: missing $f"
      return 2
    fi
  done

  mkdir -p "$out_dir"

  echo "=== [$case_name] Ensuring namespaces $ns_ip and $ns_vb ==="
  ensure_namespace "$ns_ip" &
  local pid_ns_ip=$!
  ensure_namespace "$ns_vb" &
  local pid_ns_vb=$!
  wait "$pid_ns_ip" "$pid_ns_vb"

  echo "=== [$case_name] Bringing up fresh vLLM in both namespaces (parallel) ==="
  bring_up_vllm "$ns_ip" "$vllm_yaml" &
  local pid_vllm_ip=$!
  bring_up_vllm "$ns_vb" "$vllm_yaml" &
  local pid_vllm_vb=$!

  local rollout_ok=0
  wait "$pid_vllm_ip" || rollout_ok=1
  wait "$pid_vllm_vb" || rollout_ok=1
  if [ "$rollout_ok" -ne 0 ]; then
    echo "Error: vLLM rollout failed in one or both namespaces" >&2
    return 1
  fi

  echo "=== [$case_name] Running inference-perf and vllm bench (parallel) ==="
  run_inference_perf "$ns_ip" "$ip_config" &
  local pid_ip=$!
  run_vllm_bench "$ns_vb" "$vb_job" &
  local pid_vb=$!

  local client_ok=0
  wait "$pid_ip" || client_ok=1
  wait "$pid_vb" || client_ok=1

  # Extract whatever results we can even if a Job failed; logs may still hold useful info.
  echo "=== [$case_name] Extracting results ==="
  extract_inference_perf_results "$ns_ip" "$out_dir" || client_ok=1
  extract_vllm_bench_results "$ns_vb" "$out_dir" || client_ok=1

  if [ "$client_ok" -ne 0 ]; then
    echo "Error: one or both client Jobs failed for $case_name" >&2
    return 1
  fi

  echo "=== [$case_name] Rendering comparison ==="
  python3 "$SCRIPT_DIR/render_comparison.py" \
    --inference-perf "$out_dir/inference_perf.json" \
    --vllm-bench "$out_dir/vllm_bench.json" \
    --output "$out_dir/comparison.txt"

  echo "Results for $case_name written to $out_dir/"
  return 0
}

cleanup_leftover_namespaces() {
  local pattern="^${NAMESPACE_PREFIX}-"
  local leftover
  leftover=$(kubectl get namespaces -o jsonpath="{.items[*].metadata.name}" \
    | tr ' ' '\n' | grep -E "$pattern" || true)
  if [ -n "$leftover" ]; then
    echo "Deleting leftover namespaces: $leftover"
    kubectl delete namespace $leftover --ignore-not-found --wait=true
  fi
}

echo "=== Checking for leftover namespaces from previous runs ==="
cleanup_leftover_namespaces

PASSED=()
FAILED=()
SKIPPED=()
RUN_NAMESPACES=()

for case_dir in "$CASES_DIR"/*/; do
  CASE_DIR="${case_dir%/}"
  CASE_NAME="$(basename "$CASE_DIR")"
  if [ -n "$ONLY_CASE" ] && [ "$CASE_NAME" != "$ONLY_CASE" ]; then
    continue
  fi

  NS_SUFFIX="${CASE_NAME//_/-}"
  NS_IP="${NAMESPACE_PREFIX}-${NS_SUFFIX}-ip"
  NS_VB="${NAMESPACE_PREFIX}-${NS_SUFFIX}-vb"
  RUN_NAMESPACES+=("$NS_IP" "$NS_VB")

  echo
  echo "########################################"
  echo "### Case: $CASE_NAME"
  echo "###   inference-perf ns: $NS_IP"
  echo "###   vllm bench    ns: $NS_VB"
  echo "########################################"

  run_case "$CASE_DIR" "$CASE_NAME" "$NS_IP" "$NS_VB"
  rc=$?
  case "$rc" in
    0) PASSED+=("$CASE_NAME") ;;
    2) SKIPPED+=("$CASE_NAME") ;;
    *) FAILED+=("$CASE_NAME") ;;
  esac
done

echo
echo "########################################"
echo "### Summary"
echo "########################################"
echo "Passed:  ${#PASSED[@]} ${PASSED[*]}"
echo "Failed:  ${#FAILED[@]} ${FAILED[*]}"
echo "Skipped: ${#SKIPPED[@]} ${SKIPPED[*]}"

if [ "$CLEANUP" = true ] && [ "${#RUN_NAMESPACES[@]}" -gt 0 ]; then
  echo
  echo "=== Cleaning up namespaces ==="
  kubectl delete namespace "${RUN_NAMESPACES[@]}" --ignore-not-found --wait=false
fi

[ "${#FAILED[@]}" -eq 0 ]
