#!/usr/bin/env bash
# End-to-end run on one multi-GPU machine: prepare -> split -> units -> servo -> smoke
# tests -> B1 (all GPUs, data parallel) and belief (one GPU) in parallel -> B0 vs B1.
#
#   bash scripts/run_pipeline.sh            # everything, skipping stages already done
#   STAGES="eval" bash scripts/run_pipeline.sh   # one stage
#
# Every stage stops the script on failure and leaves a marker when it succeeds, so a
# re-run resumes after the last finished stage. Settings below can be overridden from
# the environment. Logs go to $RUN/logs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
WORK="${WORK:-/workspace}"
DATA="${DATA:-$WORK/data/armnet}"
RUN="${RUN:-$WORK/runs/armnet}"
REPO_ID="${REPO_ID:-armnet/armnetbench_v01_lerobot_so101}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-50}"
CAMERAS="${CAMERAS:-front=6 top=5}"
VIEWS="${VIEWS:-6 5}"
ENCODER="${ENCODER:-$WORK/ckpt/vjepa2_1_vitG_384.pt}"
PRED_CONFIG="${PRED_CONFIG:-sources/wm_2b_config.json}"
WEIGHTS="${WEIGHTS:-$WORK/ckpt/cooldown_2b_450k.pth.tar}"
DEVICE="${DEVICE:-cuda}"
SETTINGS="${SETTINGS:-}"
# Joint repeatability in compact units is a property of YOUR arm; for public data this is
# a placeholder that the belief's calibration step then rescales (spec 6.5, 7.5).
JOINT_SENSOR_STD="${JOINT_SENSOR_STD:-0.002}"
# Short schedule: early stopping makes these ceilings. Full spec values: 20000 / 40000.
B1_MAX_UPDATES="${B1_MAX_UPDATES:-2000}"
B1_WARMUP="${B1_WARMUP:-100}"
B1_CURRICULUM="${B1_CURRICULUM:-500:2,1000:5,500:10}"
B1_VAL_INTERVAL="${B1_VAL_INTERVAL:-250}"
B1_PATIENCE="${B1_PATIENCE:-4}"
BELIEF_MAX_UPDATES="${BELIEF_MAX_UPDATES:-4000}"
BELIEF_VAL_INTERVAL="${BELIEF_VAL_INTERVAL:-500}"
STAGES="${STAGES:-setup prepare split units servo smoke train eval}"
SKIP_PREP="${SKIP_PREP:-0}"

if [ "$DEVICE" = "cuda" ]; then NGPU="${NGPU:-$(nvidia-smi -L | wc -l)}"; else NGPU="${NGPU:-2}"; fi
EFFECTIVE=16
if [ $((EFFECTIVE % NGPU)) -ne 0 ]; then echo "STOP: $NGPU GPUs do not divide the batch of $EFFECTIVE (use 1, 2, 4, 8 or 16)"; exit 1; fi
MICRO=$((EFFECTIVE / NGPU))   # one microbatch per GPU per update
SET=(); [ -n "$SETTINGS" ] && SET=(--settings "$SETTINGS")
mkdir -p "$RUN/logs" "$DATA"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}" TORCH_HOME="${TORCH_HOME:-$WORK/torch_cache}"

say() { echo; echo "=== $(date +%H:%M:%S) $* ==="; }
done_marker() { [ -f "$RUN/.done_$1" ]; }
mark() { touch "$RUN/.done_$1"; }
want() { [[ " $STAGES " == *" $1 "* ]]; }
SPLITS="$DATA/splits"

if want setup && ! done_marker setup; then
  say "setup"
  if [ "$DEVICE" = "cuda" ]; then
    FREE=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc 0-9)
    echo "free disk: ${FREE} GB, GPUs: $NGPU"
    [ "$FREE" -ge 250 ] || { echo "STOP: need at least 250 GB free on $WORK"; exit 1; }
    pip install -q einops pyyaml wandb scipy huggingface_hub safetensors pandas pyarrow av
    pip install -q --no-deps timm
    python -c "import torch, av; assert torch.cuda.is_available(); assert 'libdav1d' in av.codecs_available; print('torch', torch.__version__, torch.cuda.device_count(), 'x', torch.cuda.get_device_name(0))"
    mkdir -p "$(dirname "$ENCODER")" "$(dirname "$WEIGHTS")"
    [ -f "$ENCODER" ] || wget -q -O "$ENCODER" https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt &
    [ -f "$WEIGHTS" ] || wget -q -O "$WEIGHTS" https://dl.fbaipublicfiles.com/robojepa/wm/wm-2b/cooldown_2b_450k.pth.tar &
    wait
    ls -la "$ENCODER" "$WEIGHTS"
  fi
  mark setup
fi

if want prepare && ! done_marker prepare && [ "$SKIP_PREP" != "1" ]; then
  say "prepare: pilot, 1 episode per task"
  COMMON=(--repo-id "$REPO_ID" --use-degrees --gripper-closed 0 --gripper-open 100 --gripper-verified
          --cameras $CAMERAS --expect-fps 20 --encoder "$ENCODER")
  python scripts/prepare_lerobot.py "${COMMON[@]}" --episodes-per-task 1 --block 1 --out "$WORK/data/pilot" > "$RUN/logs/pilot.log" 2>&1 \
    || { tail -30 "$RUN/logs/pilot.log"; echo "STOP: pilot preparation failed"; exit 1; }
  python -c "import torch, glob; f=sorted(glob.glob('$WORK/data/pilot/features/*.pt')); t=torch.load(f[0]); assert t.shape[1:] == (2, 15, 20, 1664), t.shape; assert torch.isfinite(t.float()).all(); print('PILOT OK', len(f), tuple(t.shape))"
  say "prepare: $EPISODES_PER_TASK per task on $NGPU GPUs"
  pids=()
  for i in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$i python scripts/prepare_lerobot.py "${COMMON[@]}" --episodes-per-task "$EPISODES_PER_TASK" --block 10 \
      --shard "$i" --num-shards "$NGPU" --out "$DATA" > "$RUN/logs/prepare_$i.log" 2>&1 &
    pids+=($!)
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { tail -30 "$RUN/logs/prepare_$i.log"; echo "STOP: preparation shard $i failed"; exit 1; }
  done
  tail -n 1 "$RUN"/logs/prepare_*.log
  mark prepare
fi

if want split && ! done_marker split; then
  say "split by pseudo-session"
  python scripts/split_dataset.py --root "$DATA" --group-size 10 | tee "$RUN/logs/split.log"
  mark split
fi

if want units && ! done_marker units; then
  say "units from training positions"
  python scripts/make_units.py --episodes "$SPLITS/train/episodes" --out "$DATA/units.json" | tail -3
  mark units
fi

if want servo && ! done_marker servo; then
  say "servo fit on training episodes"
  python scripts/fit_servo.py --episodes "$SPLITS/train/episodes" --out "$DATA/servo.json" | tee "$RUN/logs/servo.log"
  mark servo
fi

B1_ARGS=(--predictor-config "$PRED_CONFIG" --weights "$WEIGHTS" --units "$DATA/units.json" --servo "$DATA/servo.json"
         --train "$SPLITS/train/episodes" --validation "$SPLITS/validation/episodes" --features "$DATA/features" --device "$DEVICE" "${SET[@]}")
BELIEF_ARGS=(--units "$DATA/units.json" --servo "$DATA/servo.json" --train "$SPLITS/train/episodes"
             --validation "$SPLITS/validation/episodes" --features "$DATA/features" --views $VIEWS
             --joint-sensor-std "$JOINT_SENSOR_STD" --device "$DEVICE" "${SET[@]}")

if want smoke && ! done_marker smoke; then
  say "smoke: 2 updates of B1 and of the belief on real features"
  rm -rf "$RUN/smoke_b1" "$RUN/smoke_belief"
  python scripts/train_offline.py "${B1_ARGS[@]}" --out "$RUN/smoke_b1" --max-updates 2 --warmup-updates 1 --curriculum 2:2 \
    --validation-interval 1 --validation-windows 4 --microbatch 1 > "$RUN/logs/smoke_b1.log" 2>&1 \
    || { tail -40 "$RUN/logs/smoke_b1.log"; echo "STOP: B1 smoke run failed"; exit 1; }
  grep -q '"validation"' "$RUN/logs/smoke_b1.log" && test -f "$RUN/smoke_b1/best.pt" || { tail -20 "$RUN/logs/smoke_b1.log"; echo "STOP: B1 smoke produced no validated checkpoint"; exit 1; }
  python scripts/train_belief.py "${BELIEF_ARGS[@]}" --out "$RUN/smoke_belief" --max-updates 2 --validation-interval 1 \
    --sketch-frames 2000 > "$RUN/logs/smoke_belief.log" 2>&1 \
    || { tail -40 "$RUN/logs/smoke_belief.log"; echo "STOP: belief smoke run failed"; exit 1; }
  test -f "$RUN/smoke_belief/belief.pt" || { echo "STOP: belief smoke wrote no model"; exit 1; }
  echo "smoke OK"
  mark smoke
fi

if want train && ! done_marker train; then
  say "train: B1 on $NGPU GPUs (microbatch $MICRO each) and the belief on GPU $((NGPU - 1)), in parallel"
  CUDA_VISIBLE_DEVICES=$((NGPU - 1)) nohup python scripts/train_belief.py "${BELIEF_ARGS[@]}" --out "$RUN/belief" \
    --calibration "$SPLITS/calibration/episodes" --max-updates "$BELIEF_MAX_UPDATES" --validation-interval "$BELIEF_VAL_INTERVAL" \
    > "$RUN/logs/belief.log" 2>&1 &
  belief_pid=$!
  if [ "$DEVICE" = "cuda" ]; then LAUNCH=(torchrun --standalone --nproc_per_node "$NGPU"); else LAUNCH=(torchrun --standalone --nproc_per_node "$NGPU"); fi
  "${LAUNCH[@]}" scripts/train_offline.py "${B1_ARGS[@]}" --out "$RUN/b1" --microbatch "$MICRO" \
    --max-updates "$B1_MAX_UPDATES" --warmup-updates "$B1_WARMUP" --curriculum "$B1_CURRICULUM" \
    --validation-interval "$B1_VAL_INTERVAL" --patience "$B1_PATIENCE" > "$RUN/logs/b1.log" 2>&1 \
    || { tail -40 "$RUN/logs/b1.log"; echo "STOP: B1 training failed (belief keeps running, pid $belief_pid)"; exit 1; }
  echo "B1 finished: $(grep -o "{'updates'.*" "$RUN/logs/b1.log" | tail -1)"
  wait "$belief_pid" || { tail -40 "$RUN/logs/belief.log"; echo "STOP: belief training failed"; exit 1; }
  echo "belief finished"
  mark train
fi

if want eval && ! done_marker eval; then
  say "evaluate B0 (zero adapters) against B1 on the untouched test split"
  python scripts/evaluate_b0_b1.py --predictor-config "$PRED_CONFIG" --weights "$WEIGHTS" --units "$DATA/units.json" \
    --servo "$DATA/servo.json" --episodes "$SPLITS/test/episodes" --features "$DATA/features" \
    --adapters "$RUN/b1/best.pt" --device "$DEVICE" | tee "$RUN/b0_vs_b1.json"
  python - "$RUN/belief/report.json" <<'EOF'
import json, sys
report = json.load(open(sys.argv[1]))
print("belief validation NLL per coordinate:", round(report["validation_nll"], 4))
for name, block in report.get("innovations_after_calibration", report.get("innovations", {})).items():
    print(f"  {name}: NIS per coordinate {block['nis_per_coordinate']:.3f} (1 is calibrated)")
if "calibration" in report:
    print("calibration multipliers:", [round(m, 3) for m in report["calibration"]["multipliers"]])
EOF
  mark eval
fi
say "pipeline finished; results in $RUN"
