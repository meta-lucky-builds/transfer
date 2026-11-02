# TTT layers for RoboJEPA: build status and training guide

This code implements baseline B1 of [SPECIFICATION.md](SPECIFICATION.md): RoboJEPA with a nominal servo bridge and persistent adaptation of LoRA and token-residual layers. The recurrent belief is not built. It has reserved inputs in the residual, so it can be added later.

## What exists

| Part | File | Spec |
|---|---|---|
| Units, additive joint propagation | `robojepa_ttt/robot_io/so101_units.py` | 3.1, 3.4 |
| Shared limiter | `robojepa_ttt/robot_io/command_limiter.py` | 14.3 |
| Event log and session recorder | `robojepa_ttt/robot_io/event_log.py`, `recorder.py` | 4, 10.1 |
| Local plan acceptance | `robojepa_ttt/robot_io/supervisor.py` | 15.2 |
| Servo bridge and fit | `robojepa_ttt/models/servo.py` | eq. 8 |
| LoRA, for both attention layouts | `robojepa_ttt/models/lora.py` | eq. 11 |
| Command encoder, token residual | `robojepa_ttt/models/command_encoder.py`, `token_residual.py` | 6.3, eq. 12 |
| Causal rollouts, roles, KV cache versions | `robojepa_ttt/models/adapted_predictor.py` | 9.2, 13.3 |
| Losses | `robojepa_ttt/learning/losses.py` | eq. 22, 25, 30 |
| Dense windows | `robojepa_ttt/learning/windows.py` | eq. 32 |
| Replay, credits, feature cache | `robojepa_ttt/learning/replay.py` | 11 |
| Offline stage 3 | `robojepa_ttt/learning/offline_train.py` | 10.6 |
| Online builder | `robojepa_ttt/learning/online_builder.py` | 9.7, 12 |
| Gate, versions, rollback | `robojepa_ttt/learning/promotion.py` | 13 |
| Online loop | `robojepa_ttt/learning/online_runner.py` | 12.2 |
| Six-joint CEM, deterministic | `robojepa_ttt/planning/joint_cem.py` | 14.3 to 14.6 |
| Dataset import, feature store | `robojepa_ttt/data/` | 10 |
| Metrics, software shifts, simulated arm | `robojepa_ttt/evaluation/` | 19 |

## Verification status

**Checked on CPU:** the test suite in `tests/`, run on the real pinned predictor code at tiny size, plus LoRA attachment on the released 300M and 2B architectures at full size without weights.

**Not checked yet:**
- released weights, real features and any real dataset
- bf16 on CUDA, memory and latency
- `prepare_lerobot.py` and the event recorder, which have never run against LeRobot or an arm
- whether adaptation helps on real data. The simulated-arm tests prove that the machinery works, nothing more

## Training order

1. **Environment.** Install PyTorch for your CUDA version, then `pip install -r requirements.txt`. Clone `facebookresearch/robo_jepa` at `f93106f5` into `third_party/robo_jepa`, or copy it from this folder.

2. **GPU acceptance, about 15 minutes.** Needs only the released checkpoint.

   ```bash
   python scripts/gpu_acceptance.py --predictor-config sources/wm_300m_config.json --weights /path/to/world_model.pth.tar
   ```

   Read `gpu_acceptance.json`. `lora_grad_with_guard` must be above zero. `achievable_updates_per_s_at_10pct_duty` decides the real online update rate.

3. **Prepare data.** Run once per dataset, and state its units explicitly. Confirm degrees and gripper direction by inspecting the dataset first: the script cannot detect them. Camera names map to semantic ids from the RoboJEPA `view_map`: left 0, right 1, wrist 2, top 5, front 6.

   ```bash
   python scripts/prepare_lerobot.py --repo-id USER/DATASET --use-degrees --gripper-closed 0 --gripper-open 100 --gripper-verified --cameras front=6 wrist=2 --encoder /path/to/vjepa2_1_vitG_384.pt --out data/DATASET
   ```

4. **Fit the servo model.**

   ```bash
   python scripts/fit_servo.py --episodes data/DATASET/episodes --out data/DATASET/servo.json
   ```

5. **Write `units.json`** with `use_degrees`, the gripper device values, `gripper_direction_verified`, and `center` and `scale` per joint. Use allowed-range midpoint and half-range for the arm, and 0.5 and 0.5 for the gripper.

6. **Split by session.** Use `split_episodes` from `robojepa_ttt/data/import_lerobot.py`, and keep test sessions untouched. Remember that public LeRobot data may overlap RoboJEPA's own training set, so a held-out recording from your arm is the only clean test.

7. **Train stage 3.** Start with `--microbatch 1` and raise it only after checking memory.

   ```bash
   python scripts/train_offline.py --predictor-config sources/wm_300m_config.json --weights W --units U --servo S --train data/train/episodes --validation data/val/episodes --features data/all/features --out runs/stage3
   ```

8. **Evaluate B0 against adapted weights.**

   ```bash
   python scripts/evaluate_b0_b1.py --predictor-config sources/wm_300m_config.json --weights W --units U --servo S --episodes data/test/episodes --features data/all/features --adapters runs/stage3/best.pt
   ```

   Report per-episode results. Stop and investigate if horizons 1 and 2 do not improve.

9. **Online, shadow mode only.** `OnlineRunner` with `shadow=True` predicts, trains and evaluates, and cannot change what the robot runs. Apply `CommandShift` to test adaptation.

## Remaining work

- **Live robot driver.** It is deliberately not written. It needs measured limits and a commissioned stop policy, which do not exist yet. `OnlineRunner`, `Supervisor`, `JointCem` and `record_session` are the pieces it connects.
- **Belief.** Compact dynamics, observation model, IMM-EKF, belief-sampled planning and CVaR.
- **Regime bank.** Layers are regime-keyed, but detection, mode switching and quarantine promotion are not built.
- **Foreground and task probes in the gate.** The gate accepts them; they need task labels.

## Decisions that differ from the papers or the spec

| Decision | Why |
|---|---|
| Residual has no `h` or `x` input | The belief does not exist in B1 |
| Compact correction head omitted | It corrects `F`, which does not exist in B1 |
| `L_cons` absent offline, other weights unchanged | It needs the compact model |
| Online windows hold 10 keyframes | The spec leaves the length open. 8 context plus 2 rollout |
| Snapshot needs at least one builder update | An untrained snapshot equals serving and wastes the gate |
| Action probe limits: ratio at least 0.5, cosine at least 0 | The spec names the check and gives no numbers |
| Anchors grouped by episode by default | There are no task labels yet |
| Limiter refuses a start outside limits | Clipping it produced a full-range swing on a hold command |
| Adapters, L1, and rollout terms instead of JEPA-TTT's full-predictor MSE | From the spec. A JEPA-TTT replication needs its own settings: AdamW 5e-5, weight decay 1e-3, batch 64, MSE, all of the predictor, its admission rule |

## Findings from the upstream code

- `RoPEAttention` uses fused `qkv` when `num_kv_heads == num_heads` (300M) and split `q`, `kv` otherwise (1B, 2B). LoRA handles both.
- Frozen layers build no autograd graph, so the first 16 blocks cost no gradient memory. This needed no code.
- `WorldModel.rollout` feeds measured states, so this package does not use it.
- The paper states equal teacher-forcing and rollout mixing. The code gives the rollout `(K-1)/(2K)` in total: 0.25 at `K=2`, 0.45 at `K=10`. This does not affect this package.
- A CPU test cannot show that `torch.clear_autocast_cache()` matters. `gpu_acceptance.py` measures it on CUDA.
