# Sleep: offline memory consolidation and dreaming for the RoboJEPA TTT system

**Status:** proposed specification, not implemented or validated. Prepared 2026-11-03 as the theory/spec part of the Sleep build. **Scope:** a frozen RoboJEPA latent world model with LoRA and token-residual test-time adapters, a recurrent IMM-EKF belief, an SO-101 arm, two fixed phone cameras, no wrist camera, no force sensing. **Parent documents:** [SPECIFICATION.md](../SPECIFICATION.md) (cited as "spec §n"), [TRAINING_GUIDE.md](../TRAINING_GUIDE.md), [sleep_paper_notes.md](sleep_paper_notes.md), [sleep_research_notes.md](sleep_research_notes.md).

The source is "Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories" (Behrouz, Hashemi, Javanmard, Mirrokni; arXiv 2606.03979v2, cited as "the paper"). I read its full text: main body, appendices A and B, and the figure captions. I could not see the plotted values in its figures. What I read and what I could not access is listed in [sleep_research_notes.md](sleep_research_notes.md).

All numerical defaults in this document are **hypotheses** unless they are marked *measured* or *inherited* (inherited means already fixed in the parent spec or the code). A hypothesis must be tested before anyone relies on it.

---

## 0. Contract issues (read first; the orchestrator must reconcile these)

The interface given to SLEEP-2/3/4 is close to right, but it is incomplete in ways that would produce **silently wrong behaviour** with the existing code. Each item gives the evidence (file and function), the consequence, and the required resolution. Items C1–C5 are blocking.

| # | Issue | Evidence in current code | Consequence if ignored | Required resolution | Owner |
|---|---|---|---|---|---|
| **C1** | Slot activation state must live in the snapshot and must not be shared between roles | `AdaptedPredictor._shared_base_tensors` shares **every predictor buffer** and every predictor parameter whose name lacks `"deltas"`. `adapter_state()` collects **named parameters only** | A mask stored as a buffer is aliased by `clone_role`: activating a slot in the student also activates it in the teacher and in serving. A mask stored as a Python attribute is missing from `adapter_state()`, from `state_digest` and from `VersionStore`, so a rollback cannot restore it and two different slot maps can share one digest | Store activity as a non-trainable `nn.Parameter` (`requires_grad=False`) whose name contains `".deltas."`, for example `...qkv.tiers.slow.deltas.<regime>.slot_active` (a float 0/1 vector). Invariant: an inactive slot has `B == 0` exactly. Tests: clone non-aliasing, snapshot round-trip of the slot map, and digest change on activation | SLEEP-2 |
| **C2** | Legacy trainable-parameter selection must reach the fast tier only | `set_trainable("online")` calls `lora_parameters(predictor, regime)`, which iterates `module.deltas[regime].parameters()` for every `isinstance(module, LoRALinear)`. `OnlineBuilder.__init__` and `reset_optimizer` call `set_trainable("online")` | If the tiered module puts all tiers under `deltas[regime]`, online training also trains slow and offline slots, which breaks the whole frequency separation. If the tiered module is not a `LoRALinear`, online training trains **no LoRA at all** and nothing raises | The tiered wrapper must subclass `LoRALinear` and keep `self.deltas` (a `ModuleList[regime][slice]`) as **exactly the fast tier**. Slow and offline tiers live under a different attribute whose names still contain `".deltas."`. `set_lora_regime` must keep working. Test: after conversion, `set_trainable("online")` returns exactly `tier_parameters(model, "fast", r)` | SLEEP-2 |
| **C3** | The token residual is not covered by the tier contract | `TokenResidual.online_parameters(r)` returns `regimes[r].modulation_out` and `regimes[r].b_o`, and the forward pass has one branch | Online training keeps writing the stage-3 residual, so residual knowledge can never be consolidated or reset | `convert_to_tiered` must also convert the residual. The trunk (`a_o`, modulation hidden layers, embeddings) stays shared and frozen. Each tier slot $k$ owns its own (`modulation_out_k`, `b_o_k`). Output: $\mathrm{LN}_0[z^0+\sum_k b_{o,k}\tanh(a_o z^0+m_k(M(c)))]$. The legacy `regimes[r].modulation_out`/`b_o` become the **fast** slot (the same C2 rule), and the stage-3 values move to the offline tier (§3.3) | SLEEP-2 |
| **C4** | The compact belief's online correction is a fast tier with no slots and no versioning | `CompactBelief.correction_a/b` (rank 8, shared by all regimes). `VersionStore` saves visual adapter snapshots only. The slow belief prior has no store (`initial_state` accepts `slow_mean/slow_var`, but nothing persists them) | Belief consolidation (§7) has no slow slot to write to. Rolling back across a belief consolidation cannot restore the compact correction or the slow prior | Add slow correction slots to `CompactBelief` (same rule: zero-initialised B, frozen, activity vector in the state). Version the compact correction state and a per-regime `SlowPrior(mean, var)` together with the visual snapshot under one version id. Until this exists, belief consolidation is **disabled** (§7.6) | belief agent, with SLEEP-2 |
| **C5** | The promotion gate cannot accept a function-preserving candidate | `evaluate_gate` requires relative improvement $\ge 2\%$ **and** a block-bootstrap upper bound $<0$ | An exact consolidation has challenger−serving differences of exactly 0, so the gate rejects it **deterministically** (Proposition 9.8). Every consolidation would be rejected | Add (a) a **verify-only** path for exact transplants and (b) a **non-inferiority** gate for lossy consolidation (§9.8, §10.3). Keep the superiority gate for dreaming. Implement these in `robojepa_ttt/sleep/` (wrapping `GateInput`), not by editing `promotion.py` | SLEEP-3 (decision logic), SLEEP-4 (wiring) |
| C6 | Snapshot schema: tier conversion changes the meaning of existing keys | `load_adapter_state` checks the key set only. After conversion under C2, `...deltas.0.0.lora_a` means *fast* where it used to mean *stage-3* | Loading a pre-conversion snapshot (for example the stored `offline-reference`) into a converted model puts stage-3 knowledge into the fast tier and leaves the offline tier inconsistent. Nothing raises | Record a `parameterisation_schema` (for example `"legacy-v1"` or `"tiered-v1"` plus the tier configs) in every `VersionStore` payload and refuse a cross-schema load. `convert_to_tiered` performs the only legal migration (§3.3) | SLEEP-2, SLEEP-4 |
| C7 | The builder's references go stale after a sleep | `OnlineBuilder.reference_state` is captured once, keyed by trainable names. `OnlineRunner.builder` and `ModelHub.reference` have no replace API | After the fast tier is reset, the proximal term pulls toward the pre-sleep fast weights and the anchor toward the pre-sleep reference | After an accepted sleep, construct a **new** `OnlineBuilder(model=new_builder, reference=new_reference)` from `clone_role` of the promoted version, and swap `hub.builder`, `hub.reference` and `runner.builder` atomically at the planning boundary (§10.4) | SLEEP-4 |
| C8 | External challengers | `ModelHub.maybe_snapshot` can only snapshot the builder. `OnlineRunner.resolve` scores the challenger only for predictions made while it was live, at horizon 1 | A sleep candidate cannot enter the future-data gate through the existing API | SLEEP-4 sets `hub.challenger`/`hub.challenger_snapshot` from a sleep candidate (the same fields `maybe_snapshot` sets), clears the score lists as `OnlineRunner.maybe_snapshot` does, and adds horizon-2 and ranking diagnostics (§10.3) | SLEEP-4 |
| C9 | Duty accounting | `CreditLedger.can_update` uses cumulative `busy_s / (now - start_s)` since the start | (a) If sleep compute is recorded with `record_busy`, wake training starves for hours afterwards. (b) A long idle hold leaves "duty credit" that allows bursts far above 10% | Sleep never calls `record_busy` on the wake ledger and keeps its own `SleepLedger`. Separately (not owned here), the wake duty should use a sliding window | SLEEP-4; ledger fix to the code owner |
| C10 | Held-out data for sleep | `ReplayBuffer` has recent, reservoir, anchor and quarantine partitions only | Sleep would check fidelity on windows it trains on, and dense windows overlap | The sleep data plan (§5.6) builds a train/validation split by temporal blocks with **guard gaps of at least one window span** (54 raw ticks, 1.8 s) at sleep time, from window references. It does not mutate `ReplayBuffer` | SLEEP-3/4 |
| C11 | `TierConfig.consolidate_every` | Contract field | Sleep is event-triggered (§10.1), not periodic | Interpret it as $U_{\max}$, the maximum builder updates between consolidations of that tier. The scheduler owns the triggers | SLEEP-2/4 |
| C12 | Dreams must stay structurally unable to enter replay | `WindowRef = (episode_id, start_tick)` points at raw records | None today. Keep it that way: a dream has no episode, so it cannot be admitted, anchored or scored | Dreams live in a separate in-memory `DreamBatch` with no `Episode`. Test: no code path converts a dream into a `WindowRef` | SLEEP-4 |

Contract summary for SLEEP-2 (normative): `convert_to_tiered(model, tiers)` with tiers `fast` (1 slot), `slow` ($N_S$ slots) and `offline` (1 slot, frozen); `tier_parameters(model, tier, regime=0, slot=None)`; `activate_slot(model, tier) -> slot index` (zero-initialised B, so function is preserved, Theorem 9.1); `reset_tier(model, tier)` (B←0, A←fresh seeded init, slot inactive); `active_slots(model) -> {tier: {regime: [slots]}}`. Additionally required: `transplant(model, src=("fast", r), dst=("slow", r, k))` (exact, Theorem 9.2), `merge_slots(...)` (exact concatenation when rank permits, Proposition 9.3), and `whitened_compaction(...)` (Theorem 9.4), or SLEEP-3 implements these on top of `tier_parameters`.

---

## 1. What Sleep means here, in one page

**The system already has a memory hierarchy.** From fastest to slowest:

1. recurrent belief state ($h_t$, $\xi_t$, $d_t$, regime posterior $\pi_t$) and KV caches;
2. online builder adapters;
3. the promoted serving version;
4. stage-3 offline adapters;
5. the frozen base and encoder.

What it lacks is any mechanism that moves validated fast knowledge into **separate, protected, bounded** slow capacity, and then frees the fast capacity. Without that mechanism the fast adapters accumulate everything: under A→B→A recurrence with one regime ($K=1$, the initial production setting), learning B overwrites A in the same parameters. The only protection is the replay anchors and a proximal term pointing at the offline reference.

**Sleep, as specified here**, is a gated offline procedure run during a commissioned robot hold:

- **NREM (consolidation, §5–§7).** Copy or distil validated fast knowledge into a newly activated slot of the slow tier, using real replay windows, on-policy student rollouts, planner-chosen command sequences and an action-ranking agreement term. Optionally distil what the belief has persistently inferred into weights. Then reset the fast tier and its optimizer. The fast tier recovers its plasticity and the function is preserved.
- **REM (dreaming, §8).** Train a separate dream slot on synthetic command sequences that start from real contexts and are labelled by a **teacher that knows more than the student**: a hindsight teacher with a smoothed slow context, or a frozen older model for retention. A dream is never an observation. A dream slot is kept only if it improves held-out **future real** transitions through the promotion gate.
- **Gating (§10.3).** Nothing a sleep produces is installed until it passes. An exact transplant needs a verify-only check. A lossy consolidation needs a non-inferiority gate. A dream needs the existing superiority gate. Aborting a sleep is always safe: the pre-sleep serving model and builder state are untouched until promotion.

**Two results shape the design (§9).**

1. A rank-matched LoRA or residual transplant is **exact**. Distillation is required only for lossy transfers: compaction when slots run out, belief-to-weights consolidation, and selective consolidation that keeps the fast tier's gains while discarding its anchor regressions.
2. For a **deterministic** world model, distilling a model into a copy of itself is a zero-gradient fixed point. Dreaming therefore needs a teacher that differs from the student. LLM self-training escapes this only because sampled outputs differ from the model's expectation.

---

## 2. Notation and units

| Symbol | Meaning | Value / unit |
|---|---|---|
| $\Delta$ | macro step | 0.2 s nominal (inherited, spec §4) |
| $f_\cdot$ | update frequency (paper Definition 1) | updates per hour of **valid operation** |
| $W_0$ | frozen base projection | — |
| $\Delta W_\tau^{(r,k)}$ | LoRA delta of tier $\tau$, regime $r$, slot $k$ | $\frac{\alpha_\tau}{r_\tau}B A$ |
| $\mathcal F,\mathcal S,\mathcal O$ | fast, slow, offline tiers | — |
| $N_S$ | slow slots per regime | 8 (hypothesis) |
| $D_z$ | masked LN0 token L1, eq. 22 (`token_metric`) | dimensionless |
| $D_g$ | goal distance, eq. 34, divided by the fixed calibration scale | dimensionless |
| SUE | *student-update equivalent*: one optimizer step on effective batch 16, horizon 2 | 3.66 s at 2B (*measured*, TRAINING_GUIDE GPU table) |
| $T$, $S$ | teacher, student | — |
| $\mathcal R_{\rm tr},\mathcal R_{\rm val}$ | sleep train / validation window sets | §5.6 |

Effective function of a projection for regime $r$ after conversion:
\[
W^{(r)}=W_0+\Delta W_{\mathcal O}^{(r)}+\sum_{k\in\mathcal A_{\mathcal S}(r)}\Delta W_{\mathcal S}^{(r,k)}+\Delta W_{\mathcal F}^{(r)},
\tag{S1}
\]
where $\mathcal A_{\mathcal S}(r)$ is the set of active slow slots. The residual has the analogous form given in C3.

---

## 3. Memory tiers and update frequencies in this system's units

### 3.1 Tier table

| Tier | Contents | Written by | Frequency (2B, one GPU) | Lifetime / reset | Paper analogue |
|---|---|---|---|---|---|
| T0 context (∞) | $h_t$, $\xi_t$ (10 s decay), $d_t$ (identity dynamics, $q_d=10^{-5}$ per 0.2 s), $\pi_t$, KV caches, fixed-lag window (2 s) | filter | 5 Hz cloud; 30 Hz local joint observer | reset on power cycle or camera change (spec §4 table); KV caches invalidated on any version change | attention / in-context memory |
| T0′ slow prior | per-regime $(\mu_d^{\rm prior},P_d^{\rm prior})$ carried across episodes | end of episode; **belief consolidation resets it** (§7) | per episode | versioned with the model (C4) | — (no analogue in the paper) |
| T1 fast $\mathcal F$ | fast LoRA slot (rank 8, Q/V/out, last 8 blocks) + fast residual slot + fast compact correction slot, per regime | `OnlineBuilder` | at most 1/s by the ledger; *measured* 3.66 s/update, so **about 1 per 37 s at 10% duty, about 98 per hour** | reset (B←0, fresh A, fresh AdamW) only when a sleep candidate is promoted | high-frequency CMS block |
| T1′ validated | serving version: a gated snapshot of T1 | `ModelHub.promote` | at most 1 per (30 s snapshot interval + ≥128 valid future transitions = 25.6 s of valid data) ≈ at most 1 per 56 s; in practice much rarer (hypothesis) | rollback keeps ≥3 versions | **no CMS analogue**: a validation filter over T1, not a separate memory |
| T2 slow $\mathcal S$ | $N_S$ pre-allocated masked slots per regime per adapter family | sleep only | 1 per sleep; default triggers give about 1 per 2–4 h of operation (hypothesis) | frozen after writing; freed only by compaction or an explicit eviction policy | mid/low-frequency block with new expert |
| T3 offline $\mathcal O$ | stage-3 LoRA and residual | offline stage 3/4, or deep consolidation (§6.5) | 0 at runtime; days to weeks offline | replaced only by a validated offline run | lowest-frequency block |
| T4 frozen | RoboJEPA base, V-JEPA encoder, PCA sketch, GRU, observation model, covariances | never online | 0 | — | frozen MLPs (zero frequency) |

CMS chunk sizes in this system: $C(\mathcal F)=1$ update ≈ 37 s; $C(\mathcal S)=U_{\max}$ updates ≈ 2 h at $U_{\max}=200$; $C(\mathcal O)$ ≈ the $N_S$ sleeps it takes to fill the slow bank. The frequency ratios $f_{\mathcal F}/f_{\mathcal S}\approx 200$ and $f_{\mathcal S}/f_{\mathcal O}\to\infty$ at runtime are hypotheses. For 300M, the update cost is **not measured**; measure it with `scripts/gpu_acceptance.py` before setting any 300M budget.

### 3.2 Why promotion is not a tier

The serving version shares its parameter tensors with the builder lineage. It is a delayed, filtered copy of T1, not separate capacity. CMS consolidation means moving knowledge into **different** parameters before the faster ones are overwritten (paper §2.2, §3.2). In this system the fast parameters are overwritten continuously, at every update. The CMS rule "consolidate a block before it updates" therefore translates to: **consolidate before the fast tier starts absorbing a new persistent condition**. With $K=1$ that event is a confirmed regime change (trigger T-c, §10.1).

### 3.3 Conversion of an existing stage-3 model (one-way migration)

Run once at deployment start, before `ModelHub` is constructed:

1. Copy the stage-3 LoRA factors from `deltas[r]` into the `offline` tier slot of regime $r$. Copy the stage-3 residual (`modulation_out`, `b_o`) into the offline residual slot.
2. Zero the fast tier: B←0, A←seeded fresh init. Zero the legacy residual `b_o` (now the fast slot) and give `modulation_out` a fresh init.
3. Mark all slow slots inactive with B=0.
4. Assert function equality on a fixed probe batch. In exact arithmetic the change is a regrouping of the same sum, so the bound is floating-point reassociation only: relative $D_z$ ≤ $10^{-6}$ in fp32, ≤ $10^{-2}$ in bf16 (hypothesis tolerance, set from the measured bf16 KV parity of 0.53%).
5. Tag the schema `tiered-v1` (C6) and re-save `offline-reference` under the new schema.

Offline stage-3 training runs on **unconverted** models only. After conversion, `set_trainable("offline")` is not a valid operation and must raise. New offline knowledge enters through deep consolidation (§6.5).

---

## 4. Parameter expansion with bounded pre-allocated slots

### 4.1 Layout

For each converted projection (Q slice, V slice and attention output, in each of the last 8 blocks; K is untouched, inherited from spec §6.6), each regime ($\le 4$, inherited), and each tier:

- `fast`: 1 slot, rank $r_{\mathcal F}=8$, $\alpha=16$ (inherited);
- `slow`: $N_S=8$ slots, rank $r_{\mathcal S}=8$, $\alpha=16$ (hypothesis);
- `offline`: 1 slot holding the stage-3 rank (8).

The residual has the same slot structure per regime: each slot owns `modulation_out` (256→32) and `b_o` (32→1664, zero-initialised). The compact correction (C4) has 8 slow slots of rank 8 (256→54).

### 4.2 Cost

One LoRA slot costs $r(d_{\rm in}+d_{\rm out})$ parameters per projection, so $r(5d+d_v)$ per block, where $d_v$ is the V slice width. For the inspected 2B width $d=2048$ that is ≈ $8\cdot(10240+d_v)\approx 0.1$ M per block and ≈ 0.8 M for 8 blocks: about 0.04% of the 1.86 B base. A full slow bank for 4 regimes holds ≈ 26 M LoRA parameters plus about 0.4 M residual parameters per slot set. **Memory is not the binding constraint.** The binding constraints are:

- (i) interference: an active slot applies to every input of its regime;
- (ii) the evaluation and gating cost per sleep;
- (iii) attribution and rollback bookkeeping.

The bound $N_S$ exists so that storage and per-step compute are $O(1)$ in lifetime (spec: bounded memory, no automatic eviction).

### 4.3 Masking

An inactive slot is skipped in the forward pass (a Python-level list of active slots) and is never handed to an optimizer.

- **Invariant I1:** inactive ⇒ $B=0$ exactly, $A$ equals its recorded seeded init, and `slot_active=0`.
- **Invariant I2:** activating a slot does not change any tensor shape. The pre-allocation means no reallocation and no change to the parameter count of the state dict. Graph capture may recompile once per activation, which is acceptable because activation happens once per sleep. Graphs must read slot tensors live, never constant-fold them (spec §16).
- **Invariant I3:** every activation, reset, transplant or merge produces a **new version string**. `StaleCacheError` then forces a KV and context rebuild (spec §13.3). This holds even when Theorem 9.1 says the outputs are bit-identical: the safety rule is simpler than tracking bit-identity.

### 4.4 Policy when slots run out (strict order; no silent forgetting)

Let $F_r$ be the number of free slow slots of regime $r$ and $\rho=1$ the reserve.

1. **Normal:** $F_r>\rho$. Consolidation writes into a new slot.
2. **Pressure:** $F_r\le\rho$ when a sleep starts. Run **compaction first** (§6.4):
   - choose the $m=2$ oldest active slow slots (or the pair with the smallest whitened residual energy);
   - merge them into one slot by whitened SVD (Theorem 9.4);
   - refine by distillation with teacher = the pre-compaction model on all real data available for regime $r$ (reservoir + anchors + retention dreams, §8.3);
   - pass the non-inferiority gate. Compaction frees $m-1$ slots.
3. **Compaction fails** (whitened energy retained $<0.98$ and the gate rejects). **Defer consolidation:**
   - the fast tier is not reset, so regime $r$ continues exactly as B1 does today;
   - raise `capacity_exhausted(r)`;
   - request deep consolidation (§6.5). This step needs an operator.
4. **Eviction** (opt-in, off by default, as spec §8.2 requires an explicitly evaluated policy):
   - compute each slot's ablation utility $u_k$ = increase in held-out real loss of regime $r$ when slot $k$ is masked;
   - evict $\arg\min_k u_k$ only if $u_k\le\varepsilon_{\rm ev}$ (1% relative, hypothesis) and the non-inferiority gate passes with the slot removed;
   - log the eviction as forgetting.
5. **Regime-level pressure** (all 4 regime slots occupied and a new regime is in quarantine). Spec §8.2 already says quarantine plus offline consolidation. A sleep may propose a **regime merge**: distil two functionally close regimes, whose cross-anchor $D_z$ is below the no-shift noise floor, into one. This is version-2 work and requires its own evaluation.

---

## 5. NREM stage 1: consolidation by knowledge seeding

### 5.1 Compute → consolidate → update, mapped exactly

The paper (§3.3) has three steps.

- (a) **Compute.** The sender's prospective base update is computed but not applied.
- (b) **Consolidate.** The teacher is the model before the update. The student is the model with the prospective sender update, the sender's experts reset and a new target expert. Only the new expert trains.
- (c) **Update.** Apply the sender update, reset the sender experts, activate the new expert.

Here, for regime $r$:

**(a) Compute.**
1. *Drain:* stop new snapshots. Let the pending challenger finish its gate, or discard it.
2. Pin the **teacher** $T$ := serving version $v_T$ by version id and digest. Only validated knowledge is consolidated: builder progress that was never promoted is **not** consolidated. That progress is at most one snapshot interval of updates, about 1–5 updates at 2B, and it is kept or dropped together with the builder state (§10.4).
3. The prospective sender update is "fast tier ← reset" (B←0, fresh A, fresh optimizer). It is recorded but not applied to any live role.

**(b) Consolidate.**
1. The **student** $S$ := `clone_role(T)` with:
   - the fast tier reset;
   - a newly activated slow slot $k^\ast$, initialised by the chosen mode (§5.2);
   - every other parameter frozen.
2. Optimise only slot $k^\ast$ (LoRA, residual and, in belief mode, compact correction) with the objective of §5.3–§5.5.

**(c) Update.** Build the candidate version $C_{\rm cons}$ = $S$ with slot $k^\ast$ active and the fast tier reset. It is promoted **only** through §10.3. The live builder and serving roles are untouched until then. The reset of the live fast tier ("synaptic pruning") therefore happens exactly at promotion.

### 5.2 Consolidation modes

| Mode | When | Slot init | Distillation | Gate |
|---|---|---|---|---|
| **M1 transplant** (default) | always available when $r_{\mathcal S}\ge r_{\mathcal F}$ | exact copy (Theorem 9.2) | none | verify-only (§10.3) |
| **M2 selective** | some anchor group of serving is worse than the pre-fast model $\bar T$ (= $T$ with fast tier zero) by >1% (hypothesis) | exact copy | split teachers: $T$ on current-regime windows, $\bar T$ on regressed anchor groups (§5.3.5) | non-inferiority on future data; anchor groups must improve or hold |
| **M3 compaction** | slot pressure (§4.4) | whitened SVD merge (Theorem 9.4) | teacher = pre-compaction model | non-inferiority |
| **M4 belief** | persistence criterion (§7.2) | zero (new slots, compact and visual) | belief-conditioned teacher → belief-reset student (§7) | non-inferiority, plus belief checks (§7.5) |
| **M0 distil-from-zero** (paper-faithful ablation) | experiments only | zero | full KS objective | non-inferiority |

Why M1 is the default: Theorem 9.2 makes it lossless on **every** input. Distillation can only match the teacher on the support of the sleep data (Theorem 9.5). Its potential advantages are a smaller-norm transfer that drops directions the data never excited, and selective retention (M2). Those advantages are empirical claims, tested in experiment E-KS1 (§11.4). The paper's "upward" framing (a smaller teacher distilled into a larger student) does not apply literally: teacher and student have the **same** active capacity after the reset. What expansion buys is that consolidation writes into **fresh** parameters instead of overwriting $\mathcal S$. That is exactly what the "sleep without expansion" baseline removes.

### 5.3 Knowledge-seeding objective (distilling modes M2–M4, M0)

All terms use $D_z$ (eq. 22) with uniform weights over **both** fixed phone views. There is no wrist view. Frame averaging is within a sequence, then over sequences, exactly as `token_metric` does. Every term is the mean over its own batch, so the weights below are relative scales. The spec §9.7 rule applies: if one term's gradient norm exceeds ten times every other term's for sustained steps, inspect the scaling before tuning any weight.

**5.3.1 Paths.** Take a window $w$: real context frames $z_{0:c-1}$, the issued command ticks for horizon $H$, and the causal motion inputs. Motion comes from the servo in B1, or from that model's own compact prior in B4, built as in `AdaptedPredictor.sequence_inputs` and `belief_conditioning`. Two generated sequences are used:

- the **teacher path** $\tilde z^{T}_{c:c+H-2}$ = teacher's own autonomous predictions, generated under `no_grad`;
- the **student path** $\tilde z^{S}_{c:c+H-2}$ = student's own autonomous predictions, generated under `no_grad`.

For a path $p\in\{T,S\}$, the sequence $\sigma^p=[z_{0:c-1},\tilde z^p]$ is fed **once** to each model with `forward_tokens`. Because attention is frame-causal and fed-back frames are detached (spec §9.2, inherited in `rollout`), the output at position $c-1+k$ is exactly that model's one-step prediction given the prefix of $\sigma^p$. Write $\hat y^{M}_k(\sigma^p)$ for model $M$'s output at step $k$.

**5.3.2 Generalised distillation (GKD analogue).**
\[
L_{\rm off}(\mathcal B)=\frac{1}{|\mathcal B|}\sum_{w\in\mathcal B}\frac{\sum_{k=1}^{H}\gamma^{k-1}D_z\big(\hat y^{S}_k(\sigma^T_w),\ \mathrm{sg}\,\hat y^{T}_k(\sigma^T_w)\big)}{\sum_{k=1}^{H}\gamma^{k-1}},
\]
\[
L_{\rm on}(\mathcal B)=\frac{1}{|\mathcal B|}\sum_{w\in\mathcal B}\frac{\sum_{k=1}^{H}\gamma^{k-1}D_z\big(\hat y^{S}_k(\sigma^S_w),\ \mathrm{sg}\,\hat y^{T}_k(\sigma^S_w)\big)}{\sum_{k=1}^{H}\gamma^{k-1}},
\]
\[
L_{\rm dist}(\mathcal B)=(1-\lambda)L_{\rm off}(\mathcal B)+\lambda L_{\rm on}(\mathcal B),\qquad \gamma=0.95,\ \lambda=0.5\ \text{(hypothesis)}.
\tag{S2}
\]

Notes on (S2):

- $L_{\rm on}$ is on-policy: the student is corrected on **its own** compounding errors. This is the DAgger/GKD argument transplanted to latent rollouts.
- As in GKD, there is no backpropagation through the student's generated frames. Here that rule falls out of the existing detached-rollout contract; it is not an extra approximation.
- Steps whose $D_z$ is `None` (no valid views) are skipped and renormalised exactly as `discounted_rollout_loss` does.
- Divergence choice: a deterministic predictor emits point predictions. GKD's forward/reverse-KL and generalised-JSD distinctions concern full output distributions and **do not apply**. $D_z$ is the same metric the planner and the gate use.

**5.3.3 Data sources.**

- **Real windows** $\mathcal R_{\rm tr}$ (§5.6): $L_{\rm real}=L_{\rm dist}(\mathcal B_{\rm real})$, with $H=2$ (the online horizon, inherited). Use $H=5$ only if the teacher's measured 5-step real error is validated (spec §11.4 rule).
- **Planner-chosen sequences** $\mathcal P$:
  - contexts from $\mathcal R_{\rm tr}$;
  - for each context, a goal (§8.4) and a run of the deterministic `JointCem` with the **teacher** as model, horizon 5;
  - keep the final elite set, the executed-style best, hold, and the shifted previous plan when available;
  - commands go through `JointCem.commands_for`, so they are limiter-feasible by construction;
  - $L_{\rm plan}=L_{\rm dist}(\mathcal B_{\mathcal P})$ with $H_{\mathcal P}=\min(5,H_{\rm val})$, where $H_{\rm val}$ is the largest validated teacher horizon.
  - There are no real targets: these are distillation inputs, never observations (spec §7.6).
- **Ground truth (optional):** $L_{\rm gt}=L_{\rm cmdTF}+0.5L_{\rm AR}$ on $\mathcal R_{\rm tr}$ with real targets (spec eq. 24–25). Weight $\mu_{\rm gt}=0$ by default, because the teacher was already fitted to these windows and re-fitting them in sleep double-counts old data. Use 0.1 as an ablation.

**5.3.4 Action-ranking agreement (the analogue of Learning-to-Imitate).** The paper's LTI rewards the student for imitating the teacher's **sampling behaviour** (§3.3, eq. 3–4). The behaviour of a world model in this system is **which command sequences the planner prefers**, and spec §14.7 already warns that lower token loss can coexist with worse ranking. For a context $c$ and a candidate set $\{U_1,\dots,U_M\}$ ($M=16$):

- the teacher's CEM final population: elites + 4 random members + hold + previous plan;
- terminal costs $J^M_i=D_g(\hat z^{M}_{t+H}(U_i),z_g)$ for $M\in\{T,S\}$, deterministic rollout, $H=5$;
- the student's terminal tokens keep their gradient through the final `forward_tokens` pass (`rollout` already does this).

Let $s_c=\mathrm{MAD}_i(J^T_i)+\epsilon_s$, with $\epsilon_s$ = 1% of the calibration goal scale (a floor). Let the pair set be $\mathcal P_c=\{(i,j):|J^T_i-J^T_j|>m\,s_c\}$, $m=0.05$, which excludes near-ties. Then
\[
L_{\rm rank}=\frac{\sum_{c}\sum_{(i,j)\in\mathcal P_c}\omega_{ij}\,\mathrm{softplus}\!\Big(-\,s_{ij}\frac{J^S_j-J^S_i}{s_c}\Big)}{\sum_c\sum_{(i,j)\in\mathcal P_c}\omega_{ij}},
\quad s_{ij}=\mathrm{sign}(J^T_j-J^T_i),
\quad \omega_{ij}=1+\beta_{\rm top}\mathbb 1[i\in\mathrm{top}_k^T\ \vee\ j\in\mathrm{top}_k^T],
\tag{S3}
\]
with $\beta_{\rm top}=1$, $k=4$ (hypotheses).

- Dividing by $s_c$ makes the loss invariant to the cost scale of each context.
- The top-$k$ weight concentrates on the pairs that decide CEM elites.
- **No reinforcement learning is needed.** Everything here is differentiable in the student's parameters. The paper needs RL because token sampling is discrete; here the only non-differentiable object, the CEM argmin, is replaced by the standard pairwise surrogate.
- Proposition 9.7 shows $L_{\rm rank}/\log 2$ bounds the weighted misordering rate.
- The paper's thresholded similarity reward $r_{\rm abs}$ maps to the evaluation metrics of §5.7: top-1 agreement and teacher-regret.
- The semantic reward model $r_{\rm sem}$ has no analogue and is dropped (§12).

**5.3.5 Selective split teachers (M2).** Let $G^-$ be the anchor groups on which the serving model $T$ is worse than the pre-fast model $\bar T$ on real targets by more than 1%. On windows in $G^-$, the target in (S2) is $\bar T$. Everywhere else it is $T$. This consolidates the fast tier's validated gains while undoing the regressions the per-promotion gate tolerated. It is the explicit cure for the *anchor ratchet* of §9.9.

**5.3.6 Total objective.**
\[
L_{\rm KS}=w_{\rm real}L_{\rm real}+w_{\rm plan}L_{\rm plan}+w_{\rm rank}L_{\rm rank}+w_{\rm ret}L_{\rm ret}+\mu_{\rm gt}L_{\rm gt}+w_{\rm prox}L_{\rm prox}(k^\ast),
\tag{S4}
\]

| Weight | Default (hypothesis) |
|---|---|
| $w_{\rm real}$ | 1 |
| $w_{\rm plan}$ | 0.5 |
| $w_{\rm rank}$ | 0.25 |
| $w_{\rm ret}$ | 0.5 (retention dreams, §8.3; 0 when $K=1$ and no older model exists) |
| $\mu_{\rm gt}$ | 0 |
| $w_{\rm prox}$ | $10^{-4}$ |

$L_{\rm prox}(k^\ast)$ is spec eq. 30 applied to slot $k^\ast$ relative to its initial value, with the $10^{-3}$ RMS floor that zero-initialised tensors need. Weight decay on the slot is 0: decay toward zero would pull a transplanted slot away from the function it carries.

### 5.4 Optimisation defaults (hypotheses)

| Item | Value |
|---|---|
| Optimizer | AdamW, betas (0.9, 0.95), eps $10^{-8}$, global clip 1.0 |
| LR, zero-init slot (M0, M3 refine, M4) | LoRA $10^{-4}$, residual $3\times10^{-4}$, compact $10^{-4}$ (the offline stage-3 rates, because the slot is fresh) |
| LR, copy-init slot (M2) | LoRA $10^{-5}$, residual $3\times10^{-5}$ (the online rates, because the slot starts at the answer) |
| Warmup | 10 steps linear |
| Steps | 100 (the paper's setting, Table 5), early stop on $\mathcal R_{\rm val}$ fidelity after 3 checks without improvement, checking every 20 steps |
| Effective batch | 16 windows; the planner term uses 16 dreams per step; the ranking term uses 4 contexts × 16 candidates every 4th step |
| Precision | BF16 neural compute with FP32 reductions (inherited). The compact model in true FP32 (`check_full_precision`) |

### 5.5 Compute budget per consolidation (2B; hypotheses derived from *measured* SUE)

| Item | Estimate |
|---|---|
| One distillation step | ≈ 1 SUE (student forward/backward) + ≈ 0.4 SUE (teacher no-grad forwards over two paths) ≈ **1.4 SUE ≈ 5.1 s** |
| One ranking step | 4×16 = 64 deterministic rollouts at horizon 5 for teacher and student ≈ **4 SUE**, done every 4th step |
| M2/M3/M4 at 100 steps | ≈ 100×1.4 + 25×4 = 240 SUE ≈ **15 min** at 2B on one GPU at 100% duty |
| M1 | ≈ 1 probe batch ≈ **seconds** |

Peak memory is not measured; predict it before the first run. The base is shared. Teacher, student and pre-fast copies cost about 3× adapter memory, which is negligible. Ranking rollouts must be microbatched (spec §16 KV cost, eq. 39).

### 5.6 Sleep data plan (no leakage)

At sleep start, freeze a list of window references for regime $r$: recent ∪ reservoir($r$) ∪ anchors. Quarantine windows are **excluded**, because their regime is unknown. Windows recorded under a different camera-contract hash are excluded too (phones: exposure and white balance must be locked at commissioning).

- Partition the regime's recent data into non-overlapping temporal blocks of 2 s (inherited block length).
- Assign 15% of blocks to $\mathcal R_{\rm val}$ (hypothesis).
- Remove every training window whose span overlaps a validation block or lies within **one window span (54 raw ticks)** of one.
- $\mathcal R_{\rm val}$ is used for fidelity checks and early rejection **only**.

The authority on generalisation is the **future** gate data collected after wake (§10.3), which no part of the sleep has seen. Final-evaluation sessions (spec §10.2) never enter replay, so they never enter sleep.

### 5.7 Consolidation fidelity metrics (computed on $\mathcal R_{\rm val}$ and on fresh planner sets)

- **Relative fidelity:** $\rho_{\rm fid}=\overline{D_z(S,T)}\,/\,\overline{D_z(T,\text{real})}$, the student–teacher gap as a fraction of the teacher's own real error.
- **Ranking metrics:**
  - Kendall $\tau_b(J^S,J^T)$ per context, averaged;
  - top-1 agreement;
  - normalised teacher-regret $\big(J^T(\arg\min_i J^S_i)-\min_i J^T_i\big)/s_c$.
- **Gate-relevant real metrics:** student vs teacher real loss on $\mathcal R_{\rm val}$ at horizons 1, 2 and 5.

Acceptance to proceed to the gate (hypotheses):

| Metric | Threshold |
|---|---|
| $\rho_{\rm fid}$ | ≤ 0.05 |
| $\tau_b$ | ≥ 0.9 |
| top-1 agreement | ≥ 0.75 |
| regret | ≤ 0.02 |
| real loss, student vs teacher | not worse by more than 1% at any horizon |

For M1 the only requirement is Theorem 9.2's numerical tolerance.

---

## 6. Compaction, merging and deep consolidation

### 6.1 Exact merges first

$\sum_{k}s_kB_kA_k=[s_1B_1\ \cdots\ s_mB_m]\,[A_1;\dots;A_m]$ (Proposition 9.3). Concatenating $m$ slots of rank $r$ is exact, but only into a slot of rank $mr$. With uniform slot ranks this frees nothing. It is useful when moving slots into a higher-rank archive slot, if one is configured.

### 6.2 Whitened SVD (lossy, optimal per layer)

For each projection, collect the input second-moment matrix $C=\mathbb E[xx^\top]$ over $\mathcal R_{\rm tr}$:

- forward pre-hooks on the tiered module, fp32 accumulation;
- $2048^2$ entries per input type at 2B. Q and V share one input; the output projection has its own; so about 256 MB in total, which is fine;
- damp: $C_\epsilon=C+\epsilon I$ with $\epsilon=10^{-4}\,\mathrm{tr}(C)/d$.

With $\Delta=\sum_{k\in\text{merge}}\Delta W_k$ and $M=\Delta C_\epsilon^{1/2}=U\Sigma V^\top$, the merged slot is
\[
\Delta' = U_r\Sigma_rV_r^\top C_\epsilon^{-1/2},\qquad
B'=\tfrac{r}{\alpha}\,U_r\Sigma_r^{1/2},\quad A'=\Sigma_r^{1/2}V_r^\top C_\epsilon^{-1/2}.
\tag{S5}
\]
Report the retained energy $\eta=\sum_{i\le r}\sigma_i^2/\sum_i\sigma_i^2$. If $\eta<0.98$ for any projection, the compaction is flagged lossy and the refinement plus gate decide.

The residual slots are nonlinear (the tanh branch), so there is no closed form. Merge them by distillation from a zero-initialised slot, with teacher = pre-compaction model. Compact correction slots are linear before the tanh, so (S5) applies with $x$ = the transition hidden activation.

### 6.3 Refinement

Run the M3 objective (S4) with teacher = pre-compaction model on $\mathcal R_{\rm tr}$ of the regime, plus the planner set, plus retention dreams on other regimes' anchors, but only for parameters those regimes share. With the per-regime LoRA, other regimes are unaffected by construction.

### 6.4 When to compact

Compact only under pressure (§4.4) or as an operator-requested maintenance sleep. Each compaction is a forgetting opportunity: log $\eta$, fidelity and the gate result.

### 6.5 Deep consolidation (T2→T3)

Deep consolidation is an offline procedure, scheduled by an operator, never automatic:

- train a new offline tier on all retained real data;
- teachers: the current serving model on each regime's data, and the old offline model on offline anchors;
- objective: spec stage-3 losses plus (S2);
- then clear the slow banks;
- validate as a new offline reference under spec §10.6–§10.7 and the A→B→A suite.

This is how the bank's $O(1)$ capacity bound is reconciled with an unbounded lifetime. It is lossy by nature (NL: forgetting is a consequence of compression), and it is the only place the offline reference changes.

---

## 7. Belief consolidation (weights absorb what the belief persistently inferred)

### 7.1 Why, and the §12.3 constraint

The slow context $d_t$ explains changes within the trained dynamics quickly and without gradients. If a condition persists for a long time (for example a software gain change lasting hours), $d$ sits at a persistent offset. Every reset or power cycle then starts from a broadened prior and must re-infer the offset (spec §4 table, §7.4).

Belief consolidation moves the persistent part of what $d$ encodes into weights, then resets the slow prior. Spec §12.3 forbids letting context and weights *both adapt fast*. Belief consolidation respects that rule:

- it happens only in sleep, at most once per sleep;
- it requires a persistence criterion;
- its teacher is frozen;
- the student is gated.

During wake nothing changes: the GRU, the observation model and the covariances stay frozen, and $d$ is inferred as before.

### 7.2 Persistence criterion (all must hold; hypotheses)

Over the last $T_b=10$ min of valid operation in the dominant regime $r$, where the dominant regime held throughout under the spec §8.2 rule:

1. $|\bar\mu_{d,j}|/\bar\sigma_{d,j}>z_b=3$ for some coordinate $j$, using the posterior mean and std averaged over macro steps;
2. the sign of $\mu_{d,j}$ is stable for ≥90% of steps;
3. the offset is reproduced in at least 2 separate episodes or segments;
4. no camera-quality, network or reset fault channel was active (spec §8.1 table). This keeps camera nuisance from being consolidated as dynamics.

### 7.3 Pipeline (order matters: compact first, then visual)

1. **Teachers.**
   - $T_C$ = compact model with the current slow prior $(\mu_d^{\rm prior},P_d^{\rm prior})$;
   - $T_V$ = serving visual model fed beliefs computed by $T_C$.
2. **Compact student** $S_C$ = $T_C$ with:
   - a new slow correction slot, zero-initialised (needs C4);
   - the fast correction reset;
   - slow prior reset to $(0,P_0)$, with $P_0$ = the spec §7.4 initial slow variance (1.0).

   Train only the slot on $\mathcal R_{\rm tr}$ **sequences**, each filtered from its segment start. The loss is the forward KL between moment-matched predictive Gaussians of $y_t$ before assimilation, normalised by dimension:
\[
L_{C}=\frac{1}{d_{\rm obs}}\,\mathrm{KL}\big(\mathcal N(m_T,S_T)\,\|\,\mathcal N(m_S,S_S)\big)+0.5\,\frac16\sum_j\mathrm{Huber}_1\!\Big(\frac{\hat p^S_j-\mathrm{sg}\,\hat p^T_j}{s^{\rm loss}_j}\Big),
\tag{S6}
\]
   with $\mathrm{KL}=\tfrac12[\mathrm{tr}(S_S^{-1}S_T)+(m_S-m_T)^\top S_S^{-1}(m_S-m_T)-d_{\rm obs}+\log\det S_S-\log\det S_T]$ computed with Cholesky solves (spec §7.3).
   - Masked coordinates are removed exactly as in eq. 26.
   - The regime mixture is moment-matched (eq. 20), a stated approximation.
   - The real NLL (spec eq. 26) is added with weight 0.25 as a safeguard (hypothesis). It is evidence, not distillation: the targets are real observations.
3. **Absorption check.** Run $S_C$ from segment starts on $\mathcal R_{\rm val}$ sequences with the reset prior. Record the posterior $|\mu_d|/\sigma_d$ after 2 min.
   - $\le 1$: **full** absorption.
   - Otherwise **partial** absorption: the slow prior is not reset at the end (§7.5), but the visual step may still proceed (Proposition 9.6 explains why that is still consistent).
4. **Visual student** $S_V$ = $T_V$ with a new slow LoRA and residual slot (zero-init) and the fast tier reset. Train with (S2)/(S4), where:
   - the teacher uses beliefs and motion from $T_C$ with the old prior;
   - the student uses beliefs and motion from **$S_C$ with the reset prior, filtered from segment starts**.

   This consistency condition is what prevents double counting (Proposition 9.6): the student is trained on exactly the beliefs it will receive after deployment.
5. **Candidate.** $(S_C,S_V)$ plus the new slow prior form one version (C4). Gate: non-inferiority on future data for the joint error, compact NLL (with coverage not materially worse, inherited), and visual token loss, plus §7.5.

### 7.4 Installing a belief-consolidated version

At promotion the live belief must be **re-initialised**:

- slow components from the reset prior $(0,P_0)$;
- reconstruct over ≥20 macro frames from the event log (spec §7.4 burn-in), not just the 2 s fixed-lag window. `FixedLagBelief.rebuild` alone would carry the old $d$ posterior inside its oldest snapshot.

Spec §13.3 says "do not reset the slow belief blindly to zero". This reset is **not blind**: it is the validated semantics of the candidate, which was trained and gated with exactly this reset. Record it as a belief-consolidation event.

### 7.5 Belief-specific acceptance and failure handling

| Check | Threshold (hypothesis) | If it fails |
|---|---|---|
| Absorption | posterior $\vert\mu_d\vert/\sigma_d\le1$ on the 2 min post-reset future gate data | do not reset the prior (keep the old prior); a visual student trained under the reset prior is then inconsistent, so **reject the candidate** |
| Compact NLL | non-inferior, $\delta_{\rm NI}=1\%$ | reject |
| Joint prediction | not worse than 2% (inherited spec §13.2) | reject |
| Re-inference on regime return | after a later A-return, the time to re-reach steady $d$ is not longer than before | flag; candidate for rollback |

Honest limit: the correction head adds to the transition's pre-tanh output (joint residual and $\xi$ drive) only. It **cannot** reproduce the part of $d$'s effect that acts through the frozen observation matrix ($H\,d$ in eq. 10). It can mimic some of it by driving $\xi$ to a persistent value ($\xi^\ast\approx 0.1\tanh(\cdot)/(1-e^{-0.02})\approx5\tanh(\cdot)$ at 5 Hz), but that repurposes a "fast" coordinate for persistent content. Such repurposing is detectable as a large mean $|\xi|$ and must be logged. Consolidating visual-pathway offsets needs the observation model to be trainable in sleep, which this specification does **not** allow (spec §9.8: online frozen; sleep follows the online rule).

### 7.6 B1 and "belief not ready"

In B1 (no belief) and until C4 is resolved, M4 is disabled, and so is the hindsight dreaming of §8.2 (it needs a belief). Consolidation is then M1/M2/M3 on visual adapters only.

---

## 8. REM stage 2: dreaming that is safe for a world model

### 8.1 What a dream is, and what it is never

**A dream is** a pair (real context window, synthetic issued-command sequence) with **teacher-produced** latent targets and a weight. It is a distillation input.

**A dream is never:**

- an observation: no belief assimilation, no `ReplayBuffer` entry (C12), no anchor, no gate score, no calibration data, no change-monitor input (spec §7.6);
- a target for the compact likelihood. Dream frames must never enter eq. 26, because that would make a model's own prediction into evidence;
- labelled by the student itself. Proposition 9.10: a deterministic self-teacher gives exactly zero gradient, so it is a no-op. Iterated self-labelling is the collapse mode the paper itself cites for on-policy self-distillation.

### 8.2 Which teachers can make a dream informative

A dream improves the student only if its teacher knows something the student does not.

| Dream kind | Teacher | Information source | Use |
|---|---|---|---|
| **D-hind (hindsight)** | The consolidated model, with the slow context at the branch point replaced by a **smoothed** estimate $\tilde d_t$ = the posterior mean of $d$ after assimilating the segment up to $\min(t+T_h,\ \text{segment end})$, $T_h=60$ s | Real **future** observations of the same segment. Because $F_d$ is the identity with small process noise, the later posterior is a valid smoothed estimate of $d_t$ under the model | Improvement (REM proper) |
| **D-ret (retention)** | Frozen pre-sleep model, or the offline reference, on **other-regime / old** contexts | Older real data, already in those weights | Protects old function on counterfactual commands (LwF / generative-replay analogue). Used inside NREM as $L_{\rm ret}$ |
| **D-rep (repair)** | Pre-compaction model | The pre-compaction weights | Repairs compaction loss where the planner queries. Used inside M3 |
| ~~D-self~~ | the student itself | none | **forbidden** (no-op or collapse) |

**Mandatory teacher-advantage pre-check for D-hind.** On $\mathcal R_{\rm val}$, compare against the real targets. The hindsight teacher must beat the consolidated student by ≥1% relative token loss, with a block-bootstrap 90% upper bound <0. If it does not, the real data says there is no improving signal: **skip REM for this sleep**. This pre-check makes REM falsifiable and cheap to switch off.

### 8.3 Dream generator (command families)

Contexts are drawn from $\mathcal R_{\rm tr}$ for D-hind, and from other-regime anchors or reservoirs for D-ret. For each context, sample command sequences from a mixture of families. Every sequence passes through `JointCem.commands_for` (the shared limiter and joint limits), so every dream command is executable.

| Family | Share (hypothesis) | Construction |
|---|---|---|
| planner | 1/3 | teacher-CEM elites for a sampled goal (§8.4) |
| perturbation | 1/3 | logged issued increments + Gaussian noise of std 0.2 in normalised increment space |
| excitation | 1/6 | reversals and steps per joint at ≤50% of $\Delta p_{\max}$ (the identifiability-rich motions of spec §17.3) |
| hold | 1/6 | all-zero increments |

Horizon $H_d=2$ by default; 5 only if the teacher's real 5-step error is validated.

**Competence filter.** Drop a dream if the disagreement between teacher and offline reference, $D_z(T,\mathcal O)$, exceeds the 90th percentile of that disagreement on real windows of the same regime. Such a dream is out of distribution for every model, and its label would be a guess.

The planner share is capped at 1/3 because planner-chosen sequences are exactly where a teacher can be **exploitable** (spec §17.5). Distilling them spreads any exploit into slow memory.

Random cross-expert mixing (the paper's "each router picks a random extra expert") is **not used**: mixing regime adapters has no physical meaning for one arm (§12).

### 8.4 Goals for planner dreams

Goals come from:

- real future keyframes of the same episode, 1–3 s ahead, which are reachable by construction;
- the stored goal images of past tasks under the same camera contract.

Weights are 0.5 per fixed phone view (inherited config), with the fixed calibration scale. Goals are never generated images.

### 8.5 Prioritisation (capped and corrected, spec §11.3)

Score each dream by its current distillation loss $g_i$. This is the analogue of the paper's gradient-importance score, using the loss rather than the gradient norm, which costs one backward pass less. Sample with
\[
p_i=(1-\varepsilon)\frac{\min(g_i,\kappa\,\mathrm{med}(g))^{a}}{\sum_j\min(g_j,\kappa\,\mathrm{med}(g))^{a}}+\frac{\varepsilon}{N},\qquad
w_i=\frac{(Np_i)^{-\beta}}{\max_j (Np_j)^{-\beta}},
\tag{S7}
\]
with $\kappa=3$, $a=0.6$, $\varepsilon=0.25$, $\beta=1$ (hypotheses). The paper's "b random samples for diversity" maps to the $\varepsilon$ floor. With $\beta=1$ the weighted objective equals the uniform dream objective up to a constant (Proposition 9.11). Prioritisation applies to dreams only, never to real replay: version 1 forbids priority on real data because a large real error can be a sensor fault.

### 8.6 REM training and acceptance

1. Activate a **separate** dream slot $k_d$ (zero-init) on top of the accepted-or-pending consolidated model. $L_{\rm REM}=0.5\,L_{\rm dist}(\text{D-hind}, H_d)+0.5\,L_{\rm dist}(\mathcal R_{\rm tr},T{=}\text{hindsight teacher})+w_{\rm ret}L_{\rm ret}$. At least 50% of the loss is real-context distillation. Steps: 50, batch 16, the zero-init LRs of §5.4. **Never chain**: the REM teacher is fixed for the sleep and is never a dream-trained model.
2. Check on $\mathcal R_{\rm val}$ against **real** targets. Proceed only if real loss improves ≥1% with a block-bootstrap upper bound <0, and no anchor group regresses >2%.
3. Candidate $C_{\rm dream}=C_{\rm cons}+k_d$. It is gated with the **existing superiority gate** on future real transitions (spec §13.2), **after** $C_{\rm cons}$ has been accepted. There is only ever one challenger at a time.
4. If rejected, $k_d$ is reset and freed. Consolidation is unaffected.

**Generator learning (the paper's ReST-EM over dreams) is version 2.** A per-dream isolated LoRA copy plus evaluation costs tens of SUE per dream at 2B (SEAL reports 30–45 s per self-edit evaluation for its LLM setting). The affordable form is a bandit over the **families** of §8.3, rewarded by the $\mathcal R_{\rm val}$ improvement of family-specific isolated slots, updated multiplicatively. It is off in version 1.

### 8.7 Honest summary of what dreaming can and cannot do

Sleep has no new real data. Dreams can only:

1. move information that already exists (real future observations via smoothing, older weights) onto counterfactual command inputs the planner will query;
2. protect old behaviour on such inputs;
3. repair compaction loss.

Dreams cannot teach contact dynamics, object properties or anything the cameras and joint sensors never measured. This matters especially here: there is no force or tactile sensing and no wrist view. Every dreaming gain must therefore appear on **future real** transitions, and the "sleep without dreaming at matched compute" baseline (§11.2) is the deciding comparison.

---

## 9. Mathematics: statements and proofs

The setting: real arithmetic unless stated. "Function" means the map from (context, commands, motion, belief) to predicted tokens.

### 9.1 Theorem (zero-initialised slot preserves function exactly)

*Statement.* Let slot $k$ have $B_k=0$. Then $\Delta W_k=\frac{\alpha}{r}B_kA_k=0$, and for every input the converted projection output equals the output without the slot. For a residual slot with $b_{o,k}$ (weight and bias) $=0$, the branch output $b_{o,k}\tanh(\cdot)=0$. Hence $\mathrm{LN}_0(z^0+\cdots+0)=\mathrm{LN}_0(z^0+\cdots)$.

In IEEE floating point, if every activation is finite, `F.linear(F.linear(x,A),0)` returns exact zeros and $y+0=y$ bit for bit. The sign of an exact zero can differ ($-0+0=+0$), which is numerically irrelevant. So outputs are **bit-identical**.

At $B_k=0$ the gradient is $\nabla_{A_k}L=\frac{\alpha}{r}B_k^\top(\cdot)=0$, so only $B_k$ moves on the first step (already noted in `lora.py`). For residual slots, $\nabla_{m_k}L\propto b_{o,k}=0$ likewise.

*Proof.* Direct substitution; $0\cdot v=0$ for finite $v$. ∎

*Consequence.* Pre-allocation plus masking can never change serving behaviour by itself. Activation still bumps the version (I3).

### 9.2 Theorem (exact transplant)

*Statement.* Let the fast slot have rank $r_F$, scale $\alpha_F/r_F$ and factors $B_F,A_F$. Let the target slow slot have rank $r_S\ge r_F$ and scale $\alpha_S/r_S$. Set
\[
A_S=\begin{bmatrix}A_F\\ R\end{bmatrix},\qquad B_S=\frac{\alpha_F\,r_S}{\alpha_S\,r_F}\,\big[B_F\ \ 0\big],
\]
where $R$ is any matrix, for example a fresh seeded init. Then $\Delta W_S=\Delta W_F$. After resetting the fast slot ($B_F\leftarrow0$), the total (S1) is unchanged. For residual slots of equal architecture, copying $(m_F,b_{o,F})$ is exact. For the compact correction (linear before the tanh), the same rank argument applies.

*Proof.* $\frac{\alpha_S}{r_S}B_SA_S=\frac{\alpha_S}{r_S}\cdot\frac{\alpha_F r_S}{\alpha_S r_F}\big(B_FA_F+0\cdot R\big)=\frac{\alpha_F}{r_F}B_FA_F$. ∎

*Remarks.*
- Use a **random** $R$, not zero. With $R=0$ the padded rows of $A$ and the padded columns of $B$ are both zero, so their gradients ($\propto$ each other) are zero and the padded rank is dead under any later refinement.
- In floating point, the two branches are summed in a different order than one branch, so equality holds up to reassociation. The acceptance tolerance: relative $D_z(S,T)\le10^{-6}$ (fp32) or $\le10^{-2}$ (bf16, hypothesis). The verify-only gate (§10.3) checks it on a fixed probe batch.

### 9.3 Proposition (exact merge by concatenation)

$\sum_{k=1}^m s_kB_kA_k=\big[s_1B_1\cdots s_mB_m\big]\big[A_1;\cdots;A_m\big]$, a rank-$\le\sum r_k$ factorisation. A merge is lossless if and only if the target rank is at least $\mathrm{rank}(\sum_k s_kB_kA_k)$, which is at most $\sum_k r_k$. ∎

### 9.4 Theorem (optimal layer-wise compaction under the data distribution)

*Statement.* For a linear map $\Delta$, inputs $x\sim\mathcal D$ with $C=\mathbb E[xx^\top]\succ0$, and target rank $r$:
\[
\min_{\mathrm{rank}\,\Delta'\le r}\ \mathbb E\|(\Delta-\Delta')x\|^2
\]
is attained at $\Delta'=[\Delta C^{1/2}]_r\,C^{-1/2}$ (truncated SVD). The minimum equals $\sum_{i>r}\sigma_i^2(\Delta C^{1/2})$.

*Proof.* $\mathbb E\|(\Delta-\Delta')x\|^2=\mathrm{tr}\big((\Delta-\Delta')C(\Delta-\Delta')^\top\big)=\|(\Delta-\Delta')C^{1/2}\|_F^2$. The map $\Delta'\mapsto\Delta'C^{1/2}$ is a rank-preserving bijection because $C^{1/2}$ is invertible. So the problem is $\min_{\mathrm{rank}\,N\le r}\|M-N\|_F$ with $M=\Delta C^{1/2}$, which Eckart–Young solves by $N=[M]_r$. ∎

*Remarks.*
- With singular $C$, use $C_\epsilon$. Directions with no data energy are then truncated first, which is the intended "forget only what the data never excites".
- Layer-wise optimality is not end-to-end optimality: errors propagate through later layers and the residual. Hence the distillation refinement and the gate.
- This is the same whitening idea as the ASVD and SVD-LLM compression methods (see research notes).

### 9.5 Theorem (what distillation preserves, to first order, and only under conditions)

*Setting.* Write the deployed prediction as $g(w;\theta)$, with $\theta_0$ the student's frozen parameters. The teacher is $\theta_T=\theta_0+E_F\delta_F$ (fast tier embedded by $E_F$). The student is $\theta_S=\theta_0+E_S\delta_S$ (new slot embedded by $E_S$). Linearise $g(w;\theta_0+u)=g_0(w)+J(w)u+\rho(w,u)$ with $\|\rho\|\le\frac{L_2}{2}\|u\|^2$ on the region considered. Distil with squared error over a finite set $\mathcal D$ (L1 behaves similarly near the optimum but has a non-unique minimiser). Let $\Phi_{\mathcal D}$ stack $J(w)E_S$ and $y_{\mathcal D}$ stack $J(w)E_F\delta_F$ for $w\in\mathcal D$.

*Statement.* Assume:
- **(R) representability:** there is $\bar\delta$ with $J(w)E_S\bar\delta=J(w)E_F\delta_F$ for all $w\in\mathcal D\cup\{w'\}$;
- **(I) identifiability toward $w'$:** $\ker\Phi_{\mathcal D}\subseteq\ker\big(J(w')E_S\big)$.

Then **every** minimiser $\delta_S^\ast$ of the linearised distillation loss satisfies $J(w')E_S\delta_S^\ast=J(w')E_F\delta_F$. So
\[
|g(w';\theta_S)-g(w';\theta_T)|\le\tfrac{L_2}{2}\big(\|E_S\delta_S^\ast\|^2+\|E_F\delta_F\|^2\big).
\]
That is, preservation holds **to first order**. If (I) fails, gradient descent from zero converges to the minimum-norm minimiser $\delta^\ast=\Phi_{\mathcal D}^+y_{\mathcal D}$. Its error at $w'$ is $-J(w')E_SP_{\ker\Phi_{\mathcal D}}\bar\delta$, which is nonzero whenever the part of the fast knowledge not excited by $\mathcal D$ matters at $w'$. Adam does not exactly preserve minimum-norm selection, so the remainder is then uncontrolled.

*Proof.* Under (R) the minimum value is 0. The minimiser set is $\bar\delta+\ker\Phi_{\mathcal D}$. For $v\in\ker\Phi_{\mathcal D}$, (I) gives $J(w')E_Sv=0$. The second-order remainder bound follows from the Taylor bound applied to both models. For the minimum-norm statement: GD on least squares from 0 stays in $\mathrm{row}(\Phi_{\mathcal D})$, giving $\delta^\ast=\Phi_{\mathcal D}^+\Phi_{\mathcal D}\bar\delta=\bar\delta-P_{\ker}\bar\delta$. ∎

*Consequences for the design.*
1. **Exact transplant** satisfies (R) and (I) trivially, for every $w'$ ($E_S=E_F$, $\delta_S=\delta_F$). Distillation does not.
2. The sleep data must excite the directions the deployment uses. That is the formal reason the objective includes **on-policy rollouts** (student-generated prefixes) and **planner-chosen command sequences**, not only logged real windows: the planner queries counterfactual commands that logged data may never have excited.
3. "Consolidation then reset preserves function" is therefore true **to first order, on the excited subspace**, under (R) and (I), and with a second-order error proportional to the squared adapter magnitude. Nothing is guaranteed off that subspace. The future-data gate is the empirical check.

### 9.6 Proposition (belief consolidation: double counting and the consistency cure)

*Setting.* The visual prediction depends on weights and belief: $f(\theta,b)$. Teacher: $(\theta_T,b_T)$ with $b_T$ computed by $T_C$ under the old prior, whose slow component carries an offset $\bar d$. Suppose the student $\theta_S$ were trained against beliefs with $d=0$, but deployed with beliefs $\hat b$ from a filter that still infers $\hat d\ne0$. This happens if the compact model did not absorb the offset, or if the live belief was not re-initialised.

*Statement.* To first order, the deployment error is
\[
f(\theta_S,\hat b)-f(\theta_T,b_T)\approx\partial_df(\theta_S,b)\,\hat d.
\]
When $\hat d\approx\bar d$ this equals the full offset effect again: the offset is represented in **both** weights and belief. If instead the student is trained on beliefs $b_{S}$ produced by the **deployed** compact model $S_C$ under the **deployed** prior and initialisation (§7.3 step 4, filtering from segment starts), the training-input distribution equals the deployment-input distribution. In that case the first-order error on the excited subspace is controlled by Theorem 9.5, **whatever** the absorption level of $S_C$.

Absorption therefore decides only whether resetting the **prior** is justified: if $S_C$ still infers $\hat d\ne0$, a reset prior must re-infer it every episode, which is no worse than before but gains nothing. It does not decide whether the visual student is consistent. Because $F_d$ is the identity, a mismatched live $d$ does not decay by itself; it moves only through innovations under process variance $q_d$. Hence the mandatory re-initialisation at installation (§7.4).

*Proof.* A first-order Taylor expansion in $b$ around $b_T$, combined with the training identity $f(\theta_S,b_S)\approx f(\theta_T,b_T)$ on $\mathcal D$. The consistency claim is Theorem 9.5 applied to the joint input $(w,b_S(w))$. ∎

### 9.7 Proposition (ranking loss bounds misordering; regret bound)

1. For every pair, $\mathrm{softplus}(-x)>0$, and $\mathrm{softplus}(-x)\ge\log2$ when $x\le0$. A pair is misordered by the student exactly when $x=s_{ij}(J^S_j-J^S_i)\le0$. Hence $L_{\rm rank}\ge\log2\cdot\frac{\sum\omega_{ij}\mathbb 1[\text{misordered}]}{\sum\omega_{ij}}$: the weighted misordering rate is at most $L_{\rm rank}/\log2$.
2. If $\max_i|J^S_i-J^T_i|\le\varepsilon$ on a candidate set, then with $\hat\jmath=\arg\min J^S$ and $i^\ast=\arg\min J^T$: $J^T_{\hat\jmath}\le J^S_{\hat\jmath}+\varepsilon\le J^S_{i^\ast}+\varepsilon\le J^T_{i^\ast}+2\varepsilon$.

This is spec §17.4's $2\varepsilon_J$ bound with the teacher as the reference. It bounds regret **with respect to the teacher**, not true task regret; the true quantity needs real outcomes. ∎

### 9.8 Proposition (the existing gate rejects every exact consolidation)

For an exact transplant, the challenger and serving produce equal predictions up to reassociation, so the per-transition loss differences are 0 (or $\pm$ rounding noise). `evaluate_gate` needs a relative improvement $\ge 0.02$, but the observed value is $0$ (or $|\cdot|\ll 0.02$). The candidate is therefore rejected with probability 1 (with rounding noise: with probability ≥ $1-\Pr(\text{noise}\ge2\%)\approx1$).

A non-inferiority test is the correct decision rule for a candidate whose **purpose is preservation**:
\[
\text{accept iff}\ \ \mathrm{UCB}_{90\%}\!\Big[\tfrac{\overline{\ell_C-\ell_V}}{\overline{\ell_V}}\Big]\le\delta_{\rm NI}=0.01.
\]
Use the same block bootstrap (block 2 s) as `block_bootstrap_upper` for the UCB. ∎

### 9.9 Proposition (anchor ratchet)

The per-promotion rule allows each anchor group to regress by up to 5% relative to the **current** serving model. After $n$ accepted promotions the cumulative regression relative to the original can reach $(1.05)^n-1$: 63% at $n=10$.

Requirement: the sleep gate (and, recommended, every gate) adds a **cumulative** check against fixed references. Each anchor group's loss must stay within 5% of its value under the offline reference, or under the version at which that group was last certified. Mode M2 removes accumulated regressions during sleep. ∎

### 9.10 Proposition (self-distillation of a deterministic model is a no-op)

Let the student slot be zero-initialised on top of the teacher's parameters, so $f_S(\cdot;\phi_0)\equiv f_T$. For any distance $D$ with $D(a,a)=0\le D(a,b)$:

- $\phi_0$ is a global minimiser of $\mathbb E\,D(f_S(w;\phi),\mathrm{sg}f_T(w))$;
- for $D_z$ (L1 after LN0), autograd returns $\mathrm{sign}(0)=0$, so the gradient is **exactly** zero;
- under identical code paths, bf16 outputs are bitwise equal, so this holds numerically too.

Hence no optimizer moves. A stochastic LM differs: the cross-entropy of a model on its **own samples** is not minimised at the current parameters, because SFT on samples sharpens the distribution. That is why LLM self-training "does something" and a deterministic latent predictor's does not. Dreaming therefore requires a teacher that differs from the student (§8.2). ∎

### 9.11 Proposition (capped, corrected dream prioritisation)

With $p_i$ from (S7) and $\beta=1$:
\[
\mathbb E_{i\sim p}[(Np_i)^{-1}\ell_i]=\sum_ip_i\frac{\ell_i}{Np_i}=\frac1N\sum_i\ell_i,
\]
the uniform objective. Normalising by $\max_j(Np_j)^{-1}$ only rescales it. The floor $p_i\ge\varepsilon/N$ bounds every weight by $1/\varepsilon$, so the variance of the estimator is finite. The cap $\kappa\,\mathrm{med}(g)$ bounds how strongly any single dream can be over-sampled. ∎

### 9.12 Non-guarantees

There is no theorem here of non-forgetting, convergence, or task improvement. All of the following are first-order or layer-wise statements under explicit assumptions:

- Theorem 9.5's constants ($L_2$, the excited subspace) are unknown for a 2B transformer;
- Theorem 9.4 is per layer;
- the gate is a heuristic under adaptive reuse (spec §13.2).

The experiments of §11 are the evidence.

---

## 10. Wake/sleep scheduler

### 10.1 Triggers (evaluated only at planning boundaries)

| Id | Trigger | Default (hypothesis) | Sleep type |
|---|---|---|---|
| T-a | builder updates since the last consolidation $\ge U_{\max}$ | 200 (≈2 h at 2B) | full NREM (+REM if eligible) |
| T-b | accepted promotions since the last consolidation $\ge P_{\max}$ | 4 | full NREM |
| T-c | **consolidate before overwrite**: the CUSUM change monitor (spec eq. 21) alarms and stays alarmed for 5 valid observations, with joint-mismatch evidence and no camera or network fault (spec §8.1 table), while $K=1$ | — | **micro-sleep**: M1 only (seconds). Run it *before* the builder trains on the new condition; the builder is paused from the alarm until the micro-sleep completes |
| T-d | operator idle window (charging, no task, night) | — | full NREM + REM + optional compaction |
| T-e | slot pressure $F_r\le\rho$ | $\rho=1$ | compaction sleep (§4.4) |
| T-f | belief persistence criterion (§7.2) | — | M4 |
| — | nothing validated to consolidate: no accepted promotion since the last sleep **and** the fast tier is functionally ≈ zero (anchor $D_z$ between serving and $\bar T$ below the no-shift noise floor) | — | **skip** (consolidating nothing just costs a gate) |

**Inhibitors** (any one blocks sleep entry):

- a plan is active or committed;
- a fixed-lag reconstruction is in progress;
- a challenger is mid-gate (drain it first);
- the camera-contract hash changed and has not been re-commissioned;
- an unresolved numerical fault (`runner.faults` not empty);
- fewer than 512 valid windows in the regime since the last sleep;
- the stop/hold policy is not commissioned (`stop_policy_commissioned: false` blocks any sleep that needs a hold);
- the minimum interval since the last full sleep (30 min) has not passed. T-c is exempt.

### 10.2 State machine

```
WAKE ──trigger@boundary──► DRAIN ──► HOLD_ENTER ──► PREP ──► NREM ──► NREM_CHECK ──► REM? ──► REM_CHECK
  ▲                                                                                     │
  │                         (any failure or budget overrun: ABORT → discard candidates, builder resumes unchanged)
  └── WAKE ◄── PROMOTE/REJECT ◄── GATE(C_dream, superiority) ◄── PROMOTE ◄── GATE(C_cons) ◄── RESUME ◄┘
```

- **DRAIN:**
  - stop `maybe_snapshot`;
  - resolve or discard the current challenger;
  - finish the current plan (Profile A: the block executes, then hold);
  - pause builder updates. Credits keep being earned but are capped at the backlog of 100; discards are counted as usual.
- **HOLD_ENTER:**
  - command the commissioned hold; set movement disabled;
  - the local supervisor rejects any plan whose version or expiry predates the hold;
  - perception and filtering continue at ≥1 Hz, because they have priority over learning (spec §15.5); raw data is logged; windows are admitted normally.
- **PREP:**
  - pin the teacher (version and digest);
  - freeze the window-reference list; build $\mathcal R_{\rm tr}$/$\mathcal R_{\rm val}$ (§5.6);
  - collect the compaction covariances if needed;
  - check budgets; allocate the target slots in clones only.
- **NREM:** in order: M3 (if pressure) → M1/M2 → M4 (if eligible), each producing a candidate in a clone.
- **REM:** only if eligible (B4, C4 resolved, teacher-advantage pre-check passed).
- **RESUME:**
  - release the hold. Serving is still the pre-sleep version, so the belief does not need a rebuild yet;
  - candidates are saved to disk as **unvalidated** files outside the validated manifest.
- **GATE and PROMOTE:** see §10.3 and §10.4.

### 10.3 Gating sleep candidates

| Candidate | Decision rule | Data |
|---|---|---|
| M1 transplant | **verify-only**: all four of (a) finite parameters, (b) relative $D_z(C,V)$ on a fixed probe batch within the Theorem 9.2 tolerance, (c) action-probe ratio ∈ [0.99, 1.01] and cosine ≥ 0.999, (d) slot invariants I1–I3 | probe batch, no future data |
| M2, M3, M4 lossy | **non-inferiority** (Proposition 9.8): UCB$_{90}$ of relative future loss ≤ +1%, using the existing block bootstrap. Plus anchor groups ≤ +2% vs serving **and** ≤ +5% cumulative vs fixed references (Proposition 9.9); M2 additionally requires the regressed groups $G^-$ to improve. Plus horizon-2 future loss non-inferior; inherited joint/compact/coverage checks (spec §13.2); ranking-agreement checks of §5.7 on fresh planner sets at future contexts | ≥128 future valid macro transitions (inherited) + 512 anchors |
| C_dream | **superiority**: the existing `evaluate_gate` unchanged (≥2% improvement and UCB < 0), plus the cumulative anchor check | same |

During a sleep-candidate gate the builder **stays paused**. This avoids forking the builder lineage between "candidate accepted" and "candidate rejected". The pause lasts at least 25.6 s of valid data plus scoring, which is short. Report how many times the gate was queried (spec §13.2).

### 10.4 Promotion, builder, reference, caches and belief

On acceptance of $C_{\rm cons}$, at a planning boundary:

1. `VersionStore.save(C_cons, builder_state=None)` with metadata: schema; slot map; sleep id; mode; teacher version and digest; and, for M4, the compact state and slow prior (C4).
2. `hub.previous, hub.serving = hub.serving, challenger`.
3. **New fast tier:**
   - `builder = clone_role(C_cons)` (fast tier already reset in the candidate);
   - `reference = clone_role(C_cons)`;
   - a **new** `OnlineBuilder(builder, reference, config)` with a fresh optimizer and a **post-sleep warmup of 20 updates** (hypothesis; the inherited 100 updates would take about an hour at 2B);
   - replace `hub.builder`, `hub.reference`, `runner.builder`; set `runner.updates_at_snapshot=0`; reset the rollback monitor.
4. **Caches and belief:**
   - all KV and context caches are invalid by version (I3);
   - for M1–M3, `FixedLagBelief.rebuild(compact)` if the compact model changed, otherwise no belief change;
   - for M4, re-initialise per §7.4;
   - planning resumes only after the belief has caught up to the command log (spec §13.3 step 6).
5. If $C_{\rm dream}$ exists, it becomes the next challenger (superiority gate).

On **rejection** of $C_{\rm cons}$:

- discard it;
- **unpause the original builder unchanged**: its fast tier was never reset;
- log the reasons.

A rejected sleep costs compute only.

**Rollback across a sleep** restores the target version's full state: slot map, compact correction, slow prior, and the builder state saved with that version. The pre-sleep version must therefore be saved with its builder state before the sleep (`VersionStore` already supports `builder_state`). Physical actions and replay are unchanged (inherited spec §13.4).

### 10.5 Compute budgets (2B, one GPU; hypotheses from *measured* SUE)

| Phase | Budget | Wall time at 100% duty |
|---|---|---|
| PREP (splits, covariances, CEM planner sets: 64 contexts × 4 iterations × 128 candidates at H=5 with the teacher, microbatched) | ≤ 60 SUE | ≤ 4 min |
| NREM M1 | ≤ 1 SUE | seconds |
| NREM M2/M3/M4 (each) | ≤ 240 SUE | ≤ 15 min |
| NREM_CHECK (fidelity + ranking on $\mathcal R_{\rm val}$) | ≤ 20 SUE | ≤ 1.5 min |
| REM (pre-check + 50 steps + check) | ≤ 120 SUE | ≤ 7 min |
| **Hard cap per full sleep** | **≤ 500 SUE ≈ 30 min** | abort above 1.5× any phase budget |

Sleep compute goes on a separate `SleepLedger`, never on the wake ledger (C9). For 300M, re-derive every budget from that model's measured SUE; it has not been measured.

### 10.6 Crash, interruption and safety

- A sleep writes candidates atomically to an `unvalidated/` area. After a crash, restart from the validated manifest; unvalidated candidates are ignored. The robot stays in hold until the normal start procedure (spec §4 power-cycle row).
- Human intervention or an emergency stop during sleep aborts the sleep.
- Sleep never sends motor commands. The learning worker has no motor privilege (inherited spec §15.1).

---

## 11. Evaluation protocol and acceptance gates

### 11.1 Benchmarks

**Recurrence (primary).** A→B→A and A→B→C→A, built from:

- software command perturbations: `evaluation/perturbation.py` gain and delay shifts, the simulated arm first, then the real arm with bounded software shifts only (spec §19.2);
- safe object variations on the real arm.

Each phase lasts $T_{\rm ph}$ = 30 min of valid operation in simulation and 20 min on the arm (hypotheses). Sleeps happen wherever the triggers fire. One condition, **"sleep at phase end"**, forces a full sleep after each phase so the conditions can be paired.

**Nominal no-shift.** Sleep must not degrade it (spec §19.2 family 1).

**Belief persistence.** A long constant gain shift (≥30 min), then a power cycle and resumption (tests M4).

### 11.2 Baselines (identical checkpoints, data, limits and planner budgets)

| Id | System | Isolates |
|---|---|---|
| B1 | legacy online adaptation (spec B1), no tiers | the reference |
| **B1-T** | tiered parameterisation (§3.3), no sleep ever | the reparameterisation alone |
| B1-R | periodic fast-tier reset with no consolidation | plasticity reset ≠ consolidation |
| S-copy | sleep with M1 only + reset (no distillation, no REM) | the bookkeeping/reset effect of consolidation |
| S-full | M1/M2/M4 + KS objective + REM | the full proposal |
| **S-noexp** | sleep without expansion: distil into the most recent existing slow slot (fixed capacity) | the paper's "w/o expansion" ablation |
| **S-nodream** | S-full without REM | dreaming's contribution |
| S-norank / S-nopolicy / S-noplan | KS without $L_{\rm rank}$ / with $\lambda=0$ / without $L_{\rm plan}$ | each KS term |
| S-distill0 | M0 (paper-faithful zero-init distillation) instead of M1/M2 | transplant vs distillation (E-KS1) |
| S-nobelief | S-full without M4 (B4 systems) | belief consolidation |
| **MC-B1** | B1 that spends exactly the sleep's GPU-seconds on extra wake-style updates during the same holds | **matched compute** |
| MC-nodream | S-nodream plus REM's GPU-seconds spent on extra NREM real distillation | matched-compute test of dreaming |
| B6, B8 (spec) | episode reset / uniform replay | continuity with spec §19.1 |

### 11.3 Metrics

**Retention and forgetting.** Each phase has a frozen evaluation set: sessions never in replay. Let $L_X(t)$ be the evaluation loss on condition X at time $t$, with $L_X^\ast$ its best value during its own phase and $t_X^{\rm end}$ the end of X's phase.

| Metric | Definition |
|---|---|
| Retention ratio | $R_{A|B}=L_A(t_B^{\rm end})/L_A(t_A^{\rm end})$ |
| Forgetting | $F_A=L_A(t_B^{\rm end})-L_A^\ast$ |
| Backward transfer | $L_A(t_A^{\rm end})-L_A(t_B^{\rm end})$ |

**Recurrence recovery.**

- the prequential loss at the first 30 s of A-return;
- **time-to-recover**: wall-clock and valid transitions until the prequential loss is within 5% of the end-of-A level.

**Adaptation.**

- B-phase prequential AUC;
- time-to-adapt on B;
- final-time performance (spec §19.3: final-time and AUC are primary; best-ever is secondary).

**Ranking.**

- teacher-free, in simulation, where counterfactual ground truth exists: Kendall $\tau$ between model costs and true costs, and true regret on fixed probe contexts × candidate sets;
- teacher-relative metrics of §5.7 for every sleep.

**Sleep internals.** Per sleep: mode; $\rho_{\rm fid}$; $\eta$; gate outcome; slots used; GPU-seconds; hold duration; teacher-advantage pre-check result; absorption level.

**Control.** Task success, recovery, interventions (spec §19.3), on the arm only after G0–G3.

**Systems.** Hold time per hour, sleep duty, deadline misses, bytes, slot occupancy over time (must stay bounded).

### 11.4 Experiments (preregistered questions)

| Id | Question | Comparison |
|---|---|---|
| E-KS1 | Does distillation beat exact transplant? | S-distill0 vs S-copy, on retention and on off-distribution planner contexts (Theorem 9.5) |
| E-EXP | Does expansion matter? | S-full vs S-noexp |
| E-REM | Does dreaming help at matched compute? | S-full vs MC-nodream |
| E-BEL | Does M4 shorten re-inference after resets? | S-full vs S-nobelief |
| E-PLAST | Does the reset itself help adaptation speed on B? | B1-R vs B1-T vs S-copy |
| E-RANK | Do ranking terms reduce true regret in simulation? | S-full vs S-norank |

### 11.5 Statistics

Inherited from spec §19.4:

- ≥3 adapter/filter seeds × 3 deployment streams;
- paired starts, randomised method order;
- temporal-block bootstrap for streams, episode bootstrap for task metrics;
- report the number of independent sessions.

Primary endpoints, fixed in advance:

1. $R_{A|B}$ in A→B→A;
2. time-to-recover on A-return.

Report everything else as secondary. Correct for the number of S-variants compared, using Holm; this is a hypothesis-level choice.

### 11.6 Sleep acceptance gates

- **SG0, implementation.** Theorem 9.1 bit-identity on a probe batch. Theorem 9.2 tolerance. I1–I3. C1 non-aliasing. Snapshot round-trip of slot maps and schema. Cross-schema load refused. A no-op sleep (skip condition) leaves serving bit-identical. Abort at every state leaves serving and builder bit-identical. Dreams are unrepresentable as `WindowRef`. Sleep never records busy time on the wake ledger.
- **SG1, consolidation fidelity.** On ≥90% of sleeps, the lossy modes meet §5.7 thresholds and M1 meets the tolerance.
- **SG2, gating.** No accepted sleep candidate triggers a rollback within 60 min at a higher rate than ordinary promotions do.
- **SG3, retention (the main claim).** S-full (or S-copy, if it is no worse) improves $R_{A|B}$ and time-to-recover over **both** B1 and B1-T, with the block-bootstrap CI excluding 0. No-shift degradation stays within 2%. Slot occupancy stays bounded.
- **SG4, dreaming.** S-full beats MC-nodream on future real loss with the CI excluding 0. Otherwise REM stays **disabled** by default.
- **SG5, belief.** Post-reset $|\mu_d|/\sigma_d\le1$. Compact NLL non-inferior. Re-inference after a power cycle is faster than S-nobelief. Otherwise M4 stays disabled.
- **SG6, systems.** Sleep never overlaps motion. Holds are within the operator budget. Recovery from a crash mid-sleep is verified in the hardware-free harness.

Only after SG0–SG3 pass may sleep run outside shadow mode, and then only behind the existing `shadow` flag: in shadow mode sleeps produce and gate candidates but never install them.

---

## 12. What transfers from the LLM paper, and what does not

| Paper element | Here | Transfers? | Reason |
|---|---|---|---|
| Wake/sleep lifecycle, no train/test split | wake = online loop; sleep = gated offline procedure during a commissioned hold | **yes** | The system already has no train/test split online (spec §12.2). A hold is the physical form of "little or no external input" |
| CMS frequency spectrum | T0 context … T4 frozen (§3) | **partly** | The levels exist, but the slow levels are written by gated consolidation, not by chunk-accumulated gradients (paper eq. 2). A CMS-gradient slow tier would drift without a gate; it is a possible ablation only |
| "Consolidate before a block updates" | trigger T-c: consolidate before the fast tier absorbs a new persistent condition | **yes, reinterpreted** | The fast tier updates every step. The dangerous coincidence is a regime switch with $K=1$ |
| Pre-allocated masked experts | bounded slow slots with an activity vector (§4) | **yes** | Also gives fixed tensor shapes (no reallocation, graphs stay valid) |
| Sparse MoE routing | discrete regime index (spec §8.2) selects the slot bank | **partly** | No learned router in version 1. A belief-conditioned soft router would make weights context-dependent at fast timescales, against spec §12.3's warning |
| Compute → consolidate → update | §5.1 | **yes** | With the important change that teacher = **validated** serving version only |
| Upward distillation (small teacher → larger student) | equal capacity after reset; expansion = fresh parameters | **no (framing)** | Rank-matched transplant is exact (Theorem 9.2); the "larger student" argument is moot. What transfers is "write into fresh capacity" |
| GKD on-/off-policy mixture | (S2) with teacher-path and student-path latent rollouts | **yes, cleanly** | Detached feedback already equals "no backprop through student sampling" |
| Divergence choice (forward/reverse KL, JSD) | $D_z$ point metric | **no** | Deterministic outputs have no distributions to diverge, except the compact predictive Gaussians, where (S6) uses forward KL |
| Learning-to-Imitate (RL with semantic + Levenshtein reward) | action-ranking agreement (S3) | **analogue only** | The behaviour that matters is the planner's choice. It is differentiable, so no RL is needed. There is no semantic reward model |
| Synaptic pruning (reset sender experts) | fast-tier reset at promotion, with fresh optimizer | **yes** | And it is lossless thanks to transplant: plasticity reset without forgetting (see loss-of-plasticity and primacy-bias literature) |
| Dreaming via SEAL / ReST-EM | REM with hindsight and retention teachers; family bandit in version 2 | **partly** | Self-generated targets are forbidden (spec §7.6) and a self-teacher is a no-op (Proposition 9.10). Reward = improvement on **future real** transitions through the gate. Per-dream LoRA copies are unaffordable |
| Random extra expert per router | — | **no** | No physical meaning for one arm; mixing regime corrections fabricates dynamics |
| Gradient-importance top-k + b random | capped, corrected prioritisation of dreams (S7) | **yes, constrained** | Spec §11.3 allows priority only capped and corrected; never on real replay |
| Hyperparameters (LoRA r64/α128, lr 5e-6, batch 32, 100 steps, 5 extra width-64 blocks) | rank 8/α16 inherited; 100 steps adopted; LRs from spec stage 3/online | **no (numbers)** | Different model, data scale and loss. Only the step count is borrowed, as a starting point |
| Evidence (LLM benchmarks only; consolidation ablation deltas between −0.2 and +2.4 points on avg@16, with no confidence intervals; removing the semantic reward slightly *improved* one benchmark) | — | **no** | Says nothing about robot world models. Every claim here needs its own SG-gates |

Things the paper does not specify, which this document had to decide: $\lambda$, $\alpha$, $\gamma$, $z_0$, the divergence $F/D$, the source of KS inputs $x$ during "no external data" sleep, the RL algorithm for LTI, the router behaviour for new experts, the gating of consolidated models, and the slot-exhaustion policy.

---

## 13. Failure modes and required responses

| # | Failure mode | Mechanism | Required response / test |
|---|---|---|---|
| F1 | Consolidating unvalidated drift | Builder progress since the last promotion includes noise or sensor-fault fitting | Teacher = serving only; drain; never consolidate builder state |
| F2 | Every consolidation rejected | Superiority gate on a preservation candidate (Proposition 9.8) | Verify-only and non-inferiority modes (C5) |
| F3 | Slot masks aliased across roles | Buffers shared by `clone_role` | C1; non-aliasing test |
| F4 | Slot map missing from snapshot | Mask not a parameter | C1; round-trip and digest test |
| F5 | Online trains slow tiers, or nothing | `deltas[regime]` semantics | C2; equality test of trainable sets |
| F6 | Residual never consolidated | Residual not tiered | C3 |
| F7 | Legacy snapshot loaded into a tiered model | Same keys, different meaning | C6 schema refusal |
| F8 | Stale Adam moments push the reset fast tier | Moments from the old parameters | New `OnlineBuilder` (fresh optimizer) at promotion |
| F9 | Post-sleep adaptation stalls | Inherited 100-update warmup ≈ 1 h at 2B | Post-sleep warmup 20 (hypothesis); measure the time to first accepted promotion after sleep |
| F10 | Proximal/anchor terms point at old references | `reference_state` captured at construction | C7 |
| F11 | Anchor ratchet | Compounding 5% tolerance | Cumulative check (Proposition 9.9); M2 |
| F12 | Belief double counting | Student trained with $d=0$, deployed with inferred $\hat d$ | Consistency training (§7.3 step 4); re-initialisation (§7.4) |
| F13 | Partial absorption ($H\,d$ pathway) | Observation model frozen | Absorption check; no prior reset; reject an inconsistent candidate |
| F14 | Consolidating camera nuisance as dynamics | Phone auto-exposure, white balance, mount drift | Camera-contract hash; quality channels clean over all sleep data; locked exposure at commissioning |
| F15 | Consolidating a transient (contact) as persistent | $d$ or fast tier briefly offset | Persistence criterion (§7.2); trigger requires accepted promotions |
| F16 | Self-labelled dreams | Teacher = student | Forbidden; Proposition 9.10 test: gradient exactly 0 |
| F17 | Dreams treated as observations | Wrong data path | C12; no dream in belief, replay, anchors, gate, or eq. 26 |
| F18 | Infeasible dream commands | Commands the robot would never execute | Shared limiter and joint limits (`commands_for`) |
| F19 | Teacher exploits in planner dreams | CEM finds sequences where the teacher is wrong | Planner share ≤1/3; competence filter; real-data acceptance only |
| F20 | Long-horizon dream labels compound teacher error | AR drift | $H_d=2$ unless the real 5-step error is validated; $\gamma$ weighting |
| F21 | Prioritisation amplifies faults | Priority on real errors | Priority only on dreams; capped and corrected (S7) |
| F22 | Compaction loses knowledge | Truncation | Whitened energy $\eta$; distillation refine; non-inferiority gate; defer if failing |
| F23 | Slot exhaustion | Bounded bank | §4.4 order; `capacity_exhausted`; deep consolidation; no silent eviction |
| F24 | Sleep during regime ambiguity | Mixed responsibilities | Require a stable dominant regime; responsibility-weighted data; quarantine excluded |
| F25 | Leakage between sleep train and validation | Overlapping dense windows | Block split with guard gaps (§5.6) |
| F26 | Gate reuse inflates acceptance | Adaptive reuse | Report query counts; untouched final benchmark (spec §13.2) |
| F27 | Stale KV caches after slot changes | Parameter change | Version bump on every slot event (I3); `StaleCacheError` |
| F28 | Compiled graph serves old slot values | Constant folding | Live tensors; parity test after activation (spec §16) |
| F29 | Nonfinite student | Numerics | Abort the sleep; nothing installed; investigate |
| F30 | Sleep overruns its budget | Compute | Phase caps; abort ≥1.5× |
| F31 | Wake starved after sleep, or duty bursts | Cumulative duty ledger | Separate `SleepLedger`; windowed wake duty (C9) |
| F32 | Crash mid-sleep | Partial state | Unvalidated area; manifest untouched; restart from validated |
| F33 | Scene changed during the hold | Human moved objects | Fresh reobservation and revalidation before the first plan (spec §15.3) |
| F34 | Rollback across a sleep loses state | Slot map, slow prior or builder not restored | Version carries everything (§10.4) |
| F35 | Teacher changes during sleep | Concurrent promotion | Drain; pin by digest; re-verify the digest at the end |
| F36 | No-shift consolidation of noise | Nothing validated to consolidate | Skip condition (§10.1) |
| F37 | Two challengers at once | Dream gated alongside consolidation | Sequential gating; one challenger at a time |
| F38 | Hold-dominated gate data | Profile A gate data mostly from a stationary arm (no excitation, spec §17.3) | Report the fraction of gate transitions with $\lVert\Delta u\rVert>$ threshold; recommended minimum 30% excited transitions for lossy candidates (hypothesis) |
| F39 | Regime merge misused | Merging truly distinct regimes | Version 2; requires its own evaluation |
| F40 | Repurposed $\xi$ after M4 | Persistent drive of fast coordinates | Log mean $\lvert\xi\rvert$ before and after; flag if it rises >2× |

Like spec §18, this list is not exhaustive. Log unclassified sleep failures and add them as regression cases.

---

## 14. Default hyperparameters (all hypotheses unless marked)

| Group | Parameter | Default |
|---|---|---|
| Tiers | fast rank / α | 8 / 16 (*inherited*) |
| | slow rank / α / slots per regime / reserve | 8 / 16 / 8 / 1 |
| | compact correction slow slots / rank | 8 / 8 |
| Triggers | $U_{\max}$ / $P_{\max}$ / min interval | 200 updates / 4 promotions / 30 min |
| | belief persistence $T_b$ / $z_b$ / sign stability | 10 min / 3 / 0.9 |
| KS | steps / batch / warmup | 100 / 16 / 10 |
| | $\lambda$ / $\gamma$ / $H_{\rm real}$ / $H_{\mathcal P}$ | 0.5 / 0.95 / 2 / ≤5 |
| | $w_{\rm real},w_{\rm plan},w_{\rm rank},w_{\rm ret},\mu_{\rm gt},w_{\rm prox}$ | 1, 0.5, 0.25, 0.5, 0, $10^{-4}$ |
| | ranking $M$, contexts/step, every $n$ steps, $m$, $\beta_{\rm top}$, $k$ | 16, 4, 4, 0.05, 1, 4 |
| | LRs zero-init / copy-init | stage-3 rates / online rates (*inherited values*) |
| Fidelity | $\rho_{\rm fid}$, $\tau_b$, top-1, regret | ≤0.05, ≥0.9, ≥0.75, ≤0.02 |
| Gate | $\delta_{\rm NI}$ / anchor vs serving / cumulative | 1% / 2% / 5% |
| | future transitions / anchors / block | 128 / 512 / 2 s (*inherited*) |
| Compaction | $\epsilon$ damping / $\eta$ threshold / merge size $m$ | $10^{-4}\mathrm{tr}C/d$ / 0.98 / 2 |
| REM | contexts / sequences per context / $H_d$ / steps | 64 / 8 / 2 / 50 |
| | family shares planner/perturb/excite/hold | 1/3, 1/3, 1/6, 1/6 |
| | priority $\kappa,a,\varepsilon,\beta$ | 3, 0.6, 0.25, 1 |
| | hindsight lag $T_h$ / teacher-advantage margin | 60 s / 1% with UCB<0 |
| | competence filter quantile | 0.9 |
| Data | validation block share / guard gap | 15% / 54 raw ticks |
| Post-sleep | builder warmup | 20 updates |
| Budget | full-sleep cap | 500 SUE (≈30 min at 2B) |

---

## 15. Implementation map and required tests per agent

### 15.1 SLEEP-2 (`robojepa_ttt/sleep/tiers.py`)

- **Implement:** C1–C3, C6 and §3.3 conversion; the API of §0; `transplant`; `merge_slots`; the whitened compaction helper (or expose the factors so SLEEP-3 can implement it); seeded per-slot A init; activity vector parameter.
- **Tests:**
  - Theorem 9.1 bit-identity;
  - Theorem 9.2 equality for both the fused-qkv (300M) and split q/kv (2B) layouts, including $r_S>r_F$ padding with random $R$;
  - Proposition 9.3;
  - conversion equality (§3.3);
  - `set_trainable("online")` equals the fast set;
  - clone non-aliasing; snapshot/digest round-trip; cross-schema refusal;
  - invariants I1–I3;
  - residual tier equality and fast-only `online_parameters`.

### 15.2 SLEEP-3 (`robojepa_ttt/sleep/consolidation.py`)

- **Implement:** modes M0–M4 (M4 behind a C4 capability check); (S2) with both paths via `forward_tokens` on custom sequences; (S3); (S4); (S6) when the compact model is available; (S5) refinement; §5.7 metrics; verify-only and non-inferiority decision functions (C5) on `GateInput`-like data; Proposition 9.8 test.
- **Suggested signature:** `consolidate(teacher, student, data, config, mode) -> ConsolidationResult(status, mode, slot, metrics, candidate_state, reasons)`. `teacher` and `student` bundle the visual `AdaptedPredictor` and the optional `CompactBelief`/`ObservationSketch`.
- **Tests:**
  - teacher-path and student-path outputs equal per-step predictions (frame causality);
  - $L_{\rm on}=0$ and gradient exactly 0 when student ≡ teacher (Proposition 9.10);
  - $L_{\rm rank}\ge\log2\times$ misorder rate (Proposition 9.7);
  - M1 passes verify-only; the superiority gate rejects M1;
  - non-inferiority accepts an exact candidate and rejects a 3%-worse one;
  - fidelity metrics on synthetic cases;
  - only slot $k^\ast$ receives gradient.

### 15.3 SLEEP-4 (`scheduler.py`, `dreaming.py`, `evaluation.py`, `scripts/sleep_cycle.py`)

- **Implement:** §10 (triggers, inhibitors, state machine, DRAIN/HOLD/RESUME, `SleepLedger`, candidate gating sequence, promotion swap per C7/C8, rollback metadata, crash-safe unvalidated area); §8 (families, competence filter, (S7), hindsight teacher, teacher-advantage pre-check, dream slot); §11 (recurrence protocol, metrics, baselines incl. B1-T and the matched-compute variants).
- **Tests:**
  - no sleep with an active plan;
  - abort at every state leaves serving and builder bit-identical;
  - dreams never become `WindowRef`s or observations;
  - the wake ledger is untouched by sleep;
  - T-c micro-sleep runs before builder training on the new condition;
  - (S7) unbiasedness numerically;
  - the guard-gap split has no overlapping windows;
  - promotion swaps builder, reference and caches; rollback restores the slot map;
  - an end-to-end A→B→A run on the simulated arm in the hardware-free harness.

### 15.4 Belief agent

- **Implement:** C4: slow correction slots in `CompactBelief`, versioned compact state, a `SlowPrior` store, and re-initialisation with burn-in from the event log at promotion of an M4 candidate.

---

## 16. Reference pseudocode (ordering and ownership, not a full implementation)

```python
# at a planning boundary, wake loop
req = scheduler.should_sleep(state)            # §10.1 triggers and inhibitors
if req:
    scheduler.drain(hub, runner)               # stop snapshots, resolve or discard challenger, pause builder
    robot.enter_commissioned_hold()            # movement disabled; perception continues at >= 1 Hz
    plan = scheduler.prepare(req)              # pin teacher (version, digest); R_tr / R_val with guard gaps; budgets
    cands = []
    try:
        if plan.compaction:  cands.append(consolidate(T, S_clone(), plan.data, cfg, "M3"))
        cands.append(consolidate(T, S_clone(), plan.data, cfg, plan.mode))    # M1 or M2
        if plan.belief_ok:   cands.append(consolidate(T, S_clone(), plan.data, cfg, "M4"))
        if plan.rem_ok and teacher_advantage(plan):                            # §8.2 pre-check
            cands.append(dream(cands[-1], plan, cfg))                          # separate dream slot
    except (BudgetExceeded, NonFinite, DigestChanged) as e:
        scheduler.abort(e); cands = []          # nothing installed; builder resumes unchanged
    store.save_unvalidated(cands)
    robot.release_hold()
    for c in cands:                              # sequential, one challenger at a time
        decision = gate(c)                       # verify-only / non-inferiority / superiority (§10.3)
        if decision.accept and at_planning_boundary():
            promote_sleep_candidate(c)           # §10.4: swap builder, reference, caches; belief rebuild or reinit
        else:
            discard(c)                           # first-rejected consolidation => dreams dropped too
            if c.kind == "consolidation": break
    scheduler.resume_builder()
```

---

## 17. Open questions (empirical, or for the orchestrator)

1. **E-KS1 outcome.** If S-copy is as good as S-distill0 and S-full, the default sleep is M1 + reset + selective M2, and distillation is reserved for compaction and belief.
2. **Measured 300M SUE**, and whether a second GPU makes "sleep without hold" possible (spec §15.5).
3. **Regime bank interaction.** With $K>1$, cross-regime interference from sleep is zero by construction (per-regime slots). Is a shared slow bank across regimes useful (forward transfer), and how should it be gated?
4. Whether the observation model should ever be trainable in sleep, to consolidate the $H\,d$ pathway. It is currently forbidden.
5. Hindsight smoothing: a proper RTS/IMM smoother versus the "later posterior" approximation (§8.2).
6. Excited-transition minimum for gate data in Profile A (F38).
7. Whether the cumulative anchor check (Proposition 9.9) should be added to ordinary promotions too. Recommended, but that is a change to `promotion.py`, which these agents do not own.
