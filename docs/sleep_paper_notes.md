# Notes: "Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories"

Behrouz, Hashemi, Javanmard, Mirrokni (Google / Cornell). Public on OpenReview since September 2025.
Condensed from the paper text the project owner supplied, for agents building the Sleep framework
in this repository. Read the original where details matter; search arXiv / alphaXiv / the web for it
and for "Nested Learning" (Behrouz et al. 2025, the Hope architecture and Continuum Memory System).

## Core claims

- A continual learner has no train/test split, only **wake** (receives data, processes it) and
  **sleep** (little or no external input; consolidates and self-improves).
- Two consolidation types. **Online** (wake): end-to-end multi-frequency learning moves knowledge
  from fast to slow modules (Nested Learning / Hope). **Offline** (sleep): replay-driven, lossy
  distillation into more stable, larger-capacity parameters. The paper is about offline.
- Catastrophic forgetting is reframed as **insufficient capacity**: slow memory must be overwritten
  to absorb new knowledge unless capacity grows.

## Continuum Memory System (CMS)

Chain of MLP blocks with update frequencies f_1 >= ... >= f_k (updates per unit time; the slowest
block's update is the time unit). Block l has chunk size C(l) = max C / f_l and updates every C(l)
steps with the accumulated optimizer error over its chunk:

    theta_{i+1}^{(l)} = theta_i^{(l)} - sum_{t=i-C(l)}^{i} eta_t^{(l)} f(theta_t^{(l)}; x_t)   if i = 0 mod C(l), else unchanged.

Attention = infinite-frequency (in-context) memory; frozen MLPs = zero-frequency memory. Fragility
and stability are relative: every block is short-term relative to slower ones. Forgetting happens
when update periods of all blocks coincide, so consolidate a block BEFORE it updates.

## Sleep stage 1: memory consolidation (NREM analogue)

1. **Parameter expansion.** Target (slower) block l* is a sparse MoE. Add a new low-rank expert
   {A in R^{d x d_low}, B in R^{d_low x d}} reserved for knowledge transferred from block l*-1.
   Implementation note from the paper: pre-allocate the experts and mask them in forward and backward
   until activated (fixed brain capacity, new connections).
2. **Compute -> consolidate -> update protocol** at steps divisible by C(l*-1):
   (a) compute the sender's prospective base update but do not apply it;
   (b) teacher = model before the update/expansion; student = model with the prospective sender
       update, the sender's experts reset, and the new low-rank expert in the target; optimise ONLY
       the new expert (all other student parameters frozen) by the knowledge-seeding objective;
   (c) apply the sender update, reset the sender's previously added experts ("synaptic pruning"),
       activate the new expert in the target.
3. **Knowledge seeding** = upward distillation (small teacher -> larger student). Objective builds on
   Generalized Knowledge Distillation (on-policy mix, no backprop through student sampling):

       L = (1-lambda) E_{(x,y)~D}[F(teacher || student)(y|x)] + lambda E_{x~D} E_{y~student}[F(teacher || student)(y|x)]

   plus **Learning to Imitate** (RL): sample a prefix of a teacher-generated sequence ("dream"), the
   student completes it, reward r = gamma r_sem + (1-gamma) r_abs (semantic match by a frozen reward
   model; normalised Levenshtein similarity above a threshold). Combined:

       L_KS = E_x[(1-alpha) E_{y~student}[r(y)] - alpha E_{y~student}[D(teacher || student)(y|x)]]

## Sleep stage 2: dreaming (REM analogue)

Self-improvement from self-generated data, built on SEAL (Zweiger et al. 2025, ReST-EM):
generate m dreams from context C; each MoE router also picks a random extra expert (novel mixing of
unrelated knowledge); score dreams by gradient-based importance g = grad L_SFT(dream) and keep top-k
plus b random for diversity; fine-tune an isolated LoRA copy per dream; reward 1 if the copy's task
score tau improves, else 0; optimise the dream generator with ReST-EM. Two-stage design (consolidate,
then dream) is argued to reduce the forgetting and collapse seen when iterating self-improvement or
on-policy self-distillation directly on a fixed model.

## Reported settings and evidence (LLM only)

LoRA rank 64, alpha 128, lr 5e-6, effective batch 32, 100 steps; 5 extra MLP blocks of width 64 with
active parameter count unchanged. Ablations: removing imitation learning, semantic reward or
expansion each hurt math reasoning. Sleep ~4x slower than SFT per step but reaches equal accuracy in
less wall time. All evidence is on language tasks; nothing about robot world models.

## Mapping to this repository (agreed analysis, see conversation)

| Paper | Here |
|---|---|
| in-context / fastest memory | belief state h_t, fast xi, slow context d (robojepa_ttt/belief/) |
| fast parameters | online builder adapters (learning/online_builder.py), <= 1 update/s, measured ~1 per 37 s at 2B |
| mid parameters | promoted serving version (learning/promotion.py) |
| slow parameters | offline stage-3 reference adapters |
| frozen | RoboJEPA base transformer and V-JEPA encoder |
| teacher/student protocol | ModelHub roles + clone_role (frozen base shared, adapters copied) |
| distillation loss | token_metric between rollouts, already used as L_anchor in online_builder |
| reset fast tier | zero adapters + OnlineBuilder.reset_optimizer + version-checked cache rebuild |
| gate before applying | evaluate_gate on future real transitions + anchors + action probe |

Required changes in meaning for a deterministic latent world model:
- LTI's sampling-alignment rewards (Levenshtein, semantic) do not apply; the useful analogue is
  **action-ranking agreement**: student must rank CEM candidate command sequences like the teacher
  (spec 14.7: lower token loss can coexist with worse ranking).
- **Dreams must never be targets.** Spec 7.6: the model's own predictions are not observations.
  Dreams may be synthetic COMMAND sequences whose targets come from the teacher (distillation), and
  any "keep the dream" reward must be measured on held-out REAL transitions through the gate.
- Gradient-importance dream selection is prioritised replay, which spec 11.3 forbids in version 1
  unless capped and corrected.
- Random cross-regime expert mixing has no physical meaning for a robot; likely skip or justify.
- Capacity growth must stay bounded (spec: no automatic eviction, at most 4 regimes): pre-allocate
  slots up to a maximum and free fast slots after each consolidation; define merge/eviction when full.
- Belief consolidation: teacher conditioned on belief (recent history, inferred d / regime),
  student without that context but with a new slot -> internalise what the belief inferred, then
  reset the slow belief prior. Spec 12.3 warns context and weights must not both adapt fast.
