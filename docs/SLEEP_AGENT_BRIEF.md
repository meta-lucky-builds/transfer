# Brief for agents building the Sleep framework

Read this whole file, then the section for your role.

## Mindset (from the project owner, verbatim)

Think as much as possible and as deep as possible: first principles, ultra rigor, honest, creative, highest order of rigor and expert-level pattern recognition, edge-case and systems analyzer, Take as long as you need. There is no time limit and no reward for finishing early. Nothing here rewards a short pass think of all possible GIVE ME ENTIRE THING LIKE RESOLVE ALL THE ONES ALL THE PROBLEMS and the goal is they must build the sleep framework we thought at the start and i need them to build it by thinking like this and building like this

## State of the work

Earlier agents with your role were stopped mid-work. Their UNCOMMITTED files are in the working
tree (about 8,000 lines): `robojepa_ttt/sleep/{__init__,tiers,consolidation,scheduler,dreaming,evaluation}.py`,
`tests/test_sleep_tiers.py`, `tests/test_sleep_consolidation.py` (scheduler/dreaming/evaluation tests may be
missing). Review your files critically, keep what is correct, fix what is wrong, complete what is missing,
make the tests pass, and commit. Do not discard work without a stated reason.

## Read first (mandatory, before editing code)

1. `docs/SLEEP_SPECIFICATION.md` in full. Section 0 "Contract issues" is binding.
2. `docs/sleep_paper_notes.md`, `docs/sleep_research_notes.md`.
3. Read YOUR part of the actual paper fully: "Language Models Need Sleep: Learning to Self-Modify and
   Consolidate Memories" (arXiv 2606.03979v2) through the alphaXiv tools (load via ToolSearch, e.g.
   "alphaxiv paper content") or WebSearch / web fetch. Report exactly which sections you read.
4. The code your module touches (listed per role) and SPECIFICATION.md sections named per role.

## Binding contract (from the spec's section 0)

- C1: tier slot on/off masks are non-trainable `nn.Parameter`s whose names contain `.deltas.`. Buffers are
  shared between roles by `AdaptedPredictor.clone_role`; plain attributes are missing from
  `adapter_state`, digests and `VersionStore`.
- C2: `TieredLoRALinear` subclasses `LoRALinear`; its `deltas` is exactly the FAST tier, so
  `set_trainable("online")` trains only the fast tier. Slow slots live elsewhere under `.deltas.` names.
- C3: the residual's fast slot is the legacy `modulation_out`/`b_o` (trunk shared and frozen).
- C4: belief consolidation stays behind an explicit flag that raises when prerequisites are missing.
- C5: `evaluate_gate` only accepts improvement; build a verify-only / non-inferiority gate inside
  `robojepa_ttt/sleep/` (never edit `learning/promotion.py`).
- C6: tag tiered adapter state with a schema identifier.
- C7: after sleep a fresh `OnlineBuilder` and a new anchor reference are required.
- C8: external-challenger promotion path by wrapping, not editing promotion.py.
- C9: sleep compute is never booked on the wake `CreditLedger`; keep a separate sleep budget.
- C10: held-out sleep split with guard gaps against overlapping windows.
- C11: `consolidate_every` counts builder updates, not wall time.
- C12: dreams are structurally impossible to admit as replay windows.
- Default consolidation is EXACT TRANSPLANT (fast slot copied into an equal-rank slow slot);
  distillation only for lossy transfers; distil-from-zero is an ablation.
- Dreams are synthetic command sequences labelled by a teacher with MORE information (hindsight teacher,
  frozen older model), never observations or targets claimed as truth (spec 7.6); a dream slot is kept
  only if it improves future REAL transitions through the gate.

API (implement exactly or report deviations): `TierConfig(name, rank, alpha, slots, consolidate_every)`,
`convert_to_tiered(model, tiers)`, `tier_parameters(model, tier, regime=0, slot=None, component=...)`,
`activate_slot(model, tier) -> int`, `reset_tier(model, tier)`, `active_slots(model)`, `tier_state(model)`;
`consolidate(...) -> ConsolidationResult` documented at the top of consolidation.py.

## Rules for everyone

- Own only your files (below). Do NOT edit shared existing code (`robojepa_ttt/models/*`, `learning/*`,
  `belief/*`, `planning/*`, `data/*`, existing `scripts/*`, config). Wrap or subclass instead. If a
  shared change is truly unavoidable, keep it minimal and additive, run the B1 suites, and explain it.
- The files `robojepa_ttt/belief/online.py`, `robojepa_ttt/belief/train.py`, `tests/test_belief_adapt.py`
  are another stopped agent's incomplete work: do not touch or commit them.
- Never edit `third_party/`. Never `git add -A`, `git stash`, `git checkout`, `git reset`. Add only your
  own files by name. Other agents work in the same tree concurrently; if someone else's in-progress file
  breaks an unrelated test, note it and move on.
- Before each commit run your tests and `python -m pytest tests/test_layers.py tests/test_learning.py tests/test_online_loop.py -q -p no:warnings`.
- Push: `git push -q origin claude/dreamy-hamilton-pywhjz` (retry with backoff only on network errors).
  End every commit message with exactly these two lines:

      Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
      Claude-Session: https://claude.ai/code/session_0113EXssznAevJ8gKTuT5gNc

- Style: dataclasses, docstrings citing spec and paper sections, explicit refusals instead of silent
  defaults, deliberate deviations stated. No model identifiers anywhere in the repo. No pull requests.
- Report back: what you built (APIs), design decisions, what you kept or changed from the earlier files,
  paper sections read, pasted pytest summary lines, contract mismatches, open issues, commit hashes.

## Role SLEEP-2: tiered memory

Files: `robojepa_ttt/sleep/__init__.py`, `robojepa_ttt/sleep/tiers.py`, `tests/test_sleep_tiers.py`.
Paper: section 2.2 (Continuum Memory System), sections 3.1 to 3.2 (consolidation with parameter
expansion), the masked pre-allocated experts note, Figures 7 to 8, Appendix B; Nested Learning / Hope CMS.
Code: models/lora.py, models/adapted_predictor.py, models/token_residual.py, learning/online_builder.py,
learning/promotion.py, tests/conftest.py, tests/test_layers.py; SPECIFICATION.md 6.6, 9.6, 9.8, 12, 13.
Build: tiered LoRA (both attention layouts) and residual per the contract; B1 conversion preserving
outputs within 1e-6; masked slots get no gradient and change nothing; reset exactness; merge/compaction
(spec: whitened SVD) exact where possible with reported error; per-tier optimizer groups; spec defaults
8 slow slots per regime at rank 8. Tests on both layouts, including role isolation via clone_role,
adapter_state / VersionStore round trips, regime independence, KV-cache vs full parity with several tiers.

## Role SLEEP-3: consolidation

Files: `robojepa_ttt/sleep/consolidation.py`, `tests/test_sleep_consolidation.py`.
Paper: section 3.3 knowledge seeding in full, sections 4.1 to 4.2, Appendix A.2 and A.4; GKD objective.
Code: models/adapted_predictor.py, learning/losses.py, learning/online_builder.py, learning/promotion.py,
learning/online_runner.py, learning/replay.py, planning/joint_cem.py, belief/compact.py, belief/visual.py;
SPECIFICATION.md 9, 11 to 14, 17.5.
Build: compute -> consolidate -> update; teacher frozen clone; only the new slot trains; exact transplant
default; distillation objectives (real windows teacher-forced, on-policy student rollouts with no backprop
through generation, planner command sequences via JointCem.commands_for + limiter, action-ranking
agreement, small real-target anchor); non-inferiority gate (C5); atomic apply with sender reset, optimizer
reset and version bump; rejected gate leaves the model bit-identical; ConsolidationResult.

## Role SLEEP-4: scheduler, dreaming, evaluation

Files: `robojepa_ttt/sleep/scheduler.py`, `robojepa_ttt/sleep/dreaming.py`, `robojepa_ttt/sleep/evaluation.py`,
`scripts/sleep_cycle.py`, `tests/test_sleep_scheduler.py`, `tests/test_sleep_dreaming.py`, `tests/test_sleep_evaluation.py`.
Paper: section 1, section 3.1, section 3.4 Dreaming in full, section 4 continual-learning results and
Table 3, Appendix B.3 to B.5; SEAL method; ReST-EM objective.
Code: learning/online_runner.py, learning/replay.py, learning/promotion.py, learning/online_builder.py,
evaluation/perturbation.py, evaluation/prediction.py, planning/joint_cem.py, robot_io/command_limiter.py,
tests/test_online_loop.py; SPECIFICATION.md 7.6, 11, 12, 13, 15, 18, 19.
Build: wake/sleep scheduler (C7, C9, C11; planning boundary only; deterministic fake clock; abort on
non-finite; no partial state); safe dreaming (C12, hindsight teacher, gate on real transitions, capped and
corrected prioritisation, cross-regime mixing off with justification, dream-vs-real disagreement metric);
evaluation (A->B->A and A->B->C->A with CommandShift; prequential loss, recovery, retention, forgetting,
AUC, compute; baselines B1-without-sleep, sleep-without-expansion, sleep-without-dreaming, full sleep,
matched budgets; block-bootstrap CIs; gates SG0 to SG6); thin `scripts/sleep_cycle.py`. Each test file
under about 2 minutes on CPU. Inject SLEEP-3's `consolidate` and use test doubles until it is committed.
