# Source audit

## Sources actually inspected

The two arXiv v1 HTML manuscripts were downloaded and their equations, methods, experiments, and implementation appendices inspected. Local HTML and extracted text are in `sources/`. Relevant released RoboJEPA code/configuration and the LeRobot follower implementation were also inspected. This is a specification exercise; no pretrained model was executed.

1. [RoboJEPA: Scaling Robotic Latent World Models, arXiv:2610.10515v1](https://arxiv.org/html/2610.10515v1), especially §§2, 3.3, appendices B, C, F, H, I.
2. [JEPA-TTT: Persistent Test-Time Training of Latent World Models for Planning under Dynamics Shifts, arXiv:2610.00722v1](https://arxiv.org/html/2610.00722v1), especially §§3, 6, 7, 11–13.
3. [JEPA-TTT author project page](https://jepa-ttt.github.io/). The inspected project page links the paper but did not expose an implementation link. An exact code-level replication of JEPA-TTT has therefore not been verified here.
4. [RoboJEPA repository](https://github.com/facebookresearch/robo_jepa), inspected at commit `f93106f5f4a451fd526d51778aa8ab7c852eb824`.
5. [SO-101 action-space documentation at that commit](https://github.com/facebookresearch/robo_jepa/blob/f93106f5f4a451fd526d51778aa8ab7c852eb824/app/robo_jepa/docs/action_spaces/so101.md).
6. [State propagation implementation](https://github.com/facebookresearch/robo_jepa/blob/f93106f5f4a451fd526d51778aa8ab7c852eb824/app/robo_jepa/world_model.py), [CEM implementation](https://github.com/facebookresearch/robo_jepa/blob/f93106f5f4a451fd526d51778aa8ab7c852eb824/app/robo_jepa/mpc/mpc_utils.py), [predictor](https://github.com/facebookresearch/robo_jepa/blob/f93106f5f4a451fd526d51778aa8ab7c852eb824/app/robo_jepa/robo_jepa/models/predictor.py), [training implementation](https://github.com/facebookresearch/robo_jepa/blob/f93106f5f4a451fd526d51778aa8ab7c852eb824/app/robo_jepa/train.py).
7. [300M released configuration](https://github.com/facebookresearch/robo_jepa/blob/f93106f5f4a451fd526d51778aa8ab7c852eb824/configs/released/wm-300m/config.json), [2B configuration](https://github.com/facebookresearch/robo_jepa/blob/f93106f5f4a451fd526d51778aa8ab7c852eb824/configs/released/wm-2b/config.json), [1B training configuration](https://github.com/facebookresearch/robo_jepa/blob/f93106f5f4a451fd526d51778aa8ab7c852eb824/configs/scaling/wm_1b.yaml).
8. [LeRobot SO follower implementation](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py); the exact returned blob SHA is recorded in `sources/manifest.json`. The moving main-branch link is a discovery link; pin the fetched content before implementation.
9. [Official LeRobot SO-101 documentation](https://huggingface.co/docs/lerobot/so101) and [robot API](https://huggingface.co/docs/lerobot/main/api/robots).
10. Background precedent, not an implementation dependency: [PlaNet](https://arxiv.org/abs/1811.04551) for stochastic/deterministic recurrent latent dynamics; [Adams and MacKay](https://arxiv.org/abs/0710.3742) for probabilistic online change detection. The proposed detector below is explicitly a heuristic CUSUM, not a reproduction of Bayesian online changepoint detection.

## What is inherited, in compact form

**RoboJEPA.** A frozen V-JEPA 2.1 representation feeds an action-conditioned deterministic predictor. Training combines teacher-forced and autoregressive feature prediction, using L1 losses; planning compares predicted features with a goal image. The representation has 1664 channels per spatial token. The paper studies scaling and does not provide the persistent probabilistic filtering architecture proposed here. It explicitly describes large-model planning as taking seconds rather than being real time. Its empirical relationship between prediction error and planning performance is useful motivation, not a universal control guarantee. [Paper](https://arxiv.org/html/2610.10515v1).

**JEPA-TTT.** Predictor parameters, optimizer state, and dense replay persist across episodes; the encoder and offline-trained reward head stay fixed. Windows begin at every raw temporal offset. Its online objective uses preceding observed latents, so it is not the same as an autonomous multistep rollout loss. Reported settings include AdamW at `5e-5`, weight decay `1e-3`, batch 64, clipping 1, and at most one update per replan, with a cumulative update budget tied to new windows. Experiments concern one persistent dynamics shift, not recurring changes, visual shift, or lifelong retention. The proposed SO-101 system changes the architecture, objective, replay management, and deployment schedule. [Paper](https://arxiv.org/html/2610.00722v1).

The recurring belief, joint command bridge, parameter-efficient adaptation, mode bank, freshness policy, deployment gates, and training curriculum in `SPECIFICATION.md` are new design proposals. Their values are starting hypotheses, not settings validated by either paper.

## Findings that change the SO-101 design

| Finding | Evidence | Consequence |
|---|---|---|
| `so-101_joints` uses five arm joints plus gripper and state-difference actions | SO-101 action-space document | Keep six-dimensional joint planning and explicitly model commanded versus realized movement. |
| The SO-101 training pipeline does not use recorded leader commands as actions | Same document | A motor-target change cannot silently replace a measured displacement at inference. |
| The joint rollout path can use Cartesian/Euler integration | Same document and `world_model.py` | Add and test a joint-specific state propagation path. |
| SO-101 gripper scaling differs from a generic closedness convention | Same document | Implement an explicit legacy transform; do not treat gripper values as joint radians or invert direction implicitly. |
| Some stored FK poses are inconsistent with joints | Same document | Prefer the joint branch and validate data; do not learn the physical arm geometry from inconsistent labels. |
| The released CEM samples a hardcoded Cartesian/rotation/gripper layout | `mpc_utils.py` | Replace the action sampler for SO-101; changing `action_dim` alone is insufficient. |
| LeRobot can clip requested targets and returns the targets sent | `so_follower.py` | Log requested, issued, acknowledged, and measured quantities separately. A sent target is not a measured movement. |
| Degrees versus normalized joint ranges is configurable; gripper has its own scale | `so_follower.py` | Verify the installed version and calibration; convert once at the device boundary. |
| A pretrained KV cache depends on current weights | Predictor computation | Rebuild caches on an accepted adapter update. This is a mathematical consequence, not a feature supplied by the paper. |

## Source inconsistencies and reproduction limits

RoboJEPA's appendix prose and its planning table contain different translation-bound values; the configuration also has an epoch/iteration interpretation of warmup. Do not transfer these into SO-101 limits. The proposed configuration uses measured local joint limits and an independent training schedule. Published table enumeration of embodiments is not the array index in a checkpoint: resolve `so-101_joints` by name in the loaded configuration.

JEPA-TTT's stated dense-window admission span is longer than the minimal span needed by its keyframes. An exact paper reproduction should preserve its published admission rule. This proposal admits a window only when all explicitly indexed observations and executed commands are available, and labels that choice as a change.

The source inspection establishes compatibility requirements, not proof that a downloaded checkpoint works on the user's robot. Camera placement, gripper calibration, action normalization, network delays, command clipping, and actual hardware behavior remain to be measured.
