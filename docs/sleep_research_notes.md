# Sleep: research notes (sources, key equations, related work)

These notes back [SLEEP_SPECIFICATION.md](SLEEP_SPECIFICATION.md). They record what was read, how fully, which equations were quoted, and which related work bears on Sleep for a robot latent world model.

Every paper listed was retrieved from arXiv/alphaXiv during this work (2026-11-03), except where marked **cited, not re-read**. Quotations are short, and equations are transcribed into this document's notation.

## 1. Reading log: what was read, and what could not be accessed

| Source | Link | How much was read | Not accessed |
|---|---|---|---|
| Behrouz, Hashemi, Javanmard, Mirrokni, "Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories" (arXiv 2606.03979v2, 10 Jul 2026) | https://arxiv.org/abs/2606.03979 | **Full extracted text**: abstract, §1–§5, appendix A (related work A.1–A.4) and appendix B (B.1–B.5), all figure and table captions; the reference list was skimmed | The numeric content of the plotted figures (Fig. 3–6: class-incremental curves, memory levels, CTNL, BABILong), which is visible only as images. The September 2025 OpenReview version that the paper mentions, so it is unknown whether it differs |
| Behrouz, Razaviyayn, Zhong, Mirrokni, "Nested Learning: The Illusion of Deep Learning Architectures" (arXiv 2512.24695, NeurIPS 2025) | https://arxiv.org/abs/2512.24695 | Targeted retrieval: abstract and §1 overview; §6 takeaways (part); **§7 Continuum Memory System in full** (7.1 eq. 70–71, nested/sequential/independent variants eq. 72–74, the forgetting argument, efficiency, 7.2 the M3 optimizer); §8.1–8.3 (self-referential Titans, Hope, eq. 86–97); parts of §9; §10 conclusion | §2–§5 derivations (optimizers as associative memory, Delta GD) beyond the overview; most experiment tables |
| Zweiger et al., "Self-Adapting Language Models" (SEAL, arXiv 2506.10943, NeurIPS 2025) | https://arxiv.org/abs/2506.10943 | §1–§6 main text, Alg. 1, eq. 1–4, the limitations section (catastrophic forgetting, compute cost), appendices A.1–A.2 and B.1–B.3 | Remaining appendices (B.4 onward), apart from the forgetting figure caption |
| Agarwal et al., "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes" (GKD, ICLR 2024, arXiv 2306.13649) | https://arxiv.org/abs/2306.13649 | §1–§4.3 including Alg. 1, eq. 1–5, divergence and λ ablations, appendix ablation figures | Appendix A details beyond the ablation figures |
| Schwarz et al., "Progress & Compress" (ICML 2018, arXiv 1805.06370) | https://arxiv.org/abs/1805.06370 | §1–§6 (method, online EWC, results) | — |
| Wu, Zhang, Raghunathan, "Off-Policy Merging Beats On-Policy Self-Distillation for Continual Learning" (arXiv 2610.05872) | https://arxiv.org/abs/2610.05872 | §1–§3 and §8–§9, appendices A, C, D | Middle result sections |
| Yang, Thompson, Konidaris, "Characterizing Replay Retention Under Dynamics Shift in Model-Based Reinforcement Learning" (arXiv 2609.18167) | https://arxiv.org/abs/2609.18167 | Full text | — |
| Nagabandi, Finn, Levine, "Deep Online Learning via Meta-Learning: Continual Adaptation for Model-Based RL" (MOLe, ICLR 2019, arXiv 1812.07671) | https://arxiv.org/abs/1812.07671 | §1–§8 and appendix B | — |
| Other related work (§5) | links given | Title, abstract and index metadata, unless stated otherwise | Full texts |

The repository documents were also read in full: [sleep_paper_notes.md](sleep_paper_notes.md), [SPECIFICATION.md](../SPECIFICATION.md) and [TRAINING_GUIDE.md](../TRAINING_GUIDE.md). The code read in full: `models/lora.py`, `models/token_residual.py`, `models/adapted_predictor.py`, `learning/online_builder.py`, `learning/losses.py`, `learning/promotion.py`, `learning/online_runner.py`, `learning/replay.py`, `learning/windows.py` (header), `belief/compact.py`, `belief/online.py`, `belief/visual.py`, the uncommitted online compact builder in `belief/train.py` (WIP by another agent), `planning/joint_cem.py`, the header of `planning/belief_cem.py`, and `config_so101.yaml`.

## 2. The Sleep paper: key content and equations

**Framing (§1, §3.1).** A continual learner has no train/test split, only *wake* (input arrives) and *sleep* (minimal or no input; consolidation and self-improvement). The paper distinguishes two kinds of consolidation:

- online consolidation: end-to-end multi-frequency learning, as in Nested Learning/Hope;
- offline consolidation: sleep.

The paper is about offline consolidation. In its words, catastrophic forgetting "is an inherent cause of model's limited capacity" (§3.2), and later (A.4) the method "reframes catastrophic forgetting as a problem of insufficient capacity rather than of sampling distribution".

**CMS (§2.2, quoted from NL).** A chain $y_t=\mathrm{MLP}^{(f_k)}(\cdots\mathrm{MLP}^{(f_1)}(x_t))$. Block $\ell$ has chunk size $C^{(\ell)}=\max_{\ell'}C^{(\ell')}/f_\ell$ and update rule
\[
\theta^{(f_\ell)}_{i+1}=\theta^{(f_\ell)}_i-\begin{cases}\sum_{t=i-C^{(\ell)}}^{i}\eta^{(\ell)}_t f(\theta^{(f_\ell)}_t;x_t)& i\equiv0\pmod{C^{(\ell)}}\\0&\text{otherwise.}\end{cases}
\]
Forgetting "can happen when the update period of all models matched at some point … it is crucial that before each update of a memory block, a mechanism consolidates the abstracted knowledge of that block to more stable parameters".

**Parameter expansion (§3.2).** Each block is a sparse MoE. To consolidate block $\ell^\ast-1$ into $\ell^\ast$, a new low-rank expert $\{A\in\mathbb R^{d\times d_{\rm low}},B\in\mathbb R^{d_{\rm low}\times d}\}$ is added to $\ell^\ast$. Implementation note: experts are pre-allocated and "masked … in the forward and backward pass, before their initial activation".

**Compute → consolidate → update (§3.3).** In outline:
- (a) the sender's base update is computed from the accumulated gradients but not applied;
- (b) the teacher is the pre-update model; the student uses the prospective sender parameters, with the sender's experts reset and the new expert in the target; only $A,B$ are optimised;
- (c) the update is applied, the sender's earlier experts are reset ("synaptic pruning"), and the new expert is activated.

**Knowledge seeding objective (§3.3).** GKD-style:
\[
\mathcal L(\theta,\theta^{\rm exp})=(1-\lambda)\,\mathbb E_{(x,y)\sim\mathcal D}\big[F(\mathrm{LM}_\theta\|\mathrm{LM}_{\theta^{\rm exp}})(y|x)\big]+\lambda\,\mathbb E_{x\sim\mathcal D}\mathbb E_{y\sim\mathrm{LM}_{\theta^{\rm exp}}}\big[F(\cdot\|\cdot)(y|x)\big].
\]
There is no backpropagation through student sampling, and all student parameters except the expansion are frozen.

Learning to Imitate, eq. 3–4: reward $r=\gamma r_{\rm sem}+(1-\gamma)r_{\rm abs}$, where $r_{\rm sem}\in\{0,1\}$ comes from a frozen reward model, and $r_{\rm abs}=1-z(\hat d,d)/\max(|\hat d|,|d|)$ if the Levenshtein distance $z\le z_0$, else 0. Combined:
\[
\mathcal L_{\rm KS}=\mathbb E_x\big[(1-\alpha)\mathbb E_{y\sim\mathrm{LM}_{\theta^{\rm exp}}}[r(y)]-\alpha\,\mathbb E_{y}[D(\mathrm{LM}_\theta\|\mathrm{LM}_{\theta^{\rm exp}})(y|x)]\big].
\]
This is written as a quantity to maximise. The paper does not state which RL estimator is used for the reward term.

**Dreaming (§3.4).** Built on SEAL:
1. Generate $m$ dreams from context $C$; each MoE router additionally selects a random expert.
2. Score each dream by $g^{(i)}=\nabla_\theta\mathcal L_{\rm SFT}(\text{DREAM}^{(i)},\theta)$; keep the top-$k$ plus $b$ random dreams.
3. LoRA-SFT an isolated copy per dream.
4. Reward 1 if task score $\tau$ improves, else 0 (eq. 5); optimise with ReST-EM.

The paper claims the two-stage design (consolidate, then dream) is more robust to forgetting than iterating self-improvement directly.

**Evidence and settings (§4, App. B).** Class-incremental learning (CLINC, Banking, DBpedia; figures only), long-context QA (figures), CTNL (figure), BABILong (figure), math reasoning (Tables 1–2), SQuAD knowledge incorporation (Table 3), few-shot ARC (Table 4: Sleep 80% vs SEAL 72.5%).

The consolidation ablation (Table 1, avg@16) in points, relative to full Sleep (79.2 / 69.0 / 46.1):

| Removed component | AIME-24 | AIME-25 | HMMT-25 |
|---|---|---|---|
| imitation learning | −2.4 | −1.1 | −1.1 |
| semantic reward | −0.3 | **+0.2** | −1.6 |
| expansion | −1.0 | −1.1 | −1.2 |

The text cites Table 1 as "Figure 1" (§4.2). No confidence intervals are given.

Training configuration (Table 5): LR 5e-6, effective batch 32, 100 steps, LoRA rank 64 and α 128; "5 MLP blocks with dimension 64 as the additional parameters", with the active parameter count unchanged. Efficiency: about 4× slower per step than SFT, but 3.6–4.8× less wall time to reach equal accuracy.

**Gaps that matter for implementation.** The paper does not give values or choices for:
- $\lambda,\alpha,\gamma,z_0$;
- $F$ and $D$;
- where the KS inputs $x$ come from during a sleep without external data (D is "sampled from the teacher", but conditioned on what);
- the RL algorithm for LTI;
- how routers treat new experts;
- any gate on the consolidated model;
- what happens when pre-allocated experts run out.

The specification fixes each of these explicitly (spec §12).

## 3. Nested Learning / Hope: what matters for Sleep

- **CMS definition, eq. 70–71.** Identical to the formula above. Variants:
  - *nested*: the initial state of level $s+1$ is meta-learned at level $s$ and re-initialised at the end of its context (eq. 72);
  - *sequential*: all initial states are meta-learned at the lowest frequency (eq. 73);
  - *independent / head-wise*: $y_t=\mathrm{Agg}(\mathrm{MLP}^{(f_k)}(x_t),\dots)$, for example a learnable weighted sum (eq. 74).

  The tiered LoRA of this repository is structurally the **independent** variant with summation as Agg: (S1) in the spec.
- **Why CMS helps.** Knowledge forgotten from a fast block may still be stored in slower blocks, and backpropagation into initial states can "circle back" the knowledge (a loop through time). The NL conclusion states plainly that catastrophic forgetting is "not 'solved'" and is "a natural consequence of compression".
- **Hope.** Self-referential Titans followed by CMS (eq. 94–97). **M3 optimizer:** CMS applied to momentum (a fast and a slow momentum, the slow one updated every $\hat C$ steps). This is relevant as a reminder that AdamW moments are themselves a fast memory, which is why the specification resets them with the fast tier.

## 4. SEAL and GKD: what was borrowed and what was not

**SEAL.**
- The RL loss $\mathcal L_{\rm RL}=-\mathbb E[\mathbb E_{SE\sim\mathrm{LM}}r(SE,\tau,\theta)]$ is optimised by ReST-EM, which is filtered behaviour cloning on reward-1 self-edits.
- GRPO and PPO were tried and found unstable.
- The reward depends on the current parameters, so training must be on-policy.
- Limitations stated by the authors:
  - catastrophic forgetting under sequential self-edits;
  - every reward needs a full fine-tune plus evaluation, "approximately 30–45 seconds";
  - every context needs paired evaluation data.
- The paper's dreaming inherits all three. This is why the specification does generator learning, if ever, over command **families** and not per dream, and evaluates on future real transitions.

**GKD.**
- $\mathcal L_{\rm GKD}=(1-\lambda)\mathbb E_{(x,y)\sim(X,Y)}[D(p_T\|p_S)(y|x)]+\lambda\mathbb E_{x}\mathbb E_{y\sim p_S}[D(p_T\|p_S)(y|x)]$, with token-averaged divergence (eq. 2) and generalised JSD($\beta$) (eq. 1). There is no backpropagation through student sampling.
- The RL combination (eq. 5), $\mathbb E_x[(1-\alpha)\mathbb E_{y\sim p_S}r(y)-\alpha\mathbb E_{y\sim p_S}D]$, is exactly the form the Sleep paper's $\mathcal L_{\rm KS}$ reuses.
- Findings: on-policy and mixed data beat supervised data consistently; the best divergence is task-dependent; and the student must already be reasonable (it starts from an SFT checkpoint).
- For a deterministic latent predictor, "divergence" collapses to a distance between point predictions. The on-/off-policy distinction carries over intact as student-path versus teacher-path latent rollouts (spec eq. S2).

## 5. Related work found (cite only what was retrieved; one line each on relevance)

### 5.1 Consolidation with an active/fast learner and a slow store

- **Progress & Compress** (Schwarz et al., ICML 2018), https://arxiv.org/abs/1805.06370.
  - An active column learns each task; in a "compress" phase it is distilled into a knowledge base, which online EWC protects; the authors themselves describe the phases as day and night.
  - The paper recommends re-initialising the active column after compression for positive transfer (their Table 1).
  - This is the closest prior design to the Sleep paper's consolidation, and the Sleep paper does not cite it. Its distillation loss (their eq. 2): $\mathbb E[\mathrm{KL}(\pi_k\|\pi^{KB})]+\tfrac12\|\theta^{KB}-\theta^{KB}_{k-1}\|^2_{\gamma F^\ast_{k-1}}$.
- **Self-Distillation Enables Continual Learning** (Shenfeld et al., 2026), https://arxiv.org/abs/2601.19897. A demonstration-conditioned on-policy teacher (SDFT) reduces forgetting relative to SFT. Retrieved as index metadata; cited in Sleep A.4.
- **Off-Policy Merging Beats On-Policy Self-Distillation** (Wu, Zhang, Raghunathan, 2026), https://arxiv.org/abs/2610.05872.
  - "Grafting" learns the update on an earlier donor checkpoint, scales it (merging) and masks the most Fisher-sensitive directions.
  - It Pareto-dominates SFT and OPSD, and reports OPSD reasoning collapse.
  - Relevance: weight-space transplant and merge operations can beat distillation. This supports the specification's choice of exact transplant (M1) as the default, with distillation only where it is needed.

### 5.2 Continual learning of world models and dynamics models

- **Characterizing Replay Retention Under Dynamics Shift in MBRL** (Yang, Thompson, Konidaris, 2026), https://arxiv.org/abs/2609.18167.
  - Recency-only replay helps after large permanent shifts but hurts under **recurring** dynamics: −191 ± 17 return vs passive replay on Walker, recurring gain change.
  - A bias–variance model of retention: $E(\beta)=\beta^2M^2+\beta^2\sigma^2/n_o+(1-\beta)^2\sigma^2/n_f$, with optimal $\beta^\star=\frac{\sigma^2/n_f}{M^2+\sigma^2/n_o+\sigma^2/n_f}$.
  - Relevance: A→B→A recurrence is exactly the case where discarding old data, or old weights, is harmful. It supports per-regime reservoirs and consolidation over plain recency.
- **MOLe** (Nagabandi, Finn, Levine, ICLR 2019), https://arxiv.org/abs/1812.07671.
  - A mixture of online-SGD-adapted dynamics models with a Chinese-restaurant-process prior, $P(T_t=T_i\mid x_t,y_t)\propto p_{\theta_t(T_i)}(y_t|x_t)\,P(T_t=T_i)$, which recalls old tasks under recurrence. Plain continued gradient steps overfit to recent tasks and forget.
  - Relevance: the regime bank and quarantine of spec §8.
- **The Effectiveness of World Models for Continual RL** (Kessler et al., CoLLAs 2023), https://arxiv.org/abs/2211.15944. Index metadata. A world-model agent (Continual-Dreamer) uses selective replay against forgetting.
- **Augmenting Replay in World Models for Continual RL** (Yang, Kuhlmann, Kowadlo, 2024), https://arxiv.org/abs/2401.16650. Index metadata. Keeps a diverse long-term replay distribution (the WMAR baseline in the Yang et al. study).
- **Knowledge Retention for Continual Model-Based RL** (DRAGO, 2025), https://arxiv.org/abs/2503.04256. Index metadata. Incremental world models across tasks using synthetic experience to rehearse old knowledge, which is relevant to retention dreams.
- **Continual RL by Planning with Online World Models** (2025), https://arxiv.org/abs/2507.09177. Index metadata.
- **Mind the Model, Not the Agent: The Primacy Bias in Model-based RL** (2023), https://arxiv.org/abs/2310.15017. Index metadata. Primacy bias in world models: model resets matter.

### 5.3 Plasticity and resets (justify resetting the fast tier)

- **The Primacy Bias in Deep RL** (Nikishin et al., 2022), https://arxiv.org/abs/2205.07802. Periodic resets restore learning from later data.
- **Loss of Plasticity in Deep Continual Learning / Maintaining Plasticity** (Dohare et al.; arXiv 2306.13812; published in Nature 2024 according to citing papers), https://arxiv.org/abs/2306.13812.
- **Loss of Plasticity in Continual Deep RL** (2023), https://arxiv.org/abs/2303.07507. Index metadata.
- **Plasticity Loss in Deep RL: A Survey** (2024), https://arxiv.org/abs/2411.04832. Index metadata.
- Relevance: an exact transplant lets the fast tier be reset **without** forgetting. That combination ("plasticity reset without forgetting") is not available to plain online fine-tuning.

### 5.4 Low-rank adapter merging, expansion and compaction

- **ASVD** (2023), https://arxiv.org/abs/2312.05821, and **SVD-LLM** (truncation-aware whitening, 2024), https://arxiv.org/abs/2403.07378. Activation-aware or whitened SVD for low-rank compression. This is the per-layer optimality used in spec Theorem 9.4: truncate $[\Delta C^{1/2}]_r C^{-1/2}$.
- **Merge before Forget: single-LoRA continual learning via continual merging** (2025), https://arxiv.org/abs/2512.23017; **Shared LoRA Subspaces for almost strict continual learning** (2026), https://arxiv.org/abs/2602.06043; **GLAM: grouped LoRA adapter merging** (2025), https://arxiv.org/abs/2509.13211; **When the Merge Coefficient Stops Mattering: proximity-regularized merging** (2026), https://arxiv.org/abs/2609.32332. Index metadata. All are alternatives to "one new slot per sleep": merging into a fixed adapter budget.
- **Orthogonal subspace / decoupling methods**: CoDe-LoRA (2026), https://arxiv.org/abs/2610.08312; Making LoRA Identifiable (2026), https://arxiv.org/abs/2610.16664. Index metadata. Possible version-2 constraint: write new slots orthogonally to old ones in the whitened input space.
- **Mixture of LoRA experts / LoRA routing** is cited by the Sleep paper itself (Wu et al. 2024; Li et al. 2024, MixLoRA), not re-read. It is relevant only if a learned slot router is ever considered, which the specification defers.

### 5.5 Replay prioritisation

- **Prioritized Experience Replay** (Schaul et al., 2015), https://arxiv.org/abs/1511.05952. Prioritised sampling $P(i)=p_i^a/\sum_k p_k^a$ with importance-sampling correction $w_i=(N\cdot P(i))^{-\beta}$, normalised by $\max w$. This is the "capped and corrected" scheme spec §11.3 asks for, used here for dreams only (spec eq. S7).

### 5.6 Cited, not re-read (standard references the specification relies on)

- Kirkpatrick et al., EWC, PNAS 2017 (referenced through Progress & Compress).
- Hinton, Vinyals, Dean, "Distilling the Knowledge in a Neural Network", arXiv 1503.02531.
- Ross, Gordon, Bagnell, DAgger (AISTATS 2011): the on-policy imitation argument GKD builds on.
- Hafner et al., Dreamer (ICLR 2020), and Ha and Schmidhuber, World Models (2018): "dreams" as latent rollouts for policy learning, cited by the Sleep paper.
- McClelland, McNaughton, O'Reilly 1995, complementary learning systems: cited by the Sleep paper.

## 6. Synthesis: conclusions drawn from the reading

1. **The paper's consolidation is distillation because its sender and receiver sit at different locations.** Here, fast and slow slots sit on the same projections with the same rank, so a **transplant is exact** (spec Theorem 9.2). Distillation is needed only for lossy moves: compaction, belief→weights, and selective consolidation. This conclusion agrees with the weight-space results of Progress & Compress and of the "grafting" paper.
2. **On-policy distillation carries over cleanly to latent rollouts.** The repository's detached autoregressive feedback is exactly GKD's "no backprop through student sampling".
3. **LTI's RL is unnecessary for a differentiable planner cost.** The meaningful "behaviour" is the action ranking, and a pairwise logistic surrogate (already used in spec §15.6 for the cost surrogate) suffices.
4. **Dreaming via self-generated targets is a no-op for a deterministic predictor** (spec Proposition 9.10). It is also unsafe for a world model (spec §7.6). Informative dreams need a teacher with more information:
   - hindsight-smoothed slow context;
   - older frozen models for retention, the analogue of LwF and generative replay.
5. **Recurrence makes forgetting costly.** Both the dynamics-shift replay study and MOLe show that the A→B→A benchmark is the right primary test, and that age-only forgetting is the wrong default.
6. **Resetting fast parameters restores plasticity.** Combined with exact transplant, a reset loses no function. This is a concrete, testable benefit even if distillation turns out to add nothing (spec experiments E-PLAST and E-KS1).
