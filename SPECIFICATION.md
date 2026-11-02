# RoboJEPA with persistent test-time adaptation and a recurrent belief state

**Target:** SO-101, RGB cameras and joint-position feedback, no force/torque/tactile sensor, cloud GPU training and inference. **Status:** proposed research/engineering specification; not a trained or robot-validated controller. Prepared 2026-11-02–03.

The recommended combination is

\[
\boxed{\text{frozen visual representation}
+\text{command-to-motion dynamics}
+\text{recurrent probabilistic filtering}
+\text{persistent predictor adaptation}
+\text{latency-aware MPC}.}
\]

The central distinction is between **inferring what is happening now**, **inferring which dynamics currently apply**, and **learning parameters that improve future predictions**. These are different problems, with different update rates, reset rules, and failure modes. A replay buffer, an RNN hidden vector, and updated weights are all forms of memory, but none alone is a calibrated belief over the current physical world.

All numerical values are **proposed starting defaults** unless explicitly identified as inspected checkpoint properties. Hardware limits are measured configuration inputs. The source audit identifies inherited mechanisms and discrepancies. A smaller correct system must pass its gates before adding the next component.

## 1. Scope and assumptions

The system should predict consequences of issued SO-101 commands, maintain a state estimate and uncertainty, preserve information through temporary occlusion, adapt from observed transitions without online task-reward labels, retain useful knowledge across episodes, and plan subject to local limits and computation deadlines.

Assume a local computer can read follower joint positions, issue joint targets, timestamp events, and run a small supervisor. Assume at least one fixed RGB camera sees the task; two views, fixed external and wrist, are recommended if available. The specification does not assume the user already owns two cameras. Initial tasks are slow tabletop manipulation with frequent visual observability. Calibration, joint ordering, units, gripper direction, and action limits must be measured.

A camera-only system cannot determine arbitrary hidden contact forces, infer an occluded object's exact position without information, identify every physical parameter from passive observations, or guarantee collision avoidance from uncertain images. Joint tracking error is not a force measurement. The design represents uncertainty and restricts action when evidence is insufficient.

“Real time” is specified per component. Local joint estimation can update at the measured device rate. Cloud visual estimation can assimilate arriving keyframes and propagate to the present, but information is as old as the capture/network pipeline. Large-model CEM may take seconds. An asynchronous implementation is not automatically a real-time implementation.

## 2. What combining the papers means

RoboJEPA supplies a pretrained spatial latent predictor and goal-image planning. JEPA-TTT supplies persistent self-supervised adaptation: store observed transitions, predict a frozen representation, update dynamics, and retain parameters, optimizer, and replay across episodes. Their representations and task objectives are not plug compatible; JEPA-TTT's reward head cannot simply be attached to RoboJEPA's token grid. [RoboJEPA](https://arxiv.org/html/2610.10515v1), [JEPA-TTT](https://arxiv.org/html/2610.00722v1).

The minimal combination is a baseline: freeze the encoder and goal metric; update the RoboJEPA predictor on causal observed transition windows; retain updates and optimizer moments across episodes. The full proposal adds a recurrent probabilistic estimator and a command-conditioned motion model. These additions are hypotheses, not results established by either paper.

The user can choose a goal image without reproducing JEPA-TTT's goal-image-free reward-head setting. The essential transferable mechanism is persistent dynamics learning, not the particular task-score interface.

## 3. SO-101 interface: commands are not movements

### 3.1 Coordinates

Use ordered physical quantities

\[
p=(q_1,q_2,q_3,q_4,q_5,g),
\]

where arm joints are radians and \(g\in[0,1]\) is a calibrated aperture proxy. Define 0 as closed and 1 as open only after verifying device direction. This proxy is neither metric aperture nor force.

Order: shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, gripper. The inspected LeRobot implementation supports degree or normalized body-joint modes, a separate gripper range, and target clipping. Save the installed version and calibration hash. [Follower implementation](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py).

Compact-model coordinates are

\[
\bar p_j=(p_j-c_j)/s_j,\qquad s_j>0.
\]

Use verified allowed-range midpoint and half-range for the arm; \(c_g=s_g=0.5\) for the gripper. Save these constants. Compact coordinates do not replace legacy RoboJEPA units.

### 3.2 Three distinct action records

At raw tick \(n\), record requested target \(u_n^{\rm req}\), issued target \(u_n^{\rm sent}\) after all clamps, and subsequently observed movement \(\Delta p_n^{\rm obs}=p_{n+1}-p_n\). Record send time, acknowledgment if available, and measurement times. A successful write does not prove that the arm reached its target.

The online predictor receives issued commands and timing. Realized movement is a supervised outcome.

### 3.3 The required command bridge

The inspected SO-101 joint checkpoint conditions on measured state differences. The repository also documents a joint-state propagation mismatch in its generic rollout path. [SO-101 source documentation](https://github.com/facebookresearch/robo_jepa/blob/f93106f5f4a451fd526d51778aa8ab7c852eb824/app/robo_jepa/docs/action_spaces/so101.md).

For issued command block \(U_t\), deployment requires

\[
p(z_{t+1}\mid\mathcal H_t,U_t)
=\int p(z_{t+1}\mid\mathcal H_t,\Delta p_t,U_t)
p(\Delta p_t\mid\mathcal H_t,U_t)\,d\Delta p_t. \tag{1}
\]

The pretrained predictor approximates part of the first factor. Substituting a commanded target difference for measured displacement assumes perfect tracking. Training only on realized displacement can hide actuator failure because the hindsight action label already incorporates that failure.

The compact model below supplies command-to-motion uncertainty. A command-conditioned visual residual handles effects not determined by displacement alone, including contact with little visible arm motion. This is a modeling decomposition, not a claim that displacement alone is causally sufficient.

### 3.4 Legacy transforms and required changes

Resolve `so-101_joints` by name in the loaded checkpoint. Preserve its 64-dimensional zero padding. Reproduce the legacy gripper transform explicitly: source value times \(\pi/180\), not automatically the new \(g\). If the verified source convention is \(100g\), the legacy value is \(100g\pi/180\); otherwise apply the correct calibrated affine mapping. Do not convert arbitrary normalized body-joint values as though they were degrees.

Required code changes:

1. Joint propagation: \(p'_{1:6}=p_{1:6}+\Delta p_{1:6}\), no Euler rotation composition; padding remains zero.
2. A six-dimensional joint-target CEM sampler and command translator. The inspected CEM code assumes Cartesian/rotation/gripper coordinates; changing `action_dim` alone is insufficient. [Planner source](https://github.com/facebookresearch/robo_jepa/blob/f93106f5f4a451fd526d51778aa8ab7c852eb824/app/robo_jepa/mpc/mpc_utils.py).
3. A learned command-to-realized-motion bridge.
4. Verified camera IDs, preprocessing, units, and gripper direction.
5. A test proving that no future measured joint state enters command-conditioned rollouts.

Plan initially in the five arm joints plus gripper. Do not assume six independently controllable Cartesian pose coordinates. Any later IK path needs actual robot geometry and reachable-pose projection.

## 4. Clocks, history, and persistence

Raw I/O targets 30 Hz, subject to measurement. The world model uses nominal \(\Delta=0.2\) seconds, six raw ticks. Actual timestamps are authoritative.

\[
\mathcal H_t=\{o_{\le t}^{\rm arrived},p_{\le t}^{\rm arrived},
u^{\rm sent}_{<t},\text{timestamps, validity, calibration, resets}\}.
\]

Store capture and receipt times. An image not yet received is unavailable even if its capture time is old. \(U_t\) contains ordered issued targets, durations, and validity over the macro interval. Retain the preceding one second of issued commands for delay modeling. Never sum absolute targets.

The persistent system state is

\[
\mathcal S_t=(b_t,h_t,\Theta_t,\mathcal M_t), \tag{2}
\]

where \(b_t\) is current uncertainty, \(h_t\) recurrent history, \(\Theta_t\) parameters/optimizer/version history, and \(\mathcal M_t\) replay/regime/calibration/event memory. Visual contexts and KV caches are derived runtime objects.

| Event | Fast state and scene memory | Slow knowledge | Parameters and replay |
|---|---|---|---|
| New goal, continuous scene | Continue; replace goal | Continue | Continue |
| Episode label changes without physical reset | Continue | Continue | Continue |
| Human resets objects or arm | Reobserve and reset affected scene state | Retain a broadened compatible prior | Retain; mark boundary |
| Camera moves | Invalidate affected visual context and goal | Do not interpret visual error as actuator change | Archive old version; suspend incompatible updates |
| Robot power cycle | Read positions; clear pending commands and fast recurrence | Restore broad compatible prior | Restore compatible learning state; never resume old motion |
| Reconnect with complete event log | Replay to reconstruct | Continue if validated | Restore atomically |
| Unlogged gap | Propagate briefly with increasing uncertainty, then reobserve | Broaden | Retain knowledge; prohibit confident continuation |

Persistence means retaining the right information, not every numeric value indefinitely.

## 5. Exact inference target and its limits

Let \(s_t\) be physical scene state, \(d_t\) dynamics, \(r_t\) regime, and \(\ell_t\) actuation/communication state. Ideally,

\[
b_t^\star=p(s_t,d_t,r_t,\ell_t\mid\mathcal H_t).
\]

Writing their continuous components as \(x\), the exact Bayesian recursion is

\[
b^-_{t+1}(x',r')=\sum_r\int
p(x',r'\mid x,r,U_t,\Delta t)b_t(x,r)\,dx,
\]
\[
b_{t+1}(x',r')=
\frac{p(y_{t+1}\mid x',r')b^-_{t+1}(x',r')}
{\sum_j\int p(y_{t+1}\mid v,j)b^-_{t+1}(v,j)\,dv}. \tag{3}
\]

This defines the semantics. The implementation uses a learned compact model and interacting Gaussian approximations. It is a belief within that model, not an exact physical posterior.

If two hidden states generate identical available-observation distributions for every admissible action, no estimator can distinguish them. If discriminating actions are unavailable or unsafe, the states remain operationally unidentifiable. The correct output is uncertainty or a restricted policy.

A stationary joint can reflect contact, clipping, bus failure, obstruction, backlash, or actuator response. With these sensors the causes may remain ambiguous. Learned context coordinates are predictive variables, not certified mass/friction/force estimates.

## 6. Concrete architecture

### 6.1 Frozen visual backbone

Start with released RoboJEPA-300M for pipeline validation. Compare a released 2B predictor once data and latency justify it. The inspected 300M configuration has width 1024 and 24 blocks; the 2B has width 2048 and 40 blocks. Both use a frozen V-JEPA 2.1 representation. At 240 by 320 pixels, each view yields 300 tokens of 1664 channels. [300M config](https://github.com/facebookresearch/robo_jepa/blob/f93106f5f4a451fd526d51778aa8ab7c852eb824/configs/released/wm-300m/config.json), [2B config](https://github.com/facebookresearch/robo_jepa/blob/f93106f5f4a451fd526d51778aa8ab7c852eb824/configs/released/wm-2b/config.json).

Use the released image-mode preprocessing and ensure it uses no future frames. Preserve a golden input/output fixture. Let

\[
z_t\in\mathbb R^{V\times15\times20\times1664}.
\]

The proposed metric uses channelwise, per-token

\[
\mathrm{LN}_0(v)=\frac{v-\operatorname{mean}v}
{\sqrt{\operatorname{mean}(v-\operatorname{mean}v)^2+10^{-5}}}.
\]

Verify normalization placement and epsilon against the loaded checkpoint before claiming exact reproduction. Preserve the encoder input path. A pixel decoder is unnecessary for control and adaptation.

### 6.2 Frozen compact observation sketch

Do not multiply a million correlated token likelihoods as independent evidence. Build fixed \(\psi\):

1. Canonical token normalization.
2. Adaptive spatial pooling from 15-by-20 to 3-by-4 cells per view.
3. Flatten 12 cells, giving 19,968 values per view.
4. Fit 64 PCA coordinates per semantic camera using offline training episodes only.
5. Center/scale projected coordinates with fixed training statistics and variance floors.

Freeze \(\psi\) and its statistics. Two views give \(y_t^v\in\mathbb R^{128}\). One-view operation masks the absent block. A changed camera is a new observation contract.

\[
y_t=[\bar p_t;y_t^v]\in\mathbb R^{134}. \tag{4}
\]

If the sketch loses the relevant object information, increase to 128 coordinates per view or add a validated object observation, then retrain/recalibrate the observation model. A smaller sketch is not automatically sufficient.

### 6.3 Recurrent state and dimensions

\[
x_t=[\bar p_t^{\rm latent};\xi_t;d_t]\in\mathbb R^{64},
\quad \xi_t\in\mathbb R^{48},\quad d_t\in\mathbb R^{10},
\quad h_t\in\mathbb R^{256}. \tag{5}
\]

The first six coordinates are actual normalized joint/gripper state; \(\xi\) is fast hidden state; \(d\) is slow predictive context.

A shared one-layer GRU updates

\[
h_t=\operatorname{GRU}_{\rho}\left(h_{t-1},
[\tilde y_t^v,\bar p_t^{\rm available},e(U_{t-1}),m_t,a_t^{\rm age},\Delta t]\right). \tag{6}
\]

Its input is 128 visual + 6 joint + 64 command + 2 view-mask + 2 view-age + 1 interval = 203 values. Missing visual coordinates are replaced by prior means and masked. They do not become measurements. Missing joint feedback blocks new motion in the initial deployment profile.

Update \(h_t\) only **after** assimilating observation \(t\). The likelihood of \(y_t\) uses \(h_{t-1}\), preventing same-frame leakage.

Command encoder \(e\): GRU width 64, input issued six-dimensional normalized targets, time increment, validity. Process the ordered one-second queue plus current block; use explicit logged history, not unversioned hidden state.

### 6.4 Compact transition model and servo bridge

For regime \(r\),

\[
x_{t+1}=F_{\eta_r}(x_t,h_t,U_t,\Delta t)+\epsilon_t,
\qquad\epsilon_t\sim\mathcal N(0,Q_{r,t}). \tag{7}
\]

Use a shared three-hidden-layer MLP, width 256, SiLU, input \([x,h,e(U),\Delta t,e_r]\), with 16-dimensional regime embedding. Output six joint residuals and 48 fast-state residuals. Slow deterministic transition is \(F_d(d)=d\).

A nominal servo model integrates each actual command duration:

\[
p_{n+1}^{\rm nom}=p_n^{\rm nom}
+(1-e^{-\delta_n/\tau})\odot
\left(u^{\rm sent}(t_n-\ell)-p_n^{\rm nom}\right). \tag{8}
\]

Fit per-joint \(\tau>0\) and delay \(\ell\) on isolated motions and reversals; compare with a second-order model on held-out data. This initializes a useful approximation, not an asserted hardware law.

The joint output is the normalized nominal prediction plus \(a_{\max}^{\rm res}\odot\tanh f_p\). Derive the residual bound from validated movement per interval. Fast coordinates use

\[
F_\xi=e^{-\Delta t/(10\,{\rm s})}\xi
+(\Delta t/0.2\,{\rm s})\,0.1\tanh f_\xi. \tag{9}
\]

This is an initial parameterization; relax it offline if restrictive. Do not silently clip posterior states.

Initialize process variance per 0.2 s to \(10^{-4}\) for joints, \(10^{-3}\) for \(\xi\), \(10^{-5}\) for \(d\). Learn bounded diagonal noise offline: \([10^{-6},10^{-1}]\) for joint/fast components and \([10^{-8},10^{-2}]\) for slow components. Scale continuous-time intensity with actual interval and calibrate. Online variance learning is disabled initially.

### 6.5 Compact observation model

\[
y_t=H_r(h_{t-1},e(U_{t-1}))x_t+
b_r(h_{t-1},e(U_{t-1}))+\nu_t,
\qquad\nu_t\sim\mathcal N(0,R_{r,t}). \tag{10}
\]

The first six rows of \(H\) are \([I_6,0]\) with zero bias. A two-hidden-layer MLP, width 256, produces 128 visual rows and their bias; initialize/bound matrix outputs with \(0.1\tanh\). Inputs are past history and commands, never the current observation.

Use diagonal \(R\), calibrated separately for joints and each view. Measure joint sensor repeatability independently of tracking error. Visual initial standard deviation is 0.2 standardized PCA units; joint noise must be measured. Full covariance or Student-t likelihoods are later alternatives, not version-1 defaults.

This is a likelihood for a compact feature sketch, not pixels. Full token losses are auxiliary training objectives and are not multiplied into the online likelihood.

### 6.6 Full visual predictor

For the final eight RoboJEPA blocks add regime-specific rank-8 LoRA on Q, V, and attention-output projections:

\[
W^{(r)}=W_0+\frac{\alpha}{k}B^{(r)}A^{(r)},
\qquad k=8,\quad\alpha=16. \tag{11}
\]

Initialize \(A\) with standard small random linear initialization, \(B=0\). Handle fused QKV slices and grouped-query widths explicitly. Log matched module names and fail on an unexpected count.

Convert compact predicted movement to legacy units and compute

\[
z^0_{t+1}=g_{\theta_0,\delta_r}
(\mathcal C_t,\Delta p_t^{\rm predicted},p_t).
\]

Add a belief/command-conditioned residual:

\[
\hat z_{t+1,vj}=\mathrm{LN}_0\left[
z^0_{t+1,vj}+B_o\tanh\left(A_o z^0_{t+1,vj}
+M([h_t,x_{t+1},e(U_t),e_r,e_v,\operatorname{pos}(j)])\right)
\right]. \tag{12}
\]

Use bottleneck 32, \(A_o:1664\to32\), \(B_o:32\to1664\), zero-initialize \(B_o\); \(M\) has two width-256 hidden layers, 32 outputs, view embedding width 8, and normalized row/column position. Regime-specific final residual modules let recurring dynamics retain their own corrections.

At zero residual/adapters, check base-model equivalence under the fixed normalization contract. Online freeze encoder, sketch, GRU, observation model, noise, backbone, and latent coordinate system. Adapt only selected LoRA matrices, residual modules, and a small regime-specific mean correction in \(F\).

The compact likelihood model and full token predictor can disagree. Consistency training and disagreement monitoring are required. They are not one exact generative pixel model: filtering uses the compact likelihood, planning uses spatial token consequences. This tractable separation is a material approximation.

### 6.7 The recurring belief

\[
b_t(x,r)=\pi_t(r)\mathcal N(x;\mu_t^{(r)},P_t^{(r)}),
\qquad 1\le K\le4. \tag{13}
\]

Together with \(h_t\), timestamps, command queue, and recent token context, this is recurrent state. Start with \(K=1\); enable additional calibrated regimes only with supporting data. Empty slots are not meaningful hypotheses.

Regimes denote persistent operating conditions, not automatically contact/slip labels. Fast transient contact belongs in \(\xi\); slow differences belong in \(d\), regime identity, and eventually predictor parameters.


## 7. Recursive filtering algorithm

### 7.1 Regime prediction and mixing

For \(K>1\), a starting transition matrix is sticky:

\[
T_{ij}(\Delta t)=
\begin{cases}
1-\epsilon,&i=j,\\
\epsilon/(K-1),&i\ne j,
\end{cases}
\quad
\epsilon=1-e^{-\Delta t/(600\,{\rm s})}. \tag{14}
\]

This encodes a prior mean residence scale, not a measured property. Tune on regime-change validation sequences; for \(K=1\), \(T=[1]\). Do not count a task boundary as a dynamics change automatically.

Given posterior mode weights \(\pi_{t-1}\),

\[
\pi^-_t(j)=\sum_i T_{ij}\pi_{t-1}(i),\qquad
\omega_{i\mid j}=\frac{T_{ij}\pi_{t-1}(i)}{\pi^-_t(j)}.
\]

Mix continuous distributions using moment matching:

\[
\bar\mu^{(j)}=\sum_i\omega_{i\mid j}\mu_{t-1}^{(i)},
\]
\[
\bar P^{(j)}=\sum_i\omega_{i\mid j}
\left[P_{t-1}^{(i)}
+(\mu_{t-1}^{(i)}-\bar\mu^{(j)})
(\mu_{t-1}^{(i)}-\bar\mu^{(j)})^\top\right]. \tag{15}
\]

The between-mean term is essential. Dropping it becomes falsely certain when modes disagree. Moment matching already approximates the full exponential-growth Gaussian mixture; this is an interacting multiple-model, extended-Kalman-style filter.

### 7.2 Prediction

\[
\mu_t^{-,j}=F_{\eta_j}(\bar\mu^{(j)},h_{t-1},U_{t-1},\Delta t),
\]
\[
A_t^{(j)}=\left.\frac{\partial F_{\eta_j}}{\partial x}\right|_{\bar\mu^{(j)}},
\qquad
P_t^{-,j}=A_t^{(j)}\bar P^{(j)}A_t^{(j)\top}+Q_{j,t}. \tag{16}
\]

Compute the Jacobian through the small compact model, not through the billion-parameter visual predictor. Use vectorized autodiff and benchmark its latency. This is a first-order nonlinear approximation; test a cubature/unscented or particle alternative if nonlinear residuals or multimodality defeat it.

### 7.3 Measurement assimilation

Let \(M_t\) select only valid observed coordinates. Construct

\[
\tilde y=M_ty_t,\quad \tilde H=M_tH,\quad\tilde b=M_tb,\quad
\tilde R=M_tR M_t^\top.
\]

For each mode,

\[
e_t^{(j)}=\tilde y-\tilde H\mu_t^{-,j}-\tilde b,
\qquad
S_t^{(j)}=\tilde H P_t^{-,j}\tilde H^\top+\tilde R,
\]
\[
K_t^{(j)}=P_t^{-,j}\tilde H^\top(S_t^{(j)})^{-1},
\qquad
\mu_t^{(j)}=\mu_t^{-,j}+K_t^{(j)}e_t^{(j)}. \tag{17}
\]

Use Cholesky triangular solves, never an explicit inverse. Update covariance with Joseph form:

\[
P_t^{(j)}=(I-K\tilde H)P_t^{-,j}(I-K\tilde H)^\top
+K\tilde R K^\top. \tag{18}
\]

Symmetrize \((P+P^\top)/2\), add minimal numerical jitter only when needed, and log every repair. Default FP32 for covariance, likelihood, and solves; FP64 in mathematical checks. BF16 is appropriate for large neural layers, not covariance factorizations.

The predictive log likelihood is

\[
\lambda_j=-\frac12
\left[e^\top S^{-1}e+\log\det S+d_{\rm obs}\log(2\pi)\right],
\]

and

\[
\log\pi_t(j)=\log\pi^-_t(j)+\lambda_j
-\operatorname{logsumexp}_k(\log\pi^-_t(k)+\lambda_k). \tag{19}
\]

If there are no valid coordinates, skip measurement correction, use \(\pi_t=\pi^-_t\), and retain predicted covariance. Update the recurrent summary using masked predictions; do not call that update a measurement.

The full posterior mean/covariance, when an API needs a single summary, are

\[
\mu=\sum_j\pi_j\mu_j,\qquad
P=\sum_j\pi_j[P_j+(\mu_j-\mu)(\mu_j-\mu)^\top]. \tag{20}
\]

Planning should sample the mixture rather than replace opposing hypotheses with an impossible average state.

### 7.4 Initialization

After startup, use measured normalized joints for the first six means; set \(\xi,d\) means to zero or a compatible saved slow prior. Joint variance is measured observation noise; latent fast and slow variances initially 1. Set \(h=0\), clear command queue, and collect up to 20 macro frames of stationary/limited-motion burn-in.

Initialize weights uniformly across active compatible regimes unless metadata legitimately supplies a prior. A previous mode may inform the prior after a short reconnect, but a robot power cycle is not evidence that the scene is unchanged.

A missing camera is not a zero image. Remove its likelihood rows; store a masked token context and age. Do not pass a missing-camera tensor into an unmodified pretrained predictor as if it were a real view. Use its supported one-view path where applicable, or use last valid/predicted tokens only after missing-view training and with explicit uncertainty.

### 7.5 Calibration and robust observations

Fit covariance multipliers on episode-disjoint calibration data. For normalized innovations \(e^\top S^{-1}e\), examine empirical distributions by view, motion speed, occlusion, regime, and horizon. A Gaussian model predicts a chi-square law only under strong correctness assumptions; use that as a diagnostic, not proof.

Inflate \(R\) for measured blur, saturation, synchronization error, and partial visibility using a fixed quality policy calibrated offline. If quality is unacceptable, omit that observation block. Do not inflate \(R\) just because the model is wrong until every error disappears: preserve pre-inflation residuals and classify failures separately.

Full covariance on 134 observations may eventually be useful, but initially diagonal \(R\) plus calibration is easier to debug. PCA removes some linear correlation; it does not establish independent or Gaussian evidence.

### 7.6 Missing observations and imagined observations

No observation means prediction only. Covariance can contract in a genuinely stable known dynamical system, but the implementation must not perform a measurement update using its own generated output.

During planning, a hypothetical future image has not been observed. Generate hypothetical states and recurrent histories without applying an observation update that treats those images as truth. The default planner is open-loop within its horizon and replans after real observations. It is not an exact belief-space dual-control solver.

### 7.7 Fast local observer and cloud reconciliation

Run a separate small local joint observer at the actual I/O rate, using joint positions, command history, and a calibrated servo model. It supports stopping, interpolation, and freshness checks. It does not claim to know the current object state.

The cloud filter runs at macro keyframes. Its belief message contains:

- state capture time and processing time;
- model and calibration versions;
- mixture means/covariances and regime weights;
- per-view last actual observation time;
- incorporated command-log sequence number;
- validity/degraded flags.

To display or plan at a newer time, propagate through the *actual issued* command log. Incorporating a late image requires fixed-lag replay of chronological events, or dropping it from live filtering and keeping it for training. Never apply a late measurement directly to the current state as though it were current.

Initial fixed-lag window: 2 seconds. Keep snapshots at macro boundaries plus raw action/observation events. On rewind, restore the earlier snapshot and reprocess subsequent observations exactly once. Joint observations previously fused into a snapshot must not be counted again as independent evidence. Longer gaps require a new state initialization.

## 8. Regime changes and recurring dynamics

A single ever-growing uniform buffer is reasonable for a fixed shift but can learn an average of incompatible dynamics under A→B→A recurrence. The proposal uses persistent per-regime adapters and replay partitions, with continuous context \(d\) for within-regime variation.

The initial production candidate uses one regime. Add a bank of up to four only after controlled data establish useful distinctions. Train active mode identities using known intervention labels when available; otherwise cluster *transition residual sequences* offline and validate predictiveness. Do not cluster images alone and call the clusters dynamics.

### 8.1 Change monitor

Let \(v_t\) be a scalar pre-update discrepancy, such as normalized innovation or a calibrated combination of joint and visual innovations. Standardize it with no-shift calibration mean and standard deviation:

\[
a_t=(v_t-\mu_{\rm cal})/\max(\sigma_{\rm cal},10^{-3}),
\qquad C_t=\max(0,C_{t-1}+a_t-\kappa). \tag{21}
\]

Start with \(\kappa=0.5\), alarm threshold \(C_t>10\), and require five valid macro observations before acting on an alarm. These are calibration candidates, not statistical false-alarm guarantees. Temporal correlation makes a theoretical i.i.d. threshold unreliable.

Log separate channels for command tracking, visual innovation, camera quality, network timing, and model uncertainty:

| Evidence | Initial response |
|---|---|
| Joint mismatch with valid timestamps and normal imagery | Increase command-model uncertainty; consider dynamics adaptation |
| Visual mismatch after camera movement or exposure change | Observation fault workflow; suspend dynamics-only interpretation |
| Normal joints, altered object response | Consider contact/object dynamics; keep ambiguity |
| Missing/duplicate/out-of-order packets | Repair time alignment; do not train on corrupted transitions |
| Joint and image anomaly with possible physical disturbance | Restrict movement, reobserve, inspect multiple hypotheses |

A detector identifies inconsistency, not its cause. Current observation likelihood updates mode probabilities before any parameter update. Storing those pre-update responsibilities prevents a newly fitted model from inventing retrospective certainty about its own training examples.

### 8.2 Mode switching and new modes

Switch the planner's dominant adapter only after posterior probability exceeds 0.8 for five valid macro observations; meanwhile evaluate all modes with probability above 0.05 when compute permits. If this budget is exceeded, stop/shorten action rather than silently erase uncertainty.

A genuinely new regime enters quarantine:

1. Preserve the active and previously validated regime adapters.
2. Copy the closest compatible adapter into an unused candidate slot.
3. Collect at least 256 valid new command transitions spanning more than one motion direction and task phase.
4. Train the candidate in shadow.
5. Evaluate on later, unused transitions and old-regime anchors.
6. Activate only after the update gate in §13 and a compatible calibration check.

The initial release does not automatically evict old modes. If all four slots are occupied, keep the new regime in a bounded quarantine buffer and require offline consolidation or an explicitly evaluated eviction policy. “Lifelong memory” with fixed storage cannot retain every distinct environment perfectly.

When an old regime recurs, use its stored adapters and optimizer state only if calibration, embodiment, and model lineage match. Recompute current belief from recent evidence; do not restore an old object pose.

## 9. Loss functions and gradient boundaries

### 9.1 Masked token metric

For valid view/patch mask \(m_{tvj}\) and nonnegative fixed weights \(w_{vj}\), define

\[
D_z(\hat z,z)=
\frac{\sum_{v,j,d}m_{vj}w_{vj}
|\mathrm{LN}_0(\hat z_{vjd})-\mathrm{LN}_0(z_{vjd})|}
{D_e\sum_{v,j}m_{vj}w_{vj}},\quad D_e=1664. \tag{22}
\]

If the denominator is zero, skip that loss; do not divide by epsilon and count it as a good prediction. Average valid frames within a sequence, then average sequences, so different view counts and lengths do not implicitly reweight the objective.

Initial weights are uniform for model training. Optional task foreground masks affect the planning objective separately. Changing training weights is an explicit ablation.

### 9.2 Three visual prediction objectives

**Realized-motion diagnostic/pretraining path:**

\[
L_{\rm realized}=
\mathbb E D_z(g(\mathcal C_t,\Delta p_t^{\rm obs},p_t),z_{t+1}). \tag{23}
\]

This matches the legacy action semantics. It measures visual transition knowledge *given* the motion, not deployable command prediction.

**Command teacher-forcing path:**

\[
L_{\rm cmdTF}=
\mathbb E_{w,t,x'\sim b^-_{t+1}}
D_z(G(\mathcal C_t,U_t,x'),\operatorname{sg}z_{t+1}). \tag{24}
\]

Contexts contain observed past frames and the belief filtered through \(t\), never the posterior through \(t+1\). Predicted motion comes from \(F\). Use two antithetic latent samples per sequence initially, or the prior mean in the deterministic baseline. Report which estimator is used.

**Autonomous command rollout:**

\[
L_{\rm AR}^{(H)}=
\mathbb E_{w}\frac{1}{\sum_{k=1}^H\gamma_{\rm tr}^{k-1}}
\sum_{k=1}^H\gamma_{\rm tr}^{k-1}
D_z(\hat z_{t+k}^{\rm free},\operatorname{sg}z_{t+k}), \tag{25}
\]

with \(\gamma_{\rm tr}=0.95\), causal observed prefix only, issued future command sequence as known training inputs, predicted future joints, no intermediate observed frames, and no future posterior states. Update imagined recurrence from predicted features/joints; do not run measurement corrections.

Initially detach predicted token inputs between steps, matching a stable truncated rollout-training approach. This means later losses do not backpropagate through earlier visual predictions; state this rather than implying full backpropagation through imagination. Preserve gradients through the current predictor application. Later test full BPTT over two to five compact steps as an ablation.

### 9.3 Compact predictive negative log likelihood

At each valid time step before assimilation,

\[
L_{\rm NLL}=
-\frac{1}{d_{\rm obs}}
\log\sum_r\pi^-_t(r)\,
\mathcal N(M_ty_t;M_t(H_r\mu^-_r+b_r),S_r). \tag{26}
\]

This is predictive marginal NLL under the approximate filter. It is not an NLL computed from a posterior already conditioned on the target. Dimension normalization is for training balance, not a temperature modification of the runtime posterior.

Offline train transition, emission, GRU, and bounded covariance parameters using differentiable filtering. Reference implementation differentiates through Jacobians and covariance recursion, requiring second-order autodiff for the EKF linearization. Start on short sequences. A cheaper variant that detaches Jacobians/covariances is allowed only as an explicitly named surrogate-gradient ablation; it does not optimize the exact derivative of (26).

The likelihood remains approximate because of Gaussian projection and learned representation. Evaluate calibration independently.

### 9.4 Joint prediction loss and bridge training

For future joint mean \(\hat p\), use

\[
L_p=\frac1{6}\sum_j
\operatorname{Huber}_1\left((\hat p_j-p_j)/s_j^{\rm loss}\right), \tag{27}
\]

where \(s_j^{\rm loss}\) is the training-set robust one-step movement scale with a positive noise floor. Use Huber \(\frac12e^2\) for \(|e|\le1\), \(|e|-\frac12\) otherwise.

Use issued-command inputs and measured outputs. Evaluate delays separately. Start by fitting nominal servo parameters, then compact residual dynamics. While training the visual model, stop gradients from visual losses into the command-to-motion output. Otherwise the servo model could invent an incorrect joint motion merely because it helps explain an image. Joint/compact likelihood objectives train physical tracking.

### 9.5 Compact/full consistency

Let \(\hat y_{\rm compact}\) be the compact observation mean and \(\psi(\hat z_{\rm full})\) the full predicted feature sketch. Use

\[
L_{\rm cons}=
\frac1{128}\left\|
\psi(\hat z_{\rm full})-\operatorname{sg}(\hat y^v_{\rm compact})
\right\|_1. \tag{28}
\]

Initial training updates the full residual with this term after compact pretraining; the compact target is stopped. This prevents two models from colluding to make the same error. Keep consistency weight small and always retain observation-supervised losses. Log both models' errors against real observations, not just agreement with each other.

### 9.6 Regularization and retention

For adapters, use functional anchoring on compatible reference windows:

\[
L_{\rm anchor}=
\mathbb E_{w\sim A_r}D_z(
G_{\Theta_r}(w),\operatorname{sg}G_{\Theta_r^{\rm ref}}(w)). \tag{29}
\]

Only anchor to the reference of the same regime. Forcing shifted dynamics to imitate an incompatible nominal model prevents adaptation.

A parameter penalty can supplement it:

\[
L_{\rm prox}=\frac1{N_\theta}\sum_i
\left(\frac{\theta_i-\theta_i^{\rm ref}}
{\max({\rm RMS}(\theta^{\rm ref}_{\rm layer(i)}),10^{-3})}\right)^2. \tag{30}
\]

The floor matters for zero-initialized adapters. Do not use a raw LoRA factor norm as the only measure of functional drift: \(BA\) is unchanged by reciprocal factor rescaling. Functional anchoring and output-change tests are more informative.

### 9.7 Proposed stage objectives

All components are averaged as defined above.

- Compact pretraining: \(L_C=L_{\rm NLL}+0.5L_p+10^{-4}L_{\rm regularity}\).
- Visual bridge fine-tuning: \(L_V=0.25L_{\rm realized}+0.5L_{\rm cmdTF}+0.5L_{\rm AR}+0.05L_{\rm cons}\).
- Joint offline refinement: alternate compact and visual steps; do not let visual errors train the servo bridge.
- Online visual adaptation: \(L_{\rm onlineV}=L_{\rm cmdTF}+0.5L_{\rm AR}+0.1L_{\rm anchor}+10^{-4}L_{\rm prox}\).
- Online compact mean adaptation: \(L_{\rm onlineC}=L_{\rm NLL}+0.5L_p+0.1L_{\rm anchor,p}\), with covariance, observation map, GRU, and feature coordinates frozen.

Here \(L_{\rm regularity}\) is the average squared MLP weights on the fast residual output layer; ordinary AdamW weight decay applies separately as configured. \(L_{\rm anchor,p}\) is Huber distance between current and reference predicted joints on compatible anchors.

Weights are proposed. Monitor gradient norms of each component. If one term is more than ten times all others for sustained updates, inspect scaling before adjusting weights. Do not repeatedly tune against final test outcomes.

### 9.8 Frozen versus trainable contract

| Component | Offline | Online |
|---|---|---|
| V-JEPA encoder and canonical feature normalization | Frozen | Frozen |
| PCA sketch/statistics | Fit once on train split, then frozen | Frozen |
| Base RoboJEPA transformer | Frozen in recommended adaptation recipe | Frozen |
| Per-regime LoRA | Train | Small updates after gate |
| Command-conditioned token residual | Train | Selected output/modulation parameters |
| Compact transition / servo residual | Train | Small final mean-correction head |
| Compact observation matrix generator | Train | Frozen |
| Process/observation variances | Train and calibrate | Frozen initially |
| Recurrent GRU and latent coordinates | Train | Frozen |
| Goal image/metric or task-score head | Specify/train offline | Frozen during adaptation |
| Current belief means/covariances | Inferred | Inferred every valid observation |

Freezing the visual encoder removes one collapse route. It does not prevent predictor collapse, identity predictions, foreground neglect, overconfident likelihoods, or planner exploitation. These require explicit diagnostics.

## 10. Offline data and training curriculum

### 10.1 Data contract

Each raw record includes episode/session IDs; monotonic capture and receipt timestamps; RGB frame references and per-camera timestamps; measured positions and validity; requested and issued targets; clipping status; command sequence/acknowledgment IDs; calibration and software hashes; task/goal ID; physical-reset events; known intervention labels; and network statistics.

Keep raw frames/commands. Derived features carry encoder/preprocessing hashes. Corrupted or ambiguous transitions may be useful for diagnostics, but are excluded from supervised dynamics targets until timing is resolved.

Initial collection budget, not a sufficiency claim:

- 20–30 minutes: calibrated isolated joint motion, reversals, gripper motion, varied command rates, and pauses.
- 100–200 short exploratory episodes: free-space movement and gentle, supervised object interactions.
- 300–500 task demonstrations across object positions, failures/recoveries, camera nuisance, and relevant conditions.
- Expand toward 1,000–3,000 episodes only when learning curves justify more data; a prescribed episode count cannot guarantee coverage.

Use bounded structured motion, not random high-amplitude motor commands. Record failed attempts and tracking saturation. Dataset diversity should address observed blind spots, not merely increase nearly duplicate frames.

### 10.2 Splits

Split by entire recording session and episode before constructing windows: 70% training, 10% validation, 10% calibration, 10% final test as an initial allocation. Where objects/layouts recur, hold out meaningful object-position and session combinations. Avoid train/test overlap through near-identical adjacent clips or the same prerecorded image reused under a new ID.

Train/validation regimes may overlap, but final testing must include held-out combinations and genuinely new dynamics. Keep a separate recurrence benchmark A→B→A. Predeclare that goal images may be supplied as task specifications while their associated execution trajectories remain held out.

### 10.3 Stage 0: integration and identifiability

Before neural training, establish:

- unit round trips and gripper direction;
- correct joint ordering and clipping;
- clock synchronization/error bounds;
- delay between issued target and measured motion;
- one-step versus multistep prediction baselines;
- static-frame encoder determinism and nonzero feature variance;
- goal-distance sensitivity to the object versus arm/background;
- joint-specific rollout state propagation.

Fit the nominal servo model with constrained least squares/grid search over delay plus positive time constants. Suggested delay search: 0–300 ms at one raw-tick resolution, expanded if residual correlation indicates longer delay. Estimate a fractional delay only if sampling/timestamps support it.

If command logs are unavailable, the visual model can be trained on realized motion, but command-conditioned planning and actuator adaptation remain unvalidated. That is a blocking data-contract issue, not something to hide with a smaller learning rate.

### 10.4 Stage 1: fixed representation and observation sketch

Cache frozen encoder features and fit the PCA sketch on training data. Extract at macro rate for initial development; dense replay later requires raw-offset features on demand. Verify that camera viewpoints and crops match goals and deployment.

Fit a simple task-success probe or use manually labeled paired frames to test observability. For example, distinguish “object lifted and retained” from “gripper at lift pose without object.” If the representation cannot do this, improve camera coverage or add a validated object observation before increasing planner compute.

### 10.5 Stage 2: compact recurrent dynamics

Train 32-step sequences at 5 Hz (6.4 seconds), with 16-step burn-in when preceding data exist. Initialize a sequence from an episode start or from a stored compatible raw history; arbitrary recurrent resets must be modeled during training.

During burn-in update state but do not count losses. Detach once at the burn-in boundary for truncated BPTT. Initialize latent coordinates from broad priors and use the filtering likelihood thereafter. Include missing-view masks, timestamps, and reset boundaries.

Starting configuration: AdamW, LR \(3\times10^{-4}\), betas (0.9,0.99), epsilon \(10^{-8}\), weight decay \(10^{-4}\), gradient norm clip 1.0, effective batch 32 sequences. Warm up 1,000 updates, cosine decay to \(3\times10^{-5}\) over a maximum 40,000 updates. Evaluate every 500 updates, early stopping after eight evaluations without validation improvement. Use FP32 for compact filtering; mixed precision for the MLP is optional only after covariance tests.

Train \(K=1\) first. For \(K>1\), initialize modes from known persistent conditions and add a supervised regime cross-entropy term of weight 0.1 when labels exist. Do not supervise transient contact as a persistent condition. Without labels, offline residual clustering is only an initialization; retain modes only when validation likelihood and recurrence performance improve.

### 10.6 Stage 3: adapt the visual model to the actual robot

Freeze the compact filter temporarily. Train new visual modules on sequences of 16 macro frames, using observed prefixes and command-conditioned rollouts. Start rollout length 2 for 5,000 updates, length 5 for 10,000, and length 10 for a final up-to-5,000 updates if the task needs it. If early stopping triggers, do not perform meaningless remaining updates just to match the schedule.

Optimizer groups:

| Parameters | LR | Weight decay |
|---|---:|---:|
| LoRA factors | \(10^{-4}\) | 0.01 |
| New token residual/modulation | \(3\times10^{-4}\) | 0.01 |
| Existing embodiment projection, if explicitly unfrozen offline | \(10^{-5}\) | 0 |
| Biases and normalization scalars | same group LR | 0 |

Use AdamW betas (0.9,0.95), epsilon \(10^{-8}\), global clipping 1.0, BF16 neural computation, FP32 reductions/master optimizer state. Effective batch 16 sequences through gradient accumulation; start microbatch 1–2 and measure memory. Warm up 500 updates, then cosine decay to 10% of each group's initial LR. Use activation checkpointing where necessary.

The default keeps the existing embodiment projection frozen; unfreezing it is an offline ablation after unit correctness is established. Training a projection to compensate for wrong units is unacceptable.

No random horizontal flips or independent per-frame geometric transforms. They change action geometry. Use consistent crop parameters across a sequence and goals, or deterministic preprocessing. Small photometric augmentation is an offline robustness experiment; preserve unaugmented fixed targets for initial debugging.

### 10.7 Stage 4: alternating refinement and calibration

Alternate one compact objective step and one visual objective step for up to 5,000 additional updates at 0.1 times their stage peak LRs. Keep the visual-to-servo stop-gradient boundary. Refit covariance multipliers on calibration episodes afterward; never use calibration frames as training gradients.

Validate prior predictions at 1, 2, 5, and 10 macro steps, not only posterior reconstruction. Retain a checkpoint only if it improves command prediction and does not destroy base task sensitivity. Save compact and full model versions together.

### 10.8 Optional meta-training for fast adaptation

Do not add this to the first implementation. If ordinary adaptation is too slow, use regime/task episodes with a chronological support/query split. Starting from \(\delta_0\),

\[
\delta'_r=U^J(\delta_0;\mathcal D_r^{\rm support}),\qquad
L_{\rm meta}=\mathbb E_r L_{\rm cmd/AR}
(\delta'_r;\mathcal D_r^{\rm query}). \tag{31}
\]

Start \(J=3\), inner LR \(10^{-5}\), outer AdamW LR \(10^{-4}\), four regimes per outer batch. Query sequences must not overlap support windows. Use first-order MAML only if labeled as such; differentiating through AdamW moments and the filtering replay is a different, costlier algorithm. Benchmark against simply spending the same compute on ordinary fine-tuning.

## 11. Dense causal replay

### 11.1 Window construction

For raw observations \(o_n\) and raw issued actions \(u_n\), with stride \(K_s=6\), define a window starting at every raw offset \(i\):

\[
W_i=\left(
(o_{i+jK_s},p_{i+jK_s})_{j=0}^{L},
(U_{i+jK_s:i+(j+1)K_s-1})_{j=0}^{L-1}
\right). \tag{32}
\]

Admit a window only when all its target observations, command records, and timing alignments exist. No crossing physical resets, calibration changes, invalid command gaps, or incompatible encoder versions. This proposal uses explicit indexed availability; it is not the paper's exact dense-window admission schedule.

At 30 Hz with a 5 Hz model, dense offsets cover all six phases. That improves temporal coverage, not the number of statistically independent experiments. Separate episode-based data splits remain essential.

### 11.2 Store references, not duplicated clips

Store each raw frame and command once. Windows contain indices. Cache features on demand under encoder/preprocessing hashes. Use shared references and bounded caches.

For two views at 240 by 320, one BF16 feature frame is approximately

\[
2\times300\times1664\times2=1{,}996{,}800\ {\rm bytes}.
\]

That is about 36 GB per hour at 5 Hz or 216 GB per hour at 30 Hz, before metadata. Blindly storing repeated dense windows multiplies this unnecessarily. Keep compressed images and a bounded feature cache; measure re-encoding cost.

### 11.3 Bounded replay partitions

Initial capacities, measured in unique valid windows:

- recent FIFO: 10,000;
- per-regime reservoir: 20,000, up to four regimes;
- immutable compatible offline anchors: 5,000 total;
- quarantine/unknown: 5,000.

Maintain raw-episode garbage collection by reference counts. Retention of complete rare episodes may override random window eviction within a fixed byte budget.

Each online batch initially draws 50% recent current-compatible windows, 25% same-regime reservoir, 25% compatible anchors. With uncertain mode membership, use pre-update soft responsibilities as loss weights or draw from the mixture. Normalize weights per regime; do not train all adapters identically on all data. Freeze inactive-regime adapters.

Do not use prioritized replay in version 1. Large errors can be sensor faults or unmodeled observations. If added later, cap priorities and correct sampling weights for the intended objective. Recency bias is an intentional objective change and does not need to pretend to estimate uniform all-history risk.

### 11.4 Update-to-data ratio

Default online effective batch 16. Earn one optimizer-update credit per 16 newly admitted unique dense windows. Cap at one completed candidate update per second and a compute duty cycle of 10% during operation. These are different from “one update every frame forever.”

Do not train before at least 256 valid windows exist. A full batch must be available. Reserve credit only after a batch is constructed; failed numerical updates consume no credit but trigger investigation. Keep credit debt/backlog bounded at 100 updates; discard stale credits, not the audit log, when overloaded.

Microbatching and accumulation must normalize the total loss identically to effective batch 16. Online AR horizon starts at 2; increase to 5 only if measured duty-cycle and validation permit it.

### 11.5 Recurrent replay and stale states

A replay window's hidden state depends on the model version and history. Do not reuse a cached posterior from a different transition model without reconstruction.

Maintain sequence-start reset metadata and raw burn-in context. For offline training use 16-step burn-in then 32 scored compact steps. For online windows reconstruct a detached belief from a compatible snapshot plus up to 32 macro steps. If a snapshot is incompatible, replay from the episode start or explicitly initialize a broad prior and mask the burn-in loss.

Stored posteriors are acceleration hints, not immutable labels. Raw observations and issued commands are the authoritative evidence.


## 12. Persistent online optimization

### 12.1 Parameter and optimizer state

Maintain separate optimizer state for each regime and for visual versus compact corrections. Retain moments across task episodes within the same compatible regime. Clear or reinitialize moments after an incompatible coordinate/calibration change, a new adapter initialization, or a documented rollback policy; log the event.

For each trainable tensor \(\theta\), AdamW is

\[
g_n=\operatorname{clip}_{\|\cdot\|_2\le1}
(\nabla_\theta L_n),\quad
m_n=\beta_1m_{n-1}+(1-\beta_1)g_n,
\]
\[
v_n=\beta_2v_{n-1}+(1-\beta_2)g_n^2,
\]
\[
\theta_{n+1}=(1-\eta_n\lambda_{\rm wd})\theta_n
-\eta_n\frac{m_n/(1-\beta_1^n)}
{\sqrt{v_n/(1-\beta_2^n)}+\epsilon}. \tag{33}
\]

Clipping is global across the parameter group set for one optimizer, not independent per tensor. Decay excludes biases and normalization scalars. Reference anchoring is separate from weight decay toward zero.

Initial online values:

| Item | Default | Initial validation sweep |
|---|---:|---|
| Visual LoRA LR | \(10^{-5}\) | \(3\times10^{-6},10^{-5},3\times10^{-5}\) |
| Token residual LR | \(3\times10^{-5}\) | \(10^{-5},3\times10^{-5},10^{-4}\) |
| Compact final mean-correction LR | \(10^{-5}\) | \(3\times10^{-6},10^{-5},3\times10^{-5}\) |
| Betas | (0.9,0.99) | fixed initially |
| Epsilon | \(10^{-8}\) | fixed |
| Weight decay | \(10^{-4}\) | \(0,10^{-4},10^{-3}\) |
| Global gradient clip | 1.0 | fixed initially |
| Effective batch | 16 | 8,16,32 with matched update budget |
| Warmup after initialization | 100 updates | fixed |
| AR horizon | 2 | 2,5 |
| Updates | 1 per 16 new dense windows, at most 1/s | match compute in comparisons |

Only the builder model receives gradients. Serving weights remain immutable during a plan. Dropout is off online; normalization running statistics are fixed. “Evaluation mode” does not mean disabling autograd on trainable modules.

### 12.2 Training must not rewrite the evidence

For a transition ending at \(t+1\):

1. Predict using the version available before its target arrives.
2. Save the prediction and version for prequential scoring.
3. Receive and validate the target.
4. Assimilate it into the current belief.
5. Add the causal transition/window to replay.
6. Train a builder model later.
7. Consider promotion only on separate future data and anchors.

Do not report post-update loss on the training batch as adaptation generalization. Do not reassign all replay examples to whichever mode just fitted them best.

For online compact training, reconstruct the belief through time \(t\), then compute the prior prediction under candidate parameters. Detach the reconstructed prefix for the optimizer step. This defines a truncated online surrogate; full-sequence offline training remains the place for learning recurrence and the observation model.

### 12.3 Why the slow context and weights are both needed

The posterior over \(d_t\) handles changes explainable within trained dynamics rapidly, with few observations and no gradient update. Adapters learn systematic residuals outside that fast inference range. A single observation should normally update belief; only repeated valid evidence should update model parameters.

Do not let both adapt arbitrarily fast: that creates nonidentifiability where context and weights compensate for each other. Freeze context coordinates and observation decoder online, use small parameter updates, keep reference anchors, and evaluate on future transitions. A low prediction error alone does not tell which internal explanation is correct.

## 13. Candidate evaluation, promotion, rollback, and cache consistency

### 13.1 Three model roles

- **Serving:** immutable active weights used by filtering/planning.
- **Builder:** receives replay gradients.
- **Challenger:** frozen snapshot of the builder evaluated on subsequent data.

The base frozen transformer can share read-only weight storage; trainable adapter tensors, optimizer state, and caches must not alias. On one GPU, schedule shadow evaluation/training in idle gaps; if they delay perception, reduce their rates or use a separate worker.

Snapshot a challenger at most once every 30 seconds. Keep one challenger at a time. The builder can continue learning, but its newer weights cannot be silently substituted into the fixed challenger evaluation.

### 13.2 Promotion gate

After a challenger snapshot at cutoff \(t_0\):

1. Reconstruct its belief from a compatible earlier snapshot and raw history.
2. Collect 128 subsequent valid macro transitions not used to fit that challenger.
3. Compare serving and challenger predictions on exactly the same issued actions and observations, with version-specific recurrent states.
4. Group scores into temporal blocks of at least 2 seconds for diagnostics; do not count overlapping windows as independent.
5. Test 512 compatible anchor windows, sampled before the challenge.
6. Check finite parameters, covariance PSD, feature variance, action sensitivity, output drift, and latency.

Proposed acceptance conditions:

- mean future command-conditioned token loss improves by at least 2%, **and** a block-bootstrap 90% upper confidence bound on challenger-minus-serving loss is below zero;
- joint prediction error does not worsen by more than 2% for a visual-only update, or improves for a compact update;
- no compatible regime/critical task anchor group worsens by more than 5%;
- full-token foreground/task diagnostic does not worsen by more than 5%;
- compact uncertainty coverage is not materially worse;
- a fixed set of validated action probes does not show pathological flattening or reversal;
- all deadline and numerical checks pass.

These thresholds are initial engineering choices. With insufficient data or inconclusive intervals, retain the serving model. The gate is a heuristic risk reduction, not a statistical guarantee under arbitrary adaptive reuse. Keep an untouched final benchmark and report how often the online gate was queried.

A candidate can fail because there is no useful shift; unchanged-environment updates are not automatically beneficial. Adaptation is optional when the evidence does not support it.

### 13.3 Atomic promotion

Promote only at a planning boundary:

1. Finish/cancel the old plan.
2. Install candidate adapter and compact correction tensors as a new immutable version.
3. Restore a pre-window belief and replay the logged observations/actions to the current time with the new version.
4. Invalidate/rebuild all affected KV caches and imagined histories.
5. Publish the new belief/version atomically.
6. Resume planning only when the belief catches up to the current command log.

If reconstruction cannot finish before the next movement decision, hold and keep using the last validated local supervisor. Do not reset the slow belief blindly to zero just to make the version swap easy.

A KV cache depends on parameters and its exact input sequence. Updating Q/V/output adapters invalidates it even though the encoder is frozen. Also, because actions share frame-level attention with visual tokens, a cache for the current frame containing a speculative action cannot be reused for a different candidate action. Share only the genuinely common, action-independent prefix; verify cached versus uncached equality.

### 13.4 Rollback

Keep at least the last three validated versions plus the offline reference. Roll back on nonfinite predictions, unexplained large calibration regression, sustained prequential degradation above 10% over a 60-second valid-data window, or a diagnosed version-dependent control regression.

A rollback restores compatible parameters and optimizer lineage, then reconstructs belief and caches. It does not undo physical actions or erase their observations. New experience remains in replay with its original provenance.

## 14. Task objective and planning

### 14.1 Default goal-image objective

Use an actual goal image from the same camera contract. Encode it once with the frozen encoder. Start with external-camera goal weight 1 and wrist-camera goal weight 0; wrist input can still help dynamics. A moving wrist viewpoint can otherwise dominate the cost for the wrong reason.

For fixed masks \(w_{vj}\),

\[
D_g(z,z_g)=
\frac{\sum_{vjd}w_{vj}|\mathrm{LN}_0(z_{vjd})-\mathrm{LN}_0(z_{g,vjd})|}
{D_e\sum_{vj}w_{vj}}. \tag{34}
\]

If labeled foreground regions are available, use object/task patches weight 4 and background weight 1 as an initial ablation; freeze the mask per task and test sensitivity to the arm/background. A mask computed from a future generated image must not move arbitrarily to hide errors.

Normalize \(D_g\) by a fixed calibration scale, such as the 90th percentile of nontrivial training goal distances. The scale has a positive floor and is never updated to make the current plan appear successful.

An image goal does not fully specify a task with hidden constraints, object identity ambiguity, or a temporal requirement. Provide task metadata such as “retain object for one second” and an independently checked success predicate where required. Do not let the learned model define success solely by its own predicted score.

### 14.2 Optional offline task-score head

If the user needs goal-image-free fixed tasks, train a new head on RoboJEPA features with offline labels. Use spatially pooled features and proprioception, MLP widths 512,256,1, SiLU hidden layers, sigmoid output for bounded score. Train AdamW LR \(10^{-3}\), WD \(10^{-5}\), batch 256, max 100 epochs, patience 10; split by episode and balance high/low progress examples.

Use MSE for continuous progress and binary cross-entropy for labeled success, with calibration on separate episodes. Freeze the head online. This requires offline task labels; it is not reward-free task discovery. Goal-image and reward-head modes should be evaluated separately before mixing their objectives.

### 14.3 Action parameterization

Plan bounded incremental joint targets \(v_{t+k}\in[-1,1]^6\). Translate them to absolute command references:

\[
p_{k+1}^{\rm ref}=
\operatorname{clip}_{[p_{\min},p_{\max}]}
\left(p_k^{\rm ref}+v_k\odot\Delta p_{\max}\right). \tag{35}
\]

The first reference is the actual current local reference/position according to the command contract. A local rate/acceleration limiter interpolates at raw ticks. The simulator inside the planner applies the same limiter and records the resulting future command sequence. Otherwise the planner optimizes commands that will never be executed.

Arm joint/gripper increments have different physical scales. No generic “0.1” applies to all coordinates. \(\Delta p_{\max}\), velocity, acceleration, absolute limits, and gripper bounds come from measured safe commissioning values.

### 14.4 Belief-conditioned rollout

For candidate command sequence \(U\):

1. Sample a regime from \(\pi_t\) and a compact state from its Gaussian.
2. Roll \(F\) with sampled process noise and regime transitions.
3. Derive predicted realized joint motion.
4. Roll the full visual predictor using that motion, the same issued-command sequence, and the belief residual.
5. Update imagined recurrent history from generated summaries/joints.
6. Do not assimilate a generated image as a measurement.
7. Score the full trajectory.

Use common random numbers across candidate sequences within a CEM iteration to reduce score-comparison noise. Resample at the next replan. Preserve rare plausible regimes through stratified sampling if ordinary sampling would miss them.

### 14.5 Objective with bounded risk preference

For stochastic rollout particle \(m\),

\[
J^{(m)}(U)=
D_g(\hat z_{t+H}^{(m)},z_g)
+\lambda_u\frac1H\sum_{k=0}^{H-1}\|v_k\|_2^2/6
+\lambda_{\Delta u}\frac1H\sum_k\|v_k-v_{k-1}\|_2^2/6
+\lambda_{\rm lim}C_{\rm limit}^{(m)}. \tag{36}
\]

Use \(\lambda_u=0.01\), \(\lambda_{\Delta u}=0.05\). Hard geometric/joint violations reject a candidate; the soft limit term only discourages proximity to validated limits. Initially omit dense goal shaping so the terminal objective does not impose an unnecessary greedy preference. Add it only in a measured ablation.

The planning score is

\[
\mathcal J(U)=(1-\beta_{\rm risk})\mathbb E[J(U)]
+\beta_{\rm risk}\operatorname{CVaR}_{\alpha}(J(U))
+\lambda_{\rm epistemic}\,\sigma_{\rm model}(U), \tag{37}
\]

where \(\beta_{\rm risk}=0.25\), \(\alpha=0.8\), and

\[
\operatorname{CVaR}_{\alpha}(J)
=\min_\tau\left[\tau+\frac{\mathbb E(J-\tau)_+}{1-\alpha}\right].
\]

Use the weighted empirical form for nonuniform particles. With only four particles this tail estimate is crude; do not describe it as calibrated tail safety. Increase particles or use worst-of-sampled cost when rankings are unstable.

\(\sigma_{\rm model}\) is optional disagreement across three independently bootstrapped small residual/compact heads sharing the base encoder/predictor. It is not a Bayesian posterior over all model weights. Default \(\lambda_{\rm epistemic}=0.1\) after normalization; disable this term until independently trained heads exist. Identical copies produce no useful epistemic estimate.

Variance should discourage unsupported plans, but an excessive penalty can cause permanent inaction. Evaluate task progress and failure recovery as well as prediction error. Exploration bonuses are off on the physical arm initially.

### 14.6 CEM defaults and exact update

Initial horizon \(H=5\) macro steps (one second). Execute one macro block, then observe/replan. Use 128 candidates, four iterations, 16 elites; warm-start by shifting the last plan, initialize standard deviation 0.35 in normalized incremental-command coordinates, floor 0.05. Candidate 0 is no-op/hold, and include the previous feasible plan.

For elite sequences \(E\),

\[
\mu_{\rm new}=0.2\mu_{\rm old}+0.8\operatorname{mean}_{U\in E}U,
\]
\[
\sigma_{\rm new}=
\max(0.05,\ 0.5\sigma_{\rm old}
+0.5\operatorname{std}_{U\in E}U). \tag{38}
\]

Clip proposals to allowed normalized increments and simulate the actual local limiter. Use population standard deviation consistently. The final mean is not necessarily feasible under nonlinear constraints; re-evaluate it and the best sampled feasible sequence, execute the better valid candidate. If none is valid, hold.

Initially use one deterministic mean rollout per candidate for debugging, then four stratified belief particles per candidate. The deterministic version is an ablation, not uncertainty-aware planning. Larger profile: 256 candidates, five iterations, 32 elites, eight particles only after memory/latency measurement.

### 14.7 Why task performance must be tested separately

A smaller average feature loss can still worsen action ranking, ignore small objects, favor an arm pose over an object state, or exploit a reward head outside its training distribution. Therefore evaluate counterfactual action ordering on repeated physical setups or a validated simulator, foreground outcome prediction, and actual task success. Do not select the final system on latent loss alone.

## 15. Cloud architecture and timing contracts

### 15.1 Processes

**Local robot process:** device I/O, units, command queue, raw interpolation, hard limits, freshness checks, emergency stop, joint observer, event log, video capture.

**Cloud perception/filter worker:** frozen encoder, compact sketch, recurrent filter, timestamp reconciliation, uncertainty publication.

**Cloud planner worker:** immutable serving model, CEM, candidate rollouts, explicit start/expiry metadata.

**Cloud learning worker:** replay, builder updates, challenger evaluation, checkpoints. It has no direct motor-command privilege.

These may share one machine initially, but they have separate queues and priorities. A single GPU can serialize them; separation into processes does not guarantee concurrency.

### 15.2 Command packet

Each plan message includes robot/calibration ID, plan/model version, observation timestamp, incorporated action sequence number, local start condition, planned reference trajectory, issue time, expiry time, expected initial joint range, and limits version.

The local process rejects duplicates, reordered commands, mismatched calibration, impossible initial conditions, expired commands, or commands based on unacknowledged action history. Commands are not replayed automatically after reconnect.

### 15.3 Two deployment profiles

**Profile A: stop–observe–plan–execute.** This is the required first robot implementation. The arm holds a locally validated stationary configuration while cloud planning runs. Before executing one short block, obtain a fresh observation and check scene/joint consistency with the planning anchor. Re-score the chosen trajectory from the fresh belief if necessary. If the scene changed materially, reject and replan.

Fresh revalidation must itself satisfy a configured age budget; a seconds-old plan is not made fresh merely by resetting its timestamp. This profile assumes a sufficiently static environment during deliberation. It is inappropriate for catching moving objects or continuous fast contact.

**Profile B: continuous asynchronous operation.** Enable only after measured end-to-end performance meets a latency gate. The planner rolls forward through the already committed command prefix to the planned start time and optimizes only the uncommitted suffix. The local controller may execute only the bounded, still-valid committed prefix. If no fresh replacement arrives, it transitions to a prevalidated hold/stop behavior.

The belief at wall time is prediction from the last actual observation, not a claim of fresh vision. Limit the predicted-only gap and visualize/report observation ages.

### 15.4 Proposed timing targets

| Component | Initial target | Requirement |
|---|---:|---|
| Local position I/O and supervisor | 30 Hz | Must be measured; lower if bus cannot sustain it |
| Camera capture | 30 Hz | Save capture timestamps; dropped frames are explicit |
| Macro visual filtering | 5 Hz | Priority over planning/training |
| Continuous-profile perception-to-action p99 | <300 ms | Gate, not an expected measured result |
| Maximum visual age for new continuous motion | 500 ms | Tune downward for fast tasks |
| Command expiry after final validation | 300 ms | Local timestamp/clock-bound enforcement |
| Fixed-lag reconciliation | 2 s | Longer gaps reinitialize fast state |
| Builder duty cycle | ≤10% during operation | Train in holds or separate GPU if violated |

Clock offsets must have a measured uncertainty. If uncertainty alone consumes the expiry budget, disable continuous operation. Use local monotonic time and estimated clock-offset intervals; do not compare unsynchronized wall-clock timestamps naively.

A timeout does not automatically mean disable torque: doing so can drop an arm or object. Commission a task-specific stop/hold behavior locally. It may hold position, stop incremental target changes, or request operator intervention. Neither paper supplies that hardware behavior.

### 15.5 Scheduling and backpressure

Prioritize local safety/I/O, then perception/filtering, then current planning, then learning. Planning evaluates microbatches and yields between them. Do not claim GPU preemption unless the actual runtime supports it. A single long kernel can still delay perception.

Cap queues. Drop obsolete unprocessed planning jobs, not authoritative action events. For vision, process a recent keyframe for live state and archive skipped frames for later learning. Preserve command order. Training backlog is bounded by update credits.

If a single cloud GPU cannot maintain filtering while CEM runs, use a second worker/GPU or Profile A. Increasing an action chunk to hide latency is not automatically acceptable; it increases open-loop risk.

### 15.6 Optional fast planner after the full-model baseline works

A future speed path uses the compact model for most candidates and full RoboJEPA only to rerank a small set. Train a goal-conditioned cost surrogate

\[
\hat c_\varphi(y,y_g,\bar p)=
\operatorname{softplus}(\operatorname{MLP}_{256,256,256}([y,y_g,y-y_g,\bar p])).
\]

Fit it to full-model token distances on real observed frame/goal pairs and teacher rollouts, with equal weight on the two sources. Use Huber loss and a pairwise ranking term

\[
L_{\rm rank}=\log(1+\exp[-s_{ab}(\hat c_b-\hat c_a)]),
\quad s_{ab}=\operatorname{sign}(c_b-c_a),
\]

for pairs with nontrivial teacher margins; ties are excluded. AdamW LR \(3\times10^{-4}\), WD \(10^{-4}\), batch 256, 20,000 maximum updates, validation every 500.

First search 512 compact candidates, rerank the best eight plus hold with the full model, and execute only after the same freshness/feasibility checks. A surrogate trained on teacher rollouts inherits teacher errors; real-outcome ranking and timing tests remain necessary. This extension is not a claim that compression preserves every task-relevant distinction.

## 16. Compute and storage budget

No exact throughput or GPU bill can be inferred from parameter count alone. Benchmark the actual checkpoint, precision, compiler, views, batch size, cache occupancy, and network.

A useful initial cloud experiment is one high-memory GPU with at least roughly 48 GB for the 300M predictor plus frozen encoder and small adapters; an 80 GB class device gives more room. These are capacity planning suggestions, not verified fit guarantees. The frozen encoder can be much larger than the small predictor. Do not budget as if “300M” described the entire system.

For larger predictors, use separate inference/learning workers when possible. Avoid starting with full-parameter 8B online AdamW: parameter/gradient/master-weight/moment storage plus activations and caches dominates the memory budget. Parameter-efficient updates still require activation storage through affected layers.

For predictor depth \(L\), KV heads \(h_{kv}\), head dimension \(d_h\), token count \(N_{\rm tok}\), retained frames \(W\), batch \(B\), and \(b\) bytes per value,

\[
M_{\rm KV}\approx
2L B h_{kv} W N_{\rm tok}d_h b. \tag{39}
\]

At two views \(N_{\rm tok}=602\). With the inspected 300M architecture, \(L=24,h_{kv}=8,d_h=128,W=8,b=2\), this is about 0.47 GB **per independent candidate history**, before other tensors. Four particles and many candidates require microbatching and shared common prefixes. Verify with a profiler; functional caches can add temporary copies.

Predictor-call work is approximately

\[
N_{\rm CEM}\times I\times H\times M,
\]

plus prefix encoding. Defaults 128×4×5×4 imply 10,240 candidate-particle one-step evaluations per replan. Batching improves throughput but does not remove this workload. This is why a cloud GPU does not by itself establish 5 Hz full CEM.

Use BF16 serving initially. Quantization, static compilation, and weight hot-swapping need numerical equivalence tests. Compiled graphs must read current adapter tensors; an artifact with constant-folded old weights cannot serve a newly promoted adapter correctly. Avoid recompilation on every tiny update by keeping adapter operations explicit or by promoting compiled snapshots less frequently.

Track wall time, p50/p95/p99, peak memory, kernel occupancy, inference queue age, bytes transferred, and cost per successful episode. Evaluate dollars/episode and recovery time, not only FLOPs.


## 17. Mathematical checks, guarantees under assumptions, and non-guarantees

### 17.1 Belief preservation and mixture uncertainty

If each covariance is positive semidefinite, (15) and (20) are positive semidefinite because they sum PSD within-mode covariances and outer products with nonnegative weights. Their second terms are the uncertainty due to disagreement. A mixture of “object left” and “object right” must not be compressed into a certain “object center.”

With \(Q\succeq0\), prediction (16) preserves PSD. With \(R\succeq0\), Joseph update (18) preserves PSD in exact arithmetic. Floating-point error still requires symmetry checks, Cholesky diagnostics, and carefully bounded jitter. Large recurrent repairs indicate a model/numerical bug, not normal operation.

### 17.2 When posterior uncertainty is meaningful

Within a correct linear-Gaussian single-mode model, the Kalman recursion is exact. Here transition nonlinearities are linearized, modes are moment matched, images are compressed, and likelihoods are learned. Therefore neither the 95% ellipsoid nor mode probabilities are automatically calibrated for the physical world.

Use empirical coverage and proper predictive scoring. Report failures under held-out regimes. A deterministic GRU vector with an arbitrary “confidence head” would not fix these issues; the explicit likelihood/covariance structure makes them testable.

### 17.3 Identifiability of actuator gain: an elementary example

Suppose one joint approximately follows

\[
q_{n+1}-q_n=a(u_n-q_n)+\varepsilon_n,\qquad
\varepsilon_n\sim\mathcal N(0,\sigma^2).
\]

The Fisher information for \(a\), conditional on the chosen inputs, is

\[
\mathcal I(a)=\frac1{\sigma^2}\sum_n(u_n-q_n)^2. \tag{40}
\]

If every command equals the current joint position, there is no information about the gain. A high-confidence gain estimate from stationary data would be unjustified. Likewise, direction-reversal data are needed to distinguish backlash from simple gain, and varied command timing is needed to infer delay.

Safe excitation can be a small, prevalidated calibration motion. It cannot guarantee identification of properties whose effects are absent from the sensors or admissible tasks.

### 17.4 Prediction error and action-ranking error

Let true and modeled state transitions be \(f,\hat f\). Assume on a particular bounded region that \(f\) is \(L_f\)-Lipschitz in state and one-step model error is at most \(\epsilon\). Starting from state error \(e_0\),

\[
e_k\le L_f^k e_0+\epsilon\sum_{j=0}^{k-1}L_f^j. \tag{41}
\]

If the terminal cost is \(L_c\)-Lipschitz there, then

\[
|\hat J(U)-J(U)|\le L_c e_H
\]

for terminal-only cost, plus corresponding accumulated stage-cost bounds when present. If every candidate has cost error at most \(\varepsilon_J\), an exact modeled minimizer has true cost at most \(2\varepsilon_J\) above the best candidate's true cost. An approximate optimizer adds its modeled suboptimality gap.

This is a conditional statement. The required constants and uniform error bound are generally unknown for a neural latent model. Average test L1 error is not such a bound, and a latent metric need not be Lipschitz-equivalent to task success. Use the result to explain why horizon, uncertainty, and ranking matter, not to claim a proven robot guarantee.

### 17.5 Why a self-supervised gradient can harm control

Let \(L_{\rm pred}\) be prediction loss and \(J_{\rm task}\) task cost. A small gradient update gives

\[
\Delta J_{\rm task}\approx
-\eta\langle\nabla_\theta J_{\rm task},
\nabla_\theta L_{\rm pred}\rangle. \tag{42}
\]

If the gradients are opposed, reducing prediction loss worsens task cost. Offline loss may be dominated by background features; on-policy adaptation may improve only the states the current bad planner visits. Frozen goals, foreground diagnostics, replay diversity, and action-ranking tests reduce this risk but do not eliminate it.

### 17.6 Conditional tracking stability

Suppose a local estimator's error dynamics satisfy

\[
\|e_{t+1}\|\le\kappa\|e_t\|+\epsilon_t+
L_\Theta\|\Theta_{t+1}-\Theta_t\|,
\qquad\kappa<1.
\]

Then bounded residuals and parameter steps yield a bounded geometric error sum. This motivates small validated parameter changes and belief reconstruction. It does **not** prove \(\kappa<1\) for the proposed neural filter; contact and occlusion can violate contraction.

### 17.7 Cost of stale observations

For a quantity whose unobserved change rate is bounded by \(v_{\max}\), observation age \(a\) alone permits error up to \(v_{\max}a\), before model error. If acceleration is also bounded, a second-order term can be added. Such bounds require actual assumptions about object/arm motion; a human moving an object may invalidate them.

A fast network cannot recover information from an unseen event. An old posterior propagated forward is not equivalent to a new measurement.

### 17.8 Two timescales are useful, not automatically convergent

Classical stochastic-approximation convergence conditions do not directly apply to a finite-rate adaptive robot with nonstationary data, neural models, a changing policy, and fixed learning rates. The design uses separated state/context/parameter updates as an engineering principle. It makes no theorem of global convergence, optimal control, universal observability, or lifelong non-forgetting.

## 18. Edge cases and required responses

| Case | Failure mechanism | Required behavior/test |
|---|---|---|
| Frozen encoder misses small object | Low global loss, bad manipulation | Object-versus-background probe; improve viewpoint/resolution or observation features |
| Same image, different hidden grasp state | Aliasing | Preserve multiple hypotheses; observe from another useful view or use a bounded verification motion |
| Joint target equals current state | No excitation | Do not infer actuator gain with high confidence |
| Motor clipping | Requested action differs from issued | Train/plan on logged issued targets; simulate limiter |
| Follower lag/backlash | Command differs from motion | Explicit queue and command model; reversal/timing validation |
| Gripper zero/one inversion | Opposite behavior | Device-level direction test and transform fixture |
| SO-100 versus SO-101 source mixing | Geometry/calibration mismatch | Validate joint conventions; avoid untrusted FK labels |
| Wrist rotation treated as XYZ/Euler layout | State corruption | Joint propagation tests and explicit action sampler |
| Occlusion | Missing scene evidence | Predict with uncertainty; no self-generated measurement update |
| Camera moved or zoom/crop changed | Representation shift | Invalidate goal/observation contract; do not adapt dynamics blindly |
| View missing | Invalid token/input likelihood | Supported single-view path or trained mask handling |
| Frame duplicated/reordered | Double evidence or wrong action pairing | Timestamp/sequence deduplication; fixed-lag replay |
| Video timestamp is receive time | Apparent dynamics lag | Preserve capture time; measure clock error |
| Cloud returns an obsolete plan | Wrong initial state | Local expiry and initial-condition check |
| Network loss during chunk | Unbounded open-loop execution | Execute only approved bounded prefix, then commissioned stop/hold |
| Torque disabled while carrying object | Drop | Task-specific local stop policy, not generic disconnect behavior |
| Object moved by human during planning | Static-scene assumption broken | Fresh revalidation and replan |
| Task reset mistaken for dynamics shift | Learns discontinuity as action effect | Explicit reset boundary; no cross-reset windows |
| A→B→A regimes | Uniform replay averages incompatible dynamics | Mode/context inference, partitioned replay, retention evaluation |
| Continuous drift | Static mode bank insufficient | Slow context process noise plus small updates; reassess bank design |
| New regime beyond all modes | False certainty | Quarantine, broaden, collect data, shadow candidate |
| No free mode slot | Silent forgetting | No automatic eviction in version 1; offline consolidation |
| Likelihood covariance inflates | Hides modeling errors | Freeze noise online; inspect raw innovation and coverage |
| Likelihood covariance collapses | Overconfidence | Offline bounds/calibration, held-out NLL and coverage |
| LoRA changes while cache lives | Stale attention activations | Immutable plan versions and cache rebuild |
| Current-frame cache contains speculative action | Candidate contamination | Share only genuine common prefix; uncached equality test |
| Replayed hidden state from old model | Invalid training targets/context | Burn-in/reconstruction by version |
| Positive loss improvement, worse action ranking | Planner exploitation | Held-out counterfactual ranking and robot success gates |
| Reward head out of distribution | Invented high reward | Frozen calibrated head, uncertainty/anchor checks, independent task success |
| Correlated dense windows | Inflated sample count/confidence | Episode splits and temporal-block uncertainty estimates |
| Growing replay/features | Unbounded memory/cost | Byte budgets, references, reservoirs, bounded feature cache |
| Rare mode posterior disappears | Resampling/underflow | Log-space weights, stratified planning, broad unknown prior |
| Near joint limit | Invalid Gaussian tail actions | Hard command constraints and local rejection |
| Numerical covariance failure | Invalid belief | Stop promotion, retain valid version, diagnose; never silently zero covariance |
| Symmetric objects/goal ambiguity | Wrong identity/task completion | Task identity metadata or additional observations |
| Insufficient camera-only depth | Geometric ambiguity | Restricted tasks/views; no invented 3D certainty |
| “Contact” inferred from tracking error | Sensor ambiguity | Label as hypothesis; do not report measured contact force |
| Continuous control misses latency gate | Unsafe stale predictions | Remain in Profile A or improve actual architecture/hardware |

This list is extensive but not exhaustive. No finite specification enumerates every physical failure. Log unclassified failures and add them as regression cases.

## 19. Evaluation protocol and acceptance gates

### 19.1 Baselines that isolate the contribution

Use identical starting checkpoints, goal interface, robot limits, observation history, and planning budgets where possible.

| ID | System |
|---|---|
| B0 | Frozen RoboJEPA with corrected joint rollout and nominal command bridge |
| B1 | B0 plus persistent JEPA-TTT-style predictor learning, no extra recurrent belief |
| B2 | B0 plus recurrent belief, no parameter adaptation |
| B3 | B0 plus command-model adaptation only |
| B4 | Belief + persistent adaptation, one regime |
| B5 | Full proposal with calibrated regime bank |
| B6 | B5 but episode-reset optimizer/parameters, matched update count |
| B7 | B5 with sparse windows, matched gradient/data budgets separately |
| B8 | B5 with uniform all-history replay instead of regime-aware replay |
| B9 | B5 with encoder adaptation, offline sandbox study only |
| B10 | B5 without uncertainty term or with mean-only planning |

B0 must fix interface errors; comparing a correct proposed method against a known-broken joint integration baseline would exaggerate the contribution. Report architecture/parameter/compute differences.

### 19.2 Test families

1. Nominal no-shift operation.
2. Software-imposed command gain changes and bounded extra delay.
3. Direction-dependent response/backlash-like changes, first in simulation or bounded calibration.
4. Safe object variation that changes observed contact response.
5. Partial occlusion and camera dropout.
6. Camera/exposure/crop shifts to verify detection, not claim dynamics-only adaptation solves them.
7. A→B→A and A→B→C→A recurrence.
8. Gradual drift plus abrupt switch.
9. Network latency/jitter/dropout/reordering and process restarts.
10. New object/layout combinations and long-horizon tasks.
11. Failure/recovery states underrepresented in demonstrations.
12. Wrong or unreachable goals, with graceful failure instead of endless unsafe motion.

Do not introduce hazardous physical modifications merely to create a benchmark. Software command perturbations and simulator tests can establish many mechanisms first.

### 19.3 Metrics

**Prediction:** command-conditioned joint error in radians/gripper units; realized-motion versus command-conditioned token loss; 1/2/5/10-step prior rollout error; foreground error; action-response sensitivity; compact/full disagreement.

**Belief:** predictive NLL; innovation statistics; empirical interval/ellipsoid coverage where ground truth exists; mode-recognition delay; posterior entropy; calibration under occlusion; false-alarm rate. Latent-state coverage itself is not measurable without identified ground truth; use observed-outcome predictive coverage instead.

**Control:** independently judged task success, completion time, recovery rate, object retention, path length, reversals, hard-limit rejection, operator interventions, drop incidents, and no-op fraction.

**Adaptation:** recovery versus actual wall-clock minutes and valid transitions; area under online success/progress curves; prequential loss; no-shift degradation; old-regime retention; recurrence recovery; accepted/rejected updates.

**Systems:** p50/p95/p99 observation age and action latency, deadline misses, queue depth, stale-plan rejection, peak GPU/CPU memory, bytes/hour, and cost/success.

Do not rely on best checkpoint over many evaluations as the primary claim. Predeclare final-time performance and adaptation AUC; best-ever score is secondary and selection-sensitive.

### 19.4 Statistical design

Use at least three independently trained adapter/filter seeds in development and three independent deployment streams per seed when feasible. For physical success estimates, start with 30 episodes per main condition and expand based on confidence intervals and effect size; this is an initial budget, not a power guarantee.

Pair initial conditions/objects and randomized method order where reproducible. Reinitialize physical conditions independently; learning-state persistence follows the declared method. For recurrence tests, preserve learning state intentionally and report order effects.

Use episode-level bootstrap for task metrics and temporal-block bootstrap for continuous streams. Report the actual number of independent sessions, not just dense-window count. Evaluation trajectories never enter adaptation unless explicitly labeled an online-learning condition. Maintain a separate untouched final evaluation set.

### 19.5 Implementation acceptance gates

**G0, interface:** units/order/legacy transforms pass; issued commands and measured positions are distinguishable; joint propagation is correct; clocks and reset boundaries are verified.

**G1, model baseline:** full command-conditioned prediction beats stationary-image and nominal-servo baselines on relevant held-out motions, with meaningful action sensitivity. No future-state leakage.

**G2, belief:** compact predictive NLL/coverage improve over a deterministic history baseline on occlusion/delay cases. Missing observations do not produce fake confidence. Regime uncertainty survives ambiguity.

**G3, adaptation:** on held-out dynamics shifts, B1/B4 improve future command prediction; nominal-condition and anchor regressions stay within preregistered tolerances.

**G4, control:** the combined method improves actual task outcomes or recovery over corrected B0 under comparable compute, not just token loss.

**G5, persistence:** A→B→A recovery and retention improve over an unpartitioned/reset baseline, with storage bounded.

**G6, cloud:** stale/out-of-order/expired plans are rejected locally; disconnected operation follows the commissioned stop policy; actual timing supports the selected profile.

Only then enable broader autonomous collection or continuous Profile B.

## 20. Reference execution pseudocode

The pseudocode defines ordering and ownership, not a complete neural implementation.

### 20.1 Local device loop

```python
while connected:
    now = local_monotonic_time()
    p, p_valid = read_joint_positions_with_timestamp()
    append_joint_event(now, p, p_valid)
    capture_and_queue_camera_frames()  # capture times, not only receipt times

    plan = newest_valid_plan()
    if not p_valid:
        run_commissioned_stop_policy()
        continue

    if not accepts(plan, now, calibration_hash, action_log_sequence, p):
        run_commissioned_hold_policy()
        continue

    requested = interpolate_rate_limited_reference(plan, now)
    issued = send_clamped_target(requested)
    append_action_event(now, requested, issued, plan.version)
    # Never overwrite issued with the next measured joint position.
```

### 20.2 Cloud filtering loop

```python
for event_batch in chronological_macro_events():
    active = serving_version_snapshot()
    if event_batch.is_late:
        restore_snapshot_before(event_batch.capture_time)
        replay_each_event_once_with(active)
        continue

    U = actual_issued_commands_since_last_macro_observation()
    z, masks = encode_valid_images_only(event_batch, frozen_encoder)
    y = make_fixed_sketch_and_joint_observation(z, event_batch.joints)

    prior = imm_predict(belief, recurrent_h, U, active)
    log_prequential_predictions(prior, active.version)

    belief = masked_imm_update(prior, y, masks, active)
    recurrent_h = update_history_after_assimilation(
        recurrent_h, y, prior, masks, U
    )
    publish_belief_with_ages_and_action_sequence(belief, recurrent_h)

    append_raw_and_feature_references(event_batch, U)
    admit_only_complete_causal_windows()
    update_change_monitor_using_saved_preupdate_residuals()
```

### 20.3 Learning and promotion loop

```python
if credits > 0 and enough_valid_data() and compute_budget_available():
    batch = draw_regime_compatible_replay_batch()
    prefix = reconstruct_beliefs_with_burnin(batch, builder.version)
    loss_v = command_tf_ar_anchor_loss(batch, stopgrad(prefix))
    loss_c = compact_mean_loss(batch, stopgrad(prefix))
    update_selected_builder_parameters(loss_v, loss_c)
    assert_finite_and_log_gradient_norms()
    credits -= 1

if no_challenger() and snapshot_interval_elapsed():
    challenger = immutable_snapshot(builder)
    begin_future_only_evaluation(challenger)

if challenger_has_enough_future_data():
    decision = compare_future_predictions_and_anchors(challenger, serving)
    if decision.accept and at_planning_boundary():
        reconstruct_candidate_belief_from_event_log()
        invalidate_parameter_dependent_caches()
        atomic_promote_model_and_belief()
    else:
        retain_serving_version()
```

### 20.4 Planning loop

```python
active, belief, h, context = atomic_serving_snapshot()
check_observation_age_and_validity()
roll_through_already_committed_commands_to_plan_start()

candidates = initialize_or_shift_cem_distribution()
for iteration in range(cem_iterations):
    U = sample_joint_reference_sequences(candidates)
    U = simulate_the_same_local_limiters(U)
    costs = rollout_belief_particles_with_immutable_model(
        U, belief, h, context, active,
        assimilate_imagined_observations=False
    )
    candidates = refit_from_feasible_elites(costs)

chosen = reevaluate_mean_and_best_feasible_sample(candidates)
if fresh_state_revalidation_passes(chosen):
    send_one_bounded_block_with_version_start_conditions_and_expiry(chosen)
else:
    request_reobservation_or_hold()
```

A change of active version during a plan does not mutate that plan's snapshot. The resulting old-version proposal must still pass local version/freshness policy, or it is discarded.

## 21. Implementation modules and invariant tests

Suggested repository structure:

```text
robot_io/
  so101_units.py             # device ↔ physical ↔ compact ↔ legacy units
  command_limiter.py         # identical planner/local reference semantics
  event_log.py               # append-only timestamps and sequence numbers
  supervisor.py              # local expiry, limits, hold/stop
models/
  frozen_encoder.py
  legacy_robojepa_wrapper.py # joint integration and action-layout fix
  command_encoder.py
  compact_dynamics.py
  observation_model.py
  recurrent_history.py
  belief_residual.py
  lora_registry.py
belief/
  imm_ekf.py
  calibration.py
  fixed_lag_replay.py
learning/
  windows.py
  replay.py
  losses.py
  offline_train.py
  online_builder.py
  promotion.py
planning/
  joint_cem.py
  rollout.py
  goal_cost.py
  timing_contract.py
evaluation/
  prediction.py
  recurrence.py
  task_success.py
  latency_fault_injection.py
```

Required high-value tests:

- Exact unit round trips, gripper direction, zero padding, six-joint additive integration.
- Command-to-motion test: delayed target produces delayed movement; future observed joints cannot enter command rollout.
- Causal masking: perturb future frames and confirm earlier predictions unchanged.
- Cached/uncached parity for different actions, mode adapters, and cache wrap.
- LoRA zero-init parity and proof that promotion changes outputs rather than serving stale compiled weights.
- Kalman closed-form checks, mixture moment matching, missing-observation behavior, PSD covariance under random cases.
- Late-event replay equals a clean chronological run within numerical tolerance.
- No double-assimilation of the same observation ID.
- Every replay window remains inside its physical episode/calibration contract.
- No overlap between final evaluation targets and training data.
- Recurrent burn-in uses only preceding evidence.
- Optimizer moments persist across ordinary task episodes and remain regime-specific.
- Parameter promotion rebuilds belief/cache, and rollback preserves the actual action log.
- Disconnect, expiry, duplicate, reorder, and wrong-start-condition rejection in a hardware-free harness before the arm moves.

The companion `verify_math.py` checks selected mathematical invariants without neural dependencies. Passing it does not validate the learned model, robot I/O, network implementation, or physical safety.

## 22. Recommended build order

1. Correct SO-101 data/action units, timestamps, joint propagation, and issued-command logging.
2. Run frozen RoboJEPA with the nominal command bridge and bounded Profile A operation.
3. Collect the actual robot's command/observation data and establish useful task sensitivity.
4. Train the compact recurrent model and calibrate predictive uncertainty.
5. Train visual adapters/residuals using command-conditioned and autonomous rollout losses.
6. Add persistent replay-based online updates in shadow, with serving/challenger separation.
7. Demonstrate improvement over B0/B1/B2 and no unacceptable no-shift regression.
8. Add multiple recurring regimes only when the one-regime system and dataset justify them.
9. Optimize inference or introduce the compact surrogate only after full-model correctness.
10. Enable continuous cloud control only after real timing/fault tests pass.

This ordering does not discard the full architecture. It makes each claimed benefit falsifiable and prevents compensating errors from being hidden inside a large system.

## 23. What is settled by this specification and what remains empirical

Specified: variable meanings and units; tensor shapes; probabilistic filtering equations; command versus movement semantics; recurrence ordering; objectives and gradient stops; training stages; optimizer defaults; causal replay; parameter lifetimes; promotion and cache protocol; planning objective; cloud timing contracts; persistence/reset rules; evaluation baselines; and failure responses.

Empirical: whether the encoder sees the user's task-relevant distinctions; how much data the actual arm needs; which checkpoint size is best; actual GPU memory/latency; observation noise and servo parameters; mode separability; calibration under novel contact; gains from adaptation; and final task success.

The strongest defensible claim is that this is a buildable, testable proposal for applying persistent JEPA-style test-time learning to RoboJEPA while maintaining a recurrent uncertainty-aware state. Neither the papers nor these equations establish that the complete proposed SO-101 system already works. The experiments above are the path to establishing that.




