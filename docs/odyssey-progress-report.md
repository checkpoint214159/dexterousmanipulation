# Learning Dexterous Manipulation with Reinforcement Learning in Isaac Gym

**Odyssey project — CP2107 · Project P034301**

*Progress report and forward plan.*

---

## 0. Project overview

The goal of this project is to train an anthropomorphic robot hand (the 16-DoF
[LEAP Hand](https://arxiv.org/abs/2309.06440)) to perform contact-rich
in-hand manipulation — primarily in-hand object reorientation — entirely in
simulation, using massively-parallel reinforcement learning in NVIDIA Isaac
Gym (Preview 4). The training stack is a fork of *LEAP Hand Sim* layered on
`rl_games`, driven by PPO and a set of alternative RL algorithms benchmarked
below, with all training run headless on a WSL2 + Docker development setup.

The work to date falls into three areas with a concrete experimental
foundation — (1) building visualization and instrumentation that actually
works on a headless WSL2 machine, (2) a controlled benchmark of four RL
algorithms on the same task, and (3) extending the pipeline to harder,
non-spherical objects and tuning exploration to avoid policy collapse. The
remainder of the report is a forward plan: migrating to Isaac Lab, closing the
sim-to-real gap with a physical hand, and exploring diffusion policies as an
alternative action representation.

---

## 1. Visualization and instrumentation with Rerun

### 1.1 The problem: headless WSL2 has no render path

Training runs for dexterous manipulation are long and headless, and the only
intuitive way to know whether a policy is actually *grasping* — whether the
hand closes on the object, whether the cube is being rotated rather than
dropped, whether commanded joint angles are being tracked — is to watch the
simulation render. On this development setup that is impossible:
IsaacGym's camera sensors require Vulkan, and WSL2 ships no NVIDIA Vulkan ICD
(`libGLX_nvidia.so.0` does not exist on the Linux side because Windows WDDM
owns the GPU for display). Every available Vulkan path in the container
(LLVMpipe, Mesa software rasterizers) crashes IsaacGym's renderer with a
SIGSEGV inside `libvulkan_lvp.so`. Worse, simply enabling `record_video` forces
`enableCameraSensors=True`, which initialises Vulkan and crashes the process
*even in headless training*. TensorBoard alone gives scalars but no spatial
view, so it cannot answer the one question that matters most early on: "is the
hand even holding the object?"

This was resolved by adopting **[Rerun.io](https://rerun.io)** as the
visualization backend (documented as
[ADR-0001](adr/0001-rerun-as-visualization.md)). Rerun's key property is that
its SDK and viewer are fully decoupled: the SDK runs *in-process* inside the
training loop, logs typed spatial-temporal primitives (`Transform3D`,
`Boxes3D`, `Scalar`, `TextLog`) tagged with a timeline value, and serialises
them to an `.rrd` file on the mounted workspace filesystem. The viewer runs
natively on Windows against the host GPU and simply reads that file. No pixel
buffer ever crosses the WSL2 boundary, and Vulkan is bypassed entirely — Rerun
reconstructs the scene from the rigid-body state tensors that IsaacGym already
maintains, requiring zero changes to IsaacGym, `rl_games`, or the PhysX
pipeline.

### 1.2 What gets logged

Recording is done as **periodic windowed snapshots** rather than continuously,
to keep the file count and total size manageable. Every `record_every_n_steps`
control steps, a single environment (env 0 by default) is recorded for
`window_length_steps` steps, producing one `.rrd` per window under
`runs/<run_name>/rerun/window_<N>_step_<M>.rrd`. Two windows from different
checkpoints can be opened side-by-side in the viewer to compare a policy
against its earlier self. Each window captures:

- **Spatial scene** — world-frame `Transform3D` for all 17 rigid bodies of the
  hand (URDF link names used verbatim as entity paths) plus the object pose,
  every control step. The IsaacGym `(x, y, z, w)` quaternion convention matches
  Rerun's `rr.Quaternion(xyzw=...)` exactly, so no reshuffle is needed — a
  subtle correctness point flagged explicitly in the ADR.
- **Control overlays** — commanded vs. achieved joint angle per DOF (16 + 16
  scalars), making it visible at a glance when the PD controller is failing to
  track targets.
- **Object dynamics** — linear and angular velocity magnitudes, the latter
  being the proxy most directly tied to the rotation reward.
- **Episode events** — a `TextLog` entry on every env reset, so the timeline is
  scrubbable and resets (i.e. drops) are easy to spot.

> **[ Insert Rerun viewer screenshot here — spatial scene: hand + object, one recording window ]**
>
> **[ Insert Rerun viewer screenshot here — commanded vs. achieved joint-angle scalar panels ]**

### 1.3 Scalars and cross-algorithm comparison

In v1, training scalars (rewards, value/policy loss, entropy, KL, learning
rate) continue to be written to TensorBoard via `rl_games`' existing
`SummaryWriter`, while Rerun owns the spatial/geometric view; unifying both
behind Rerun's `Scalar` primitive is a deferred v2 work item that requires
subclassing `CommonAgent.write_stats()`. In practice this two-backend split is
exactly what made the algorithm benchmark in §2 possible: every run emits a
consistent set of scalar tags (`rewards/iter`, `rewards/step`, `rewards/time`,
`losses/entropy`, `episode_lengths/step`, throughput counters, plus
algorithm-specific tags such as SAPG's `sapg/off_policy_frac`), so runs from
different algorithms can be parsed and compared on identical axes — by
iteration, by environment step, and by wall-clock time.

> **[ Insert Rerun/TensorBoard scalar plot here — reward vs. iteration, all four algorithms overlaid ]**

---

## 2. Algorithm benchmarking: PPO vs. SAPG vs. SAC vs. TD3

A core piece of the work was a **controlled comparison** of four RL algorithms
on the same in-hand rotation task. This was treated as a proper benchmark
rather than a collection of ad-hoc runs: all four were run on the identical
`LeapHandRot` task (rotate a ball about a target axis), with **512 parallel
environments**, **5000 iterations**, **~82 M total environment steps**, and
matched batch sizes (minibatch / replay batch = 8192, horizon = 32). Holding
the environment-step budget and batch size fixed is what makes the reward
numbers comparable across an on-policy method, an off-policy-augmented
on-policy method, and two pure off-policy methods.

### 2.1 Results

All numbers below are read directly from the recorded TensorBoard scalars of
the four benchmark runs. The reward is the shaped rotation reward (angular
velocity about the target axis, minus pose / work / velocity penalties), so its
absolute scale is task-shaping-dependent; **episode length** (max 400 steps ≈ a
full 20 s episode) is the cleaner "did it keep hold of the object" proxy.

| Algorithm | Type | Peak reward | Final reward | Final episode len (/400) | Wall-clock | Throughput |
|---|---|---:|---:|---:|---:|---:|
| **PPO**  | on-policy | **66.4** | **50.1** | **393** | 2.6 h | ~10.2k steps/s |
| **SAPG** | on-policy + off-policy chunks | 47.3 | 32.1 | 356 | 2.6 h | ~10.5k steps/s |
| **SAC**  | off-policy, max-entropy | 33.5 | 26.7 | 394 | 2.9 h | ~9.8k steps/s |
| **TD3**  | off-policy, deterministic | −2.7 | −4.6 | 297 | 2.7 h | ~9.4k steps/s |

Wall-clock and throughput are roughly equal across all four (~2.6–2.9 h,
9.4–10.5k env-steps/s) — at 512 environments none of these methods is
GPU-throughput-bound, so the differences below are about *learning*, not
*speed*.

### 2.2 Per-algorithm analysis

**PPO (the clear winner at this scale).** PPO learns fastest and finishes
highest. It is already above reward ~55 by 20 % of training and peaks at 66.4,
with final episode length essentially saturated at 393/400 — the policy holds
the ball for the full episode and rotates it steadily. Its entropy anneals
smoothly and monotonically from ~22.8 down to ~7.0 over the run, the textbook
signature of healthy on-policy exploration that gradually sharpens into
exploitation without collapsing. With a recurrent (GRU) actor-critic and a
dense, well-shaped single-object reward, on-policy PPO is hard to beat in this
regime.

**SAPG (sound, but its advantage is latent at 512 envs).** SAPG (Split and
Aggregate Policy Gradients) augments PPO with off-policy data reuse by
partitioning environments into chunks and aggregating gradients across them; in
this run its measured off-policy fraction held at ~0.38. It learns a genuinely
working policy (peak 47.3, final 32.1, episode length 356) and notably *retains
more entropy* than PPO (~22.8 → ~12.1), i.e. it stays more exploratory for
longer. But it does not beat PPO here — which is the expected result, not a
failure: SAPG is designed to pay off at very large environment counts
(thousands to tens of thousands) and on harder, multi-task or
exploration-bottlenecked problems, where its off-policy reuse and chunked
aggregation overcome PPO's sample inefficiency. At a deliberately
matched-and-modest 512 envs with `num_chunks=8` (just 64 envs per chunk), it is
operating far below its intended regime — the project's normal SAPG launch
command uses 4096 envs. The benchmark therefore handicaps SAPG on purpose, in
the name of a fair per-step comparison.

**SAC (sample-efficient in theory, unstable in practice here).** SAC is a
max-entropy off-policy method that should be more sample-efficient than
on-policy PPO. In this run it was rough: reward dipped hard early (down to ~−45)
and the policy's entropy *collapsed through zero into negative territory*
(log-std driven down to ~−16 by automatic temperature tuning), indicating the
policy became near-deterministic very early. It nonetheless recovered in the
back half to a working policy (peak 33.5, final 26.7, episode length 394). The
takeaway is that off-policy critic learning on a 16-DoF contact-rich task is
viable but temperamental, and sensitive to the entropy-temperature schedule.

**TD3 (failed to learn this task).** TD3 is deterministic off-policy and never
got off the ground here — it spiked to ~−200 reward early, never reached
positive reward, and finished at −4.6 with an episode length of only 297 (it
keeps dropping the object). On a high-dimensional contact-rich task, TD3's
exploration (a fixed Gaussian noise added to a deterministic actor, with no
entropy bonus) is too weak to discover a stable grasp, and its twin critics
struggle with the discontinuous contact dynamics. This is a useful negative
result: it concretely demonstrates *why* the field defaults to on-policy or
max-entropy methods for in-hand manipulation.

> **[ Insert reward-vs-step plot here — PPO and SAPG climbing, SAC noisy-then-recovering, TD3 flat-negative ]**

### 2.3 Discussion

The headline is that **on-policy PPO dominates at single-object, modest-env
scale**, SAPG is a sound second whose strengths are deliberately masked by the
controlled setup, SAC is workable but unstable, and TD3 fails outright. This
is consistent with the broader literature on in-hand manipulation and is a
genuinely informative result rather than a foregone conclusion — it tells us
that scaling SAPG up (more envs, harder/multi-object tasks) is the *interesting*
direction, whereas the off-policy methods would need substantial stabilisation
work before they are competitive on this problem class.

---

## 3. New tasks and the difficulties they exposed

Moving beyond the well-behaved ball led to genuinely harder problems. Two are
worth detailing because they each required changing a non-trivial part of the
pipeline.

### 3.1 Efficient grasp search for non-spherical objects (the hammer task)

In-hand rotation training does not start from nothing — it bootstraps from a
**grasp cache**: a precomputed set of stable initial hand poses for the object,
sampled across object scales (e.g. `leap_hand_in_ball_grasp_50k_s10.npy`). For
a ball or a cube this is easy, because the object is roughly symmetric and
almost any reasonable closing of the fingers around the canonical pose yields a
stable hold. For a **hammer** — long handle, offset head, strongly asymmetric
mass distribution — a single fixed canonical "power-grip" seed almost never
produces a stable grasp, so the naïve upstream approach (perturb randomly
around one fixed canonical pose, keep what holds) wastes the overwhelming
majority of its samples and makes finding usable poses slow and tedious.

The fix was to replace fixed-seed sampling with a **per-environment iterative
refinement** that hill-climbs around each environment's own running-best pose
rather than a global canonical pose (see
[leap_hand_grasp.py](../src/LEAP_Hand_Sim/leapsim/tasks/leap_hand_grasp.py)).
Each of the 1024 parallel search environments tracks its `best_pose_per_env`
and `best_score_per_env`; on every reset it samples a new attempt within a
`grasp_dof_search_radius` of *its own best so far*, evaluates it, and promotes
the pose only if it improves — effectively an embarrassingly-parallel
`(1+λ)`-style local search / hill-climb running across all environments at
once. Crucially, the **fitness function** was also enriched: instead of a binary
"did it hold," `_episode_fitness` combines a fingertip-to-object-surface
distance term with a contact-duration term (rewarding a stable *hold* over
time), a minimum-contact-force requirement, and a minimum number of contacting
fingers. This lets the search distinguish a genuine "four fingers resting on the
surface, lightly holding" pose from a "fingers flailing near the object" pose
that a distance-only metric would score equally. Together these changes made
grasp-finding for the non-spherical / non-cubic object class — the hammer in
particular — substantially more efficient, turning a tedious manual search into
an automated one, and the resulting caches
(`leap_hand_in_hammer_grasp_50k_s*.npy`) feed directly into hammer rotation
training.

### 3.2 Tuning exploration to avoid policy collapse

Across the experiments, the single most common failure mode was **entropy
mismanagement**, and two recorded runs illustrate the two opposite ways it goes
wrong. In one PPO run, entropy *exploded* (from ~11 up past ~96) as the policy's
action variance blew up, and reward correspondingly crashed from near-zero down
to −32 — the policy diverged into noise. In one SAPG run, the opposite
happened: entropy *collapsed* (from ~23 down to ~4), the policy became
prematurely overconfident and near-deterministic, reward peaked at ~38 and then
fell off a cliff to −85 as the now-brittle deterministic policy left the region
it had learned. Both are failures of the same knob from opposite ends.

The practical response was to treat the entropy coefficient (and, for the
off-policy methods, the temperature schedule), the learning rate, and the KL /
clip constraints as a coupled exploration budget to be tuned per task rather
than copied across tasks. The well-behaved benchmark PPO run in §2 is what
correct tuning looks like: a smooth monotone entropy anneal from ~22.8 to ~7.0
with steadily rising reward. The goal throughout was to keep the policy
exploring long enough to discover stable manipulation strategies *without*
either diverging into noise or collapsing into a brittle deterministic policy —
a balance that turned out to be task-specific and one of the more delicate parts
of the whole pipeline.

> **[ Insert entropy-vs-iteration plot here — healthy anneal vs. the explosion and collapse runs ]**

---

## 4. Subsequent plans

The remaining work is more forward-looking. Each item below is a direction with
a clear motivation and a sketch of the technical challenges, rather than a
completed result.

### 4.1 Migrate from Isaac Gym to Isaac Lab

The current stack is built on **Isaac Gym Preview 4**, which NVIDIA has
deprecated and no longer supports. This already imposes hard, painful
constraints on the project: it ships pre-compiled `.so` files only for Python
3.6–3.8 (the repo is pinned to Python 3.8 with no path forward), it forces an
old NumPy (1.21–1.23) because `rl_games` still references the long-removed
`np.float`, and the entire headless-WSL2 Vulkan workaround in §1 exists only
because Isaac Gym's renderer cannot run in this environment. **Isaac Lab** (built
on Isaac Sim / Omniverse and the newer warp-based physics) is the supported
successor: it runs on current Python and PyTorch, has a maintained ecosystem and
active task library, and — importantly for this project — a working camera and
rendering path, which would let real in-simulator video supplement (or replace)
the Rerun workaround. The migration is non-trivial: the task implementations
(`leap_hand_rot.py`, `leap_hand_grasp.py`) are written against Isaac Gym's
tensor API and `vec_task` base class, the reward shaping and domain
randomisation would need to be ported to Isaac Lab's manager-based environment
abstraction, and the `rl_games` integration would have to be re-validated (or
swapped for Isaac Lab's RL wrappers). The grasp caches and the conceptual
pipeline carry over, but essentially every layer that touches the simulator API
has to be rewritten. The payoff is escaping a dead-end dependency stack and
unlocking the richer sensing and rendering needed for the sim-to-real work
below — which makes this migration the natural prerequisite for the rest of the
plan rather than an optional cleanup.

### 4.2 A physical LEAP Hand and the sim-to-real gap

The most substantial personal extension is to move off-simulation entirely and
deploy trained policies on a **physical LEAP Hand**, turning the project into a
study of the **sim-to-real gap**. This is where dexterous manipulation gets
genuinely hard: a policy that rotates a ball flawlessly in PhysX will, deployed
naïvely, encounter mismatches in actuator dynamics and latency, joint friction
and backlash, contact and friction models that PhysX only approximates, sensor
noise and bias, and an observation pipeline that on hardware must be estimated
(object pose, contact state) rather than read exactly from a simulator tensor.
Several mitigations already in the pipeline are directly relevant: the policy is
**recurrent (GRU-based)**, which helps it cope with the partial observability
and latency of real hardware; the training already uses **domain randomisation**
over object scale and physics parameters; and the grasp-search fitness in §3.1
explicitly rewards *robust, multi-finger, sustained-force* holds rather than
fragile fingertip touches, which should transfer better. The concrete plan is to
widen and systematise domain randomisation (mass, friction, actuator gains,
latency, observation noise), build a real observation/estimation stack for
object pose, and quantify the gap by measuring the same in-hand rotation task in
sim and on hardware under identical commands. Realistically this is the
highest-risk, highest-reward part of the project, and it depends on the Isaac
Lab migration (§4.1) for the improved sensing and rendering, and benefits from
the SAPG scaling work (§2.3) for training the more robust, randomisation-heavy
policies that survive contact with reality.

### 4.3 Diffusion policies as an alternative action representation

The third direction is to explore a **diffusion policy** as an alternative to
the Gaussian-MLP/GRU actor used by PPO and friends. Instead of outputting a
single action (or the parameters of a unimodal Gaussian over actions) at each
step, a diffusion policy models the *distribution* over short action sequences
by iteratively denoising sampled trajectories, conditioned on the current
observation. The attraction for dexterous manipulation is that the action
distribution for contact-rich tasks is often **multimodal** — there can be
several distinct, equally-good ways to reposition the fingers — and a unimodal
Gaussian policy is forced to average over them or commit to one, whereas a
diffusion policy can represent and sample from the full multimodal distribution,
producing smoother, more coherent action sequences. This is not free: diffusion
policies are typically trained by **imitation** on demonstration data rather than
by on-policy RL, so adopting them shifts the problem toward collecting good
demonstrations (e.g. from the PPO/SAPG policies already trained here, or from
teleoperation), and they carry their **own sim-to-real challenges** — the
iterative denoising sampler is comparatively expensive to evaluate, which is a
real concern for a control loop that must run at a fixed real-time rate on
hardware. A plausible mitigation, and an interesting research thread in its own
right, is **asynchronous training/inference of the policy** — decoupling the
slow multi-step denoising from the fast control loop so that action generation
can run ahead of, or in parallel with, execution rather than blocking it.
Sequenced after the work above, this would let the project compare two
fundamentally different action-representation paradigms — explicit stochastic
policy-gradient vs. learned generative denoising — on the same hand, the same
tasks, and ideally the same sim-to-real benchmark.

---

## 5. Summary

| Area | Status | Headline result |
|---|---|---|
| Rerun visualization / instrumentation | **Done** | Unblocked spatial debugging on headless WSL2 with zero changes to the sim stack |
| Algorithm benchmark (PPO/SAPG/SAC/TD3) | **Done** | Controlled 512-env / 82M-step comparison; PPO best (final 50.1), SAPG sound, SAC unstable, TD3 fails |
| Grasp search for non-spherical objects | **Done** | Per-env hill-climbing search + richer fitness; efficient hammer grasp caches |
| Entropy / exploration tuning | **Done** | Characterised both failure modes (explosion and collapse); reproducible healthy anneal |
| Isaac Lab migration | Planned | Escape deprecated Isaac Gym; unlock sensing/rendering |
| Physical hand / sim-to-real | Planned | Deploy policies on real LEAP Hand; quantify and close the gap |
| Diffusion policy | Planned | Multimodal action representation; async inference to meet real-time control |
