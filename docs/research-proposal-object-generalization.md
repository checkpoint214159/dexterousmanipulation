# Generalizing In-Hand Reorientation Across Object Groups via Compact Shape Priors

**Odyssey project — CP2107 · Project P034301**

*Research proposal (self-directed problem, project part 2). Companion to the
[progress report](odyssey-progress-report.md).*

---

## 1. Problem statement

The training stack learns in-hand object reorientation with the 16-DoF LEAP
Hand. Crucially, the policy is **blind to the object**: the 102-dimensional
observation ([`tasks/leap_hand_rot.py`](../src/LEAP_Hand_Sim/leapsim/tasks/leap_hand_rot.py))
is *proprioceptive only* — joint positions and action targets over a short
(~3-step) history. Object variation is randomized (scale, mass, centre of mass,
friction, and — via `object.type` — shape) but **hidden**: the policy must infer
whatever it can about the object implicitly, from how the object responds to its
own actions.

This works well *within* a shape class. Domain randomization over continuous
properties (a 0.9× vs 1.1× cube; different mass/friction) shows up as smooth
differences in contact dynamics that a short proprioceptive history can adapt
to. Our hypothesis is that it breaks *across* object groups:

> **A single proprioceptive policy cannot generalize across object categories,
> because categorical shape is not observable from proprioceptive history.**

A sphere rolls freely, a cuboid pivots on faces and edges, a cylinder rolls on
one axis, an off-centre tool needs regrasping — these demand qualitatively
different strategies, and "which category am I holding" is not recoverable from
a 3-step joint-angle window fast enough to act on.

The research question is then: **without resorting to a large vision or
vision-language model, can a *compact* prior `z` — a low-dimensional shape
descriptor — restore cross-group generalization?** We are explicitly interested
in the *minimal* prior: the cheapest, lowest-dimensional `z` that recovers most
of the generalization an oracle would give.

---

## 2. Hypotheses and research questions

- **H1 (gap).** A naive PPO policy with proprioceptive observation, trained on a
  mixture of object groups, generalizes poorly — both to held-out instances
  within a group and, more severely, to held-out groups.
- **H2 (cause).** The dominant cause is *observability of categorical shape*,
  not network capacity or gradient interference between strategies.
- **H3 (remedy).** Conditioning the policy on a compact shape prior `z` — known
  (privileged) at training time and *estimated* cheaply at deployment —
  recovers cross-group generalization without visual foundation models.

| RQ | Question | Decisive experiment |
|----|----------|---------------------|
| RQ1 | Is there a cross-group generalization gap? | Specialist vs naive-generalist, per-group + zero-shot held-out group |
| RQ2 | Is the *observation space* the cause? | Add a **privileged** oracle prior; if it generalizes and proprioception doesn't, observability is the bottleneck |
| RQ3 | What is the *minimal* `z`? | Ablate `z` from scalars → hand-crafted geometry → learned embedding |
| RQ4 | Can `z` be estimated at deploy without heavy vision? | Replace privileged `z` with an online/one-glimpse estimate; measure the drop |

The single most important design move is the **privileged control in RQ2**: it
converts a vague "PPO is bad at this" into a causal claim about the observation
space, and it directly reuses infrastructure already in the repo (see §3).

---

## 3. Background: what the current system does and doesn't observe

- **Observation (student).** Proprioceptive history only; no object channel by
  default (`include_obj_pose` is off).
- **Privileged information (teacher).** The task already exposes
  `privInfo: {enableObjPos, enableObjScale, enableObjMass, enableObjCOM,
  enableObjFriction}` — the classic HORA/LEAP *asymmetric* channel used to train
  a teacher that "sees" the object. This is the natural home for `z`: a
  privileged shape prior is a new channel alongside these scalars.
- **Reward.** Angular velocity about a target axis — object-agnostic in form, so
  nothing in the objective is class-specific.
- **Grasp caches.** Each object family needs a stable initial grip. The
  `LeapHandGrasp` task settles the hand from a canonical seed pose and saves 50k
  grasp states (16 hand DoF + 7 object pose) to
  `cache/<name>_grasp_50k_s<scale>.npy`, keyed by *family* and *scale*
  (one cache shared across the geometrically-similar objects in a family).
- **Object plumbing.** `object.type` supports `A+B` mixtures and per-category
  subset globbing (`cuboid_<subset>`, `cylinder_<subset>`, and — added for this
  study — `sphere_<subset>`), so a directory of URDFs is one trainable family.

These four facts define both the opportunity (privileged channel + object
plumbing already exist) and the obstacles (§6): every new family needs a grasp
cache, which needs a canonical seed pose.

---

## 4. Approach

### 4.1 Establishing the problem — the three-policy control

The proof-of-problem is a controlled comparison over the same object set and
seeds:

| Policy | Observation | Trained on | Reveals |
|--------|-------------|-----------|---------|
| **Specialist** | proprio | one group | per-group performance ceiling |
| **Naive generalist** | proprio | all groups | the hypothesized gap (H1) |
| **Privileged generalist** | proprio **+ oracle `z`** | all groups | is object info sufficient? (H2) |

All three are evaluated per-group and **zero-shot on a held-out group**. The
diagnostic outcome: if the privileged generalist generalizes across groups but
the naive one does not, the observation space is causally implicated (H2). If
even the privileged one fails, the problem is capacity/interference and we pivot
to architecture (e.g. shape-conditioned experts) rather than observation.

**Metrics.** Rotation reward and time-to-drop as primary; a normalized
"success" (sustained rotation without drop) per group; and the *generalization
gap* = specialist − generalist, per group and held-out.

### 4.2 The prior `z` — design space

`z` is a compact per-object descriptor injected into the actor (and, during
training, the critic). We propose a ladder of increasing richness so RQ3 is an
ablation, not a single design bet:

- **`z0` — scalars (oracle baseline).** The existing `privInfo` continuous
  properties (scale, mass, COM, friction). No shape. Establishes how far
  non-shape privileged info alone goes.
- **`z1` — hand-crafted geometry (the target of the "minimal prior" claim).**
  Analytic, cheap, ~5–15 dims: bounding-box extents, principal-axis aspect
  ratios, inertia-tensor eigenvalues, volume, and a couple of
  convexity/curvature statistics. Computable directly from a mesh or a point
  cloud; no encoder training.
- **`z2` — learned embedding.** A small encoder (PointNet-style over a sampled
  point cloud, or an occupancy/SDF autoencoder) producing ~16–64 dims. Pretrained
  offline on the object set (reconstruction or contrastive), then frozen or
  co-trained.
- **`z3` — categorical code.** A coarse per-family code (one-hot or learned).
  Structurally *identical* to the conditioning code our SAPG implementation
  already appends to observations (§7) — so the mechanism for "condition a shared
  backbone on a per-object latent" is already in the codebase and validated.

The empirical question is where on this ladder the generalization gap closes. A
clean, publishable result would be: **`z1` (a handful of analytic geometric
numbers) recovers most of the oracle's cross-group generalization** — i.e. the
policy needs *shape*, but only a coarse, interpretable summary of it, not a
learned visual embedding.

### 4.3 Train-time privileged vs deploy-time estimated `z` (teacher–student)

This is the crux of RQ4 and of any eventual sim-to-real story. At **training**
time `z` is *privileged*: computed exactly from the known asset. At
**deployment** (held-out objects, and ultimately a real hand) `z` must be
*estimated*. The pathways, with the training each requires and its sensing cost:

| Estimation pathway | How `ẑ` is obtained at deploy | Training required | Sensing at deploy | Sim2real viability |
|---|---|---|---|---|
| **Proprioceptive adaptation** (HORA-style) | regress `ẑ` online from proprioceptive history | supervised regression to privileged `z` (or joint RL) | none (proprio only) | high; but can `ẑ` capture *categorical* shape from touch alone? — an open question this project can answer |
| **One-glimpse encode** | a single depth image / point cloud at grasp time → encode once → hold `ẑ` fixed during rotation | train the encoder (offline or with the policy) | one RGB-D frame | high; cheap sensor, not a VLA |
| **Inference-time reconstruction** | reconstruct object geometry (depth glimpse and/or a few exploratory touches) → compute `z1`/`z2` | reconstruction pipeline + estimator | RGB-D and/or tactile | medium; heavier, but yields interpretable geometry |
| **Category prior** | assume/known family label → lookup `z` | none | prior knowledge | trivial where the object class is known |

The user's proposed pipeline — *reconstruct the object at inference, encode it
into `z`, then parameterize the policy on `z`* — is the "inference-time
reconstruction" row. It is attractive because the reconstruction step is decoupled
from control and can be as simple as fitting a bounding box / principal axes to a
single depth glimpse (which directly yields `z1`), avoiding any large model.

The scientific pairing that makes this a contribution: **measure `z` vs `ẑ`
jointly** — the *minimal* privileged `z` that closes the gap (RQ3) *and* the
*cheapest estimator* that reproduces it at deploy (RQ4). A prior that is powerful
but un-estimable is useless; a prior that is estimable but too weak doesn't
generalize. The sweet spot is the deliverable.

### 4.4 Analysis of results and interpretability

Beyond aggregate success, we ask *how* `z` shapes the learned strategy — turning a
performance number into mechanistic understanding and connecting the emergent
behavior to the human-hand-control literature (§9.7). In-hand rotation is
fundamentally *finger gaiting* (the cyclic make/break of fingertip contacts that
drives the object), so we analyze behavior at that level rather than as raw
16-DoF joint traces.

**Behavior descriptors.** Per rollout: the gait / contact sequence (which fingers
contact, in what order; limit-cycle period and inter-finger phase), the achieved
rotation axis and angular velocity, and — the key reduction — the joint
trajectory **projected onto a postural-synergy (eigengrasp) basis**
[Santello et al., 1998; Ciocarlie et al., 2007], compressing 16-DoF motion to a
few interpretable synergy activations over time (as done for human grasps).

**Linking `z` to behavior (causally, not just correlationally).**
- **Interventional latent traversal** — hold the object/state fixed and *sweep*
  `z` along a path (interpolate sphere→cuboid→cylinder, or traverse `z`'s
  principal axes); a morphing gait is then caused by `z`, not the object. *The
  central figure.*
- **`z`↔behavior alignment** — embed the synergy-trajectories and measure
  alignment to `z`-space via CCA / Procrustes, or "does `z` predict the strategy
  cluster." Quantifies that `z` organizes behavior.
- **Counterfactual (wrong-`z`) probe** — drive object A with `z_B`; measure the
  strategy shift toward B and the rotation-performance drop. Shows the policy
  *uses* `z` as a strategy selector and quantifies its reliance.
- **Sensitivity** — `∂(behavior)/∂z`: which `z`-dimensions control which
  behavioral axes.

**Figures.** A latent-traversal filmstrip (z-interpolation × hand keyframes); a
2-D `z`-projection coloured by a behavior descriptor (rotation axis / gait
frequency / lead finger); synergy phase portraits per category; a counterfactual
object×`z` performance matrix.

**Interpretability highlight.** Do the learned, `z`-conditioned synergies
resemble *human* postural synergies [Santello et al., 1998]? Does `z` reorganize
the hand's synergy manifold the way object identity reshapes human pre-grasp
shaping? We further label the discovered strategies against the Bullock–Dollar
manipulation taxonomy [Bullock et al., 2013]. A quantitative learned-vs-human
synergy comparison is largely absent from the literature (§9.7).

**Rigor.** Smooth traversal requires a *continuous* `z` (favoring `z1`/`z2` over a
one-hot `z3`); quasi-periodic gaits must be phase-aligned (DTW) before comparison;
we use shuffled-/random-`z` controls, multiple seeds with reported variance, and
interventions (not correlations) for causal claims — and distinguish "`z` changes
behavior" from "`z` changes behavior *usefully*." This is an understanding
contribution, secondary to the core "does minimal `z` close the gap" result.

---

## 5. Object taxonomy and data

We group objects by *what changes the manipulation strategy*, not by semantic
label — a two-level taxonomy:

| Category (strategy) | Why distinct | Instance variation |
|---|---|---|
| Sphere / ellipsoid | rolls freely, no stable faces | radius |
| Cuboid / box | pivots on faces and edges | aspect ratio, size |
| Cylinder / prism | rolls on one axis only | radius, length |
| *(future)* Irregular / off-centre tool | asymmetric inertia, regrasp | COM offset |

Two generalization axes fall out: **held-out instances** (within-category) and
**held-out categories** (cross-category, the hard case). The first cut is
procedural primitives (native URDF geometry, analytic collision, physically
correct per-shape inertia) — 18 objects across 3 categories × {train, heldout}
(§7). Scaling to real meshes (YCB-style) is future work gated by mesh→URDF +
convex decomposition and per-object grasp seeding.

---

## 6. Obstacles and mitigations

1. **Grasp-cache generation depends on a good canonical seed pose.** A stable
   initial grip is required per family; the settling search is sensitive to the
   hand's canonical start pose, which is hand-tuned per object today
   (`scripts/gen_hammer_grasp.sh`). *Mitigation:* per-*category* pose templates
   for primitives (regular geometry → one template per category). *Open at
   scale:* arbitrary objects need automated grasp seeding (antipodal sampling or
   a learned grasp predictor) — itself a mini research problem, deferred.
   *Watch item:* grasp *quality* must be gated — our first automated run filled
   caches with best-attempt grasps at low reported success; caches should be
   spot-checked (visually, via Rerun) before trusting them for the proof.
2. **Defining "an object group."** Handled for primitives via the taxonomy
   above; the categorical axis is the scientifically important one. Scale-out
   needs a mesh ingestion + normalization pipeline.
3. **Defining and estimating `z` at scale.** §4.2–4.3. The main risk is that
   proprioceptive adaptation cannot recover categorical shape (only continuous
   properties), forcing a one-glimpse or reconstruction estimator — which is
   still far lighter than a VLA and is the realistic sim2real path.

---

## 7. Infrastructure already built

- **Training algorithms.** PPO (rl_games baseline) plus locally-added SAC, TD3,
  and a from-scratch re-derivation of **SAPG** (Split-and-Aggregate Policy
  Gradients, ICML 2024) with recurrent (GRU) support. SAPG matters here for two
  reasons: it scales on-policy RL across large, diverse env pools (useful when
  training over many object groups at once), and its per-chunk **conditioning
  code** is exactly the mechanism `z3`/`z` needs — a shared backbone conditioned
  on a per-instance latent, already validated end-to-end.
- **Object-set automation** ([`tools/`](../src/LEAP_Hand_Sim/leapsim/tools/)):
  `gen_primitive_objects.py` (procedural URDFs with correct inertia, train/heldout
  subsets) and `gen_grasp_caches.py` (resumable, background batch grasp-cache
  runner with per-category pose hooks and status logging). Validated end-to-end:
  generated families load in sim and produce valid grasp caches.
- **Privileged channel.** `privInfo` already provides oracle object scalars —
  the substrate for the privileged `z` control.

---

## 8. Experimental plan / milestones

- **M0 — Data.** Generate primitive families; produce and *quality-gate* grasp
  caches for all families × scales. *(infra done; caches in progress)*
- **M1 — Proof (RQ1/RQ2).** Specialist vs naive vs privileged generalist;
  per-group + held-out-group. Establish the gap and its cause.
- **M2 — Minimal prior (RQ3).** Ablate `z0 → z1 → z2` as privileged input; find
  where the gap closes.
- **M3 — Deploy estimation (RQ4).** Replace privileged `z` with `ẑ`
  (proprioceptive adaptation vs one-glimpse encode); measure the drop; pick the
  minimal-yet-estimable `z`.
- **M4 — Scale (stretch).** More categories / real meshes; sim-to-real with the
  physical hand.

---

## 9. Related work

### 9.1 Dexterous in-hand manipulation with RL

Learned in-hand manipulation was popularized by OpenAI's Dactyl, which
reoriented a cube on a Shadow Hand using RL with heavy domain randomization
[Andrychowicz et al., 2020], later scaled with automatic domain randomization to
solve a Rubik's cube [OpenAI et al., 2019]. The line most relevant to us is
*proprioception-driven* in-hand rotation: **HORA** rotates diverse objects using
only joint history plus a rapid motor-adaptation module [Qi et al., 2022], and
the **LEAP Hand** platform on which this stack is built provides a low-cost
anthropomorphic hand and sim [Shaw et al., 2023]. Tactile-only variants —
**Touch Dexterity / "Rotating without Seeing"** [Yin et al., 2023] and
**AnyRotate** [Yang et al., 2024] — remove vision entirely and rotate objects
from binary/marker tactile signals, establishing that rich object priors are not
strictly necessary *within* their object distributions but leaving cross-group
generalization largely open.

### 9.2 Privileged learning and rapid adaptation (teacher–student)

The teacher–student / privileged-information paradigm underpins most of this
work: a teacher trained with privileged simulator state is distilled into a
student that estimates that state online. It originates in legged locomotion
[Lee et al., 2020; **RMA**, Kumar et al., 2021] and is exactly the mechanism
behind HORA's adaptation module [Qi et al., 2022]. Our proposal sits directly in
this paradigm — the existing `privInfo` channel is the teacher substrate — and
asks a sharper question: *what is the minimal privileged signal (a shape prior
`z`) that must be distilled to generalize across object groups?*

### 9.3 Object- and shape-conditioned manipulation & generalization

Several systems generalize reorientation across many objects by conditioning on
object geometry. Chen et al. reorient 2000+ geometrically distinct objects with a
teacher–student system and a gravity curriculum [Chen et al., 2021], extended to
novel complex shapes from point clouds in **Visual Dexterity** [Chen et al.,
2023]. Closest to our idea, **RotateIt** performs multi-axis fingertip rotation
while *inferring object shape and physical properties online* through a
**visuotactile transformer** [Qi et al., 2023]. These works establish that shape
conditioning enables generalization — but they do so with **rich, high-capacity
multimodal representations** (point clouds, vision+touch transformers). Our
contribution is orthogonal and deliberately minimalist (see §9.7).

### 9.4 Shape representations and geometric priors

Candidate encodings for `z` draw on the 3D representation literature: point-cloud
encoders [**PointNet**, Qi et al., 2017], implicit shape functions [**DeepSDF**,
Park et al., 2019; **Occupancy Networks**, Mescheder et al., 2019], and — as a
possible *source* of geometry at inference — radiance-field reconstructions
[**3D Gaussian Splatting**, Kerbl et al., 2023]. We treat these as the upper
(`z2`) rung of our prior ladder and contrast them against hand-crafted analytic
descriptors (`z1`), whose sufficiency is the paper's central empirical question.

### 9.5 Grasp synthesis

Our per-family grasp caches connect to data-driven grasp generation:
**DexGraspNet** [Wang et al., 2023] and generative/optimization grasp synthesis
[**GenDexGrasp**, Li et al., 2023; **UniDexGrasp**, Xu et al., 2023] produce
diverse dexterous grasps and are the natural path to *automated grasp seeding*
when we scale beyond primitives (§6, obstacle 1).

### 9.6 Scaling massively-parallel RL

The training regime rests on GPU-parallel simulation [**Isaac Gym**, Makoviychuk
et al., 2021; Rudin et al., 2022]. Scaling on-policy RL across large, diverse env
pools motivated **DexPBT** [Petrenko et al., 2023] and **SAPG** [Singla et al.,
2024]; we re-derived SAPG in this stack (§7), and note that its per-chunk
conditioning code is the same "shared backbone + per-instance latent" mechanism
that a shape prior `z` requires.

### 9.7 Human hand control: synergies, motor neuroscience, and manipulation taxonomy

Both our low-dimensional prior and our analysis (§4.4) connect to the study of
human hand control. Santello et al. showed hand postures across 57 objects
collapse to ~2 principal components — *postural synergies* [Santello et al.,
1998] — which robotics operationalized as *eigengrasps* for low-dimensional grasp
planning [Ciocarlie et al., 2007]. This both motivates the plausibility of a
compact `z` and supplies a synergy basis for analyzing learned gaits.
Motor-neuroscience work shows humans set *anticipatory* grip parameters from
predicted object properties using vision at grasp, then rely on tactile and
proprioceptive feedback during manipulation [Johansson & Westling, 1988;
Johansson & Flanagan, 2009] — a direct analog of our one-glimpse→`z`→proprioceptive
control design (§4.3, Path B). Manipulation taxonomies give a vocabulary to
classify the strategies a policy discovers [Elliott & Connolly, 1984; Bullock et
al., 2013; Feix et al., 2016]. That RL rediscovers human-like finger gaiting and
grasp types without demonstrations [Andrychowicz et al., 2020] legitimizes the
comparison — and the near-absence of *quantitative* learned-vs-human synergy
analysis is an opening for §4.4.

### 9.8 Positioning and novelty

Prior shape-conditioned systems answer *whether* object information helps
(it does) using **rich** representations — point clouds, SDFs, or vision+touch
transformers [Chen et al., 2021; Chen et al., 2023; Qi et al., 2023]. Our
question is different and, to our knowledge, under-studied: **what is the
*minimal, interpretable* prior `z` that closes the cross-*category*
generalization gap, and the *cheapest estimator* that reproduces it at
deployment — under an explicit no-visual-foundation-model constraint?** The
contribution is the joint minimal-prior / privileged-to-estimated framing and its
empirical characterization, not shape conditioning per se. A focused comparison
against RotateIt [Qi et al., 2023] and AnyRotate [Yang et al., 2024] is a
prerequisite before committing.

---

## 10. Risks

- **Proof depends on cache quality.** A weak generalist could reflect bad grasp
  caches rather than the observation space. *Mitigation:* quality-gate caches;
  verify specialists reach a sane ceiling first.
- **The gap may be small at our scale.** With only 3 primitive categories and a
  single GPU, effects may be modest. *Mitigation:* choose maximally
  strategy-distinct categories (sphere vs cuboid vs cylinder) and lean on the
  held-out-category axis.
- **`ẑ` may not be recoverable from proprioception.** If categorical shape
  needs a glimpse, the "no vision at all" ambition weakens — but a single depth
  frame is still far from a VLA and is the honest sim2real answer.
- **Novelty risk.** Shape-conditioned manipulation is active; the framing must
  carry the contribution.

---

## References

- Andrychowicz, M., et al. (2020). *Learning Dexterous In-Hand Manipulation.*
  International Journal of Robotics Research (IJRR). arXiv:1808.00177.
- OpenAI, et al. (2019). *Solving Rubik's Cube with a Robot Hand.*
  arXiv:1910.07113.
- Qi, H., Kumar, A., Calandra, R., Ma, Y., & Malik, J. (2022). *In-Hand Object
  Rotation via Rapid Motor Adaptation (HORA).* Conference on Robot Learning
  (CoRL). arXiv:2210.04887.
- Shaw, K., Agarwal, A., & Pathak, D. (2023). *LEAP Hand: Low-Cost, Efficient,
  and Anthropomorphic Hand for Robot Learning.* Robotics: Science and Systems
  (RSS). arXiv:2309.06440.
- Yin, Z.-H., Huang, B., Qin, Y., Chen, Q., & Wang, X. (2023). *Rotating without
  Seeing: Towards In-hand Dexterity through Touch.* Robotics: Science and Systems
  (RSS). arXiv:2303.10880.
- Yang, M., Church, A., Lin, Y., Ford, C. J., Li, H., Psomopoulou, E., Barton,
  D. A., & Lepora, N. F. (2024). *AnyRotate: Gravity-Invariant In-Hand Object
  Rotation with Sim-to-Real Touch.* Conference on Robot Learning (CoRL).
  arXiv:2405.07391.
- Lee, J., Hwangbo, J., Wellhausen, L., Koltun, V., & Hutter, M. (2020).
  *Learning quadrupedal locomotion over challenging terrain.* Science Robotics,
  5(47).
- Kumar, A., Fu, Z., Pathak, D., & Malik, J. (2021). *RMA: Rapid Motor Adaptation
  for Legged Robots.* Robotics: Science and Systems (RSS).
- Chen, T., Xu, J., & Agrawal, P. (2021). *A System for General In-Hand Object
  Re-Orientation.* Conference on Robot Learning (CoRL), Best Paper Award.
  arXiv:2111.03043.
- Chen, T., Tippur, M., Wu, S., Kumar, V., Adelson, E., & Agrawal, P. (2023).
  *Visual Dexterity: In-Hand Reorientation of Novel and Complex Object Shapes.*
  Science Robotics, 8(84).
- Qi, H., Yi, B., Suresh, S., Lambeta, M., Ma, Y., Calandra, R., & Malik, J.
  (2023). *General In-Hand Object Rotation with Vision and Touch (RotateIt).*
  Conference on Robot Learning (CoRL).
- Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017). *PointNet: Deep Learning on
  Point Sets for 3D Classification and Segmentation.* CVPR.
- Park, J. J., Florence, P., Straub, J., Newcombe, R., & Lovegrove, S. (2019).
  *DeepSDF: Learning Continuous Signed Distance Functions for Shape
  Representation.* CVPR.
- Mescheder, L., Oechsle, M., Niemeyer, M., Nowozin, S., & Geiger, A. (2019).
  *Occupancy Networks: Learning 3D Reconstruction in Function Space.* CVPR.
- Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). *3D Gaussian
  Splatting for Real-Time Radiance Field Rendering.* ACM Transactions on Graphics
  (SIGGRAPH).
- Wang, R., Zhang, J., Chen, J., Xu, Y., Li, P., Liu, T., & Wang, H. (2023).
  *DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset for General
  Objects Based on Simulation.* ICRA. arXiv:2210.02697.
- Li, P., Liu, T., Li, Y., Geng, Y., Zhu, Y., Yang, Y., & Huang, S. (2023).
  *GenDexGrasp: Generalizable Dexterous Grasping.* ICRA. arXiv:2210.00722.
- Xu, Y., Wan, W., Zhang, J., et al. (2023). *UniDexGrasp: Universal Robotic
  Dexterous Grasping via Learning Diverse Proposal Generation and Goal-Conditioned
  Policy.* CVPR. arXiv:2303.00938.
- Makoviychuk, V., et al. (2021). *Isaac Gym: High Performance GPU-Based Physics
  Simulation for Robot Learning.* NeurIPS Datasets and Benchmarks. arXiv:2108.10470.
- Rudin, N., Hoeller, D., Reist, P., & Hutter, M. (2022). *Learning to Walk in
  Minutes Using Massively Parallel Deep Reinforcement Learning.* Conference on
  Robot Learning (CoRL).
- Petrenko, A., Allshire, A., State, G., Handa, A., & Makoviychuk, V. (2023).
  *DexPBT: Scaling up Dexterous Manipulation for Hand-Arm Systems with Population
  Based Training.* Robotics: Science and Systems (RSS).
- Singla, J., Agarwal, A., & Pathak, D. (2024). *SAPG: Split and Aggregate Policy
  Gradients.* International Conference on Machine Learning (ICML).
  arXiv:2407.20230.
- Santello, M., Flanders, M., & Soechting, J. F. (1998). *Postural Hand Synergies
  for Tool Use.* Journal of Neuroscience, 18(23), 10105–10115.
- Ciocarlie, M., Goldfeder, C., & Allen, P. (2007). *Dexterous Grasping via
  Eigengrasps: A Low-Dimensional Approach to a High-Complexity Problem.* RSS
  Workshop (see also Ciocarlie & Allen, *Hand Posture Subspaces for Dexterous
  Robotic Grasping*, IJRR 2009).
- Johansson, R. S., & Westling, G. (1988). *Programmed and triggered actions to
  rapid load changes during precision grip.* Experimental Brain Research, 71(1).
- Johansson, R. S., & Flanagan, J. R. (2009). *Coding and use of tactile signals
  from the fingertips in object manipulation tasks.* Nature Reviews Neuroscience,
  10(5), 345–359.
- Bullock, I. M., Ma, R. R., & Dollar, A. M. (2013). *A Hand-Centric
  Classification of Human and Robot Dexterous Manipulation.* IEEE Transactions on
  Haptics, 6(2), 129–144.
- Feix, T., Romero, J., Schmiedmayer, H.-B., Dollar, A. M., & Kragic, D. (2016).
  *The GRASP Taxonomy of Human Grasp Types.* IEEE Transactions on Human-Machine
  Systems, 46(1), 66–77.
- Elliott, J. M., & Connolly, K. J. (1984). *A Classification of Manipulative Hand
  Movements.* Developmental Medicine & Child Neurology, 26(3), 283–296.

*Citation details (author lists, venues, years) should be re-verified against
the canonical source before external submission; several are cited from memory.*
