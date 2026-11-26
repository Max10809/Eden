# Manipulation Tasks

Eden's own manipulation tasks: arm-and-gripper reaching and lifting, a dexterous-hand lift, in-hand cube
reorientation, piano playing, two-part assembly, and hand-object tracking with dexterous hands. This page gives, for
each task, the robot, what the policy learns, its flags, what it needs installed, and the commands to train and
replay it. The ported benchmark suites (robosuite, LIBERO, SIMPLER, PolaRiS, Mochi, ...) have their own pages, linked
from the [task index](index.md), which also has the commands every task shares.

| Task | Robot | Flags | Default `num_envs` |
|---|---|---|---|
| [`reacher`](#reacher) | I2RT YAM arm | `--difficulty` | 4096 |
| [`manipulation_reach_droid`](#reach-droid) | Franka Panda + Robotiq 2F-85 | — | 4096 |
| [`manipulation_lift_box`](#lift-box) | Franka Panda + Robotiq 2F-85, or UR5e / UR10e + 2F-85 | `--manipulator`, `--reset_buffer` | 4096 |
| [`manipulation_lift_box_rgb`](#lift-box-from-rgb) | Franka Panda + Robotiq 2F-85 | — | 1024 |
| [`manipulation_lift_ycb_rgb`](#lift-ycb-from-rgb) | Franka Panda + Robotiq 2F-85 | `--object_properties` | 1024 |
| [`manipulation_dex_lift`](#dexterous-lift) | Franka + XHand1 (left) | — | 4096 |
| [`manipulation_g1_beaker_lift`](#g1-beaker-lift) | Unitree G1 + Dex3 hands, pelvis welded | — | 4096 |
| [`manipulation_g1_beaker_pour`](#g1-beaker-pour) | Unitree G1 + Dex3 hands, pelvis welded | — | 4096 |
| [`reorient`](#in-hand-reorientation) | Wuji Hand 1 (simplified collision model) | `--hand` | 4096 |
| [`piano`](#piano) | Two Wuji Hand 2 (beta2) | `--song` | 1024 |
| [`omnireset`](#omnireset) | UR5e + Robotiq 2F-85 | `--robot` | 4096 |
| [`flashdexretarget`](#flashdexretarget) | XHand, or two Sharpa Wave hands | `--motion`, `--robot`, `--learner` | 4096 |

`num_envs` is per GPU rank. Override it with `--num_envs N`.

## Reacher

The canonical Eden task config (`eden/tasks/benchmark/reacher/config.py`) and the template every other task follows:
a YAM arm moves its end effector to a target sampled in a box, with an exponential reaching reward, action-rate,
joint-limit and joint-velocity penalties, and a curriculum that stages the weight of the joint-velocity penalty. The arm uses incremental joint
position control (`ReferenceSource.DELTA`, target = current position + 0.05 × action).

```bash
eden train --task reacher --difficulty hard
eden inference --task reacher --checkpoint logs/reacher/<run>/model_<iter>.pt --rec
```

`--difficulty` sets three things together, because shrinking the target without sharpening the reward or shortening
the episode just makes a longer, sparser task:

| `--difficulty` | Target half-extent, xy / z (m) | `reaching_std` | Episode (s) |
|---|---|---|---|
| `easy` | 0.15 / 0.075 | 0.30 | 12 |
| `medium` (default) | 0.10 / 0.05 | 0.20 | 9 |
| `hard` | 0.05 / 0.025 | 0.10 | 6 |

The target box is centred on (0.4, 0.0, 0.3) m. [Your first task](../getting-started/first-task.md) builds a task
like this one from scratch.

## Reach (Droid)

`manipulation_reach_droid` is the reacher on the Droid rig (Franka arm + Robotiq 2F-85): joint-position arm control,
a uniform 3D target, an exponential reaching reward and the same penalties and curriculum. The end effector is the
midpoint of the gripper's finger pads. It mirrors mjlab's `Reach-{Robot}`.

```bash
eden train --task manipulation_reach_droid
```

!!! note "The Droid rig's arm is a Panda"
    `Droid` is a hand-less Franka Emika Panda with a Robotiq 2F-85, not the FR3 that mjlab pairs the gripper with:
    the rig was tuned on that arm and kept on it so trained policies stay valid. `Droid(arm=FrankaResearch3())`
    gives the FR3 pairing. See [Robots](../guides/robots.md).

## Lift box

A port of mjlab's `Lift-{Robot}`, trained from scratch: every episode starts at the arm's ready-to-grasp home pose
with the cube spawned in a region on the table, so the policy learns reach → grasp → move to target with no warm
start. Arm control is 6-DOF differential IK (`DifferentialIKController`); the gripper is a binary open/close
controller on the Robotiq drivers, which are force-limited so closing on the cube makes a friction grip instead of
crushing through it. The rewards follow mjlab's `lift_env_cfg`: shaped reaching, lifting, and coarse (std 0.3) and
fine (std 0.05) goal tracking gated on the lift.

```bash
eden train --task manipulation_lift_box
eden train --task manipulation_lift_box --manipulator ur5e_2f85
eden inference --task manipulation_lift_box --checkpoint logs/manipulation_lift_box/<run>/model_<iter>.pt --rec
```

| Flag | Choices / default | Effect |
|---|---|---|
| `--manipulator` | `droid` (default), `ur5e_2f85`, `ur10e_2f85` | The arm. Every rig carries the same Robotiq 2F-85, so the grasp point, gripper action and gripper rewards are unchanged; only the arm entity, the IK frame and joint set, and the experiment name change. The observation width follows the arm's DOF count, so checkpoints do not transfer between arms. |
| `--reset_buffer` | none | A reset-state buffer file to restore every reset from, replacing the cube placement and the arm-joint jitter. Capture one with `capture_reset_states(env, ["robot", "gripper", "cube"])` (`eden.utils.buffers.reset_states`). Rows are drawn uniformly; the goal command still draws each episode's goal. |

## Lift box from RGB

`manipulation_lift_box_rgb` keeps the lift-box physics, actions, rewards, terminations, events and curricula, and
changes the perception. A 128 × 72 wrist RGB camera on the tool flange feeds a spatial-softmax CNN (one conv stack
per image group ending in a soft-argmax that emits each channel's expected image location). The actor loses the
privileged object pose and must find the cube in the image; the critic keeps the full privileged state. Both read the
shared `camera` group.

```bash
eden train --task manipulation_lift_box_rgb
```

It renders with the gs-madrona batch renderer, which is not an Eden extra: install `gs-madrona` (Linux x86-64 only)
first.

!!! warning "Two Madrona traps this config avoids"
    Either one leaves the actor effectively blind, with no error. Madrona renders **ambient light only until lights
    are added**, and it fixes the light count when the scene is built, so the lights are declared on
    `BatchRendererOptions` and applied before `scene.build()`. And `BoxOptions` defaults to a grey surface, which is
    invisible on a grey table, so the cube is coloured red and the table darkened. Reuse both in any vision task on
    the batch renderer.

## Lift YCB from RGB

`manipulation_lift_ycb_rgb` extends the RGB lift with a different rigid YCB object in each env: master chef can,
tomato soup can, tuna can, pudding box, gelatin box, potted meat can, apple and foam brick. The objects are one
`GroupedEntityOptions` entity still named `cube`, so every inherited term works unchanged; Genesis distributes the
variants in contiguous blocks across the envs (128 envs per object at 1024 envs), and the variant an env holds is
fixed at build. Because each object rests at a different height, an `ObjectRestHeight` metric latches each env's
actual resting height after the reset settles, and the four lift rewards use it instead of one scalar.

```bash
eden train --task manipulation_lift_ycb_rgb --object_properties ycb
```

| `--object_properties` | Effect |
|---|---|
| `uniform` (default) | Every object at the default density and friction 2.0. |
| `ycb` | Each object at its published YCB mass (scaled with the mesh) and a friction by surface material. |

## Dexterous lift

`manipulation_dex_lift` keeps the lift-box structure (shaped reach, dense lift, grasped-and-lifted gate, coarse and
fine goal tracking, a goal-range curriculum, staged smoothness curricula, contact stiffening) and swaps the gripper for
a 12-DOF XHand1 (left). The rig is `FrankaXHand`: the hand on a right-angle coupling, palm down at the ready pose.
The arm uses differential IK and the hand a joint-space PD over its 12 DOFs. The end effector is the centroid of the
five fingertips, and a per-fingertip approach reward replaces the gripper's grasp bonus.

```bash
eden train --task manipulation_dex_lift
```

The policy is trained to be deployed on the real rig:

- It observes only what the rig measures: joint encoders, its own last action, the goal, forward kinematics of
  those, and the cube pose a camera would estimate from AprilTags on the cube, with that estimate's noise and a
  3-frame history. The cube's orientation is a 6D rotation, so the sign a pose estimator picks does not matter.
  Simulator-only state (hand joint velocities, cube velocity, fingertip-to-cube offsets) is critic-only.
- The hand keeps the URDF effort limits (0.4 / 1.1 N·m), every hand link meets the cube with stiffened contact, hand
  friction is drawn per env from U(0.8, 2.0) against a 0.5 cube and table, lift goals are 10–30 cm above the cube's
  rest position in the arm's base frame, and the action-rate penalty is 5× the lift-box task's.

!!! note "Deployment is not complete yet"
    The cube pose reaches the policy through the `cube` entity, so a deployment writes the camera's estimate into it
    each control step. The Franka and XHand backends ship with Eden; the AprilTag observer that does that write is
    not in Eden yet, and MuJoCo sim-to-sim cannot load this rig yet (the module docstring of
    `eden/tasks/manipulation/dex_lift/config.py` explains why).

## G1 beaker lift

`manipulation_g1_beaker_lift` is the dexterous lift on a humanoid at a lab bench: a `G1Dofs29Dex3` picks a 250 mL
beaker (a 7 × 9.5 cm cylinder, 0.12–0.30 kg) off a 0.85 m bench with its right Dex3 hand, keeps it upright, and holds
it at a goal 10–25 cm above the bench. It reuses the dexterous lift's goal tracking and curricula.

```bash
eden train --task manipulation_g1_beaker_lift
```

- **Upper body only.** The pelvis is welded to an invisible fixed mount (the G1 MJCF's free base cannot take
  `is_fixed_base=True`). The action is 13-D: differential IK at the right palm over the seven right-arm joints, whose
  deltas accumulate on the last joint target so the arm holds its pose without gravity compensation, and a joint PD
  over the seven right Dex3 joints. Every other joint is held at its default, and only the right forearm, the hand
  and the torso collide. Without the torso a policy sank its forearm into the body in half its episodes; with it, the
  next rested the forearm there as often, and penalizing the contact changed nothing, because the IK, not the policy,
  places the elbow (seven joints under a six-row task). The IK now keeps the elbow 16 cm out from the torso's
  midline with steps in its nullspace, which leave the palm nearly where the action puts it.
- **A stiff waist and arm.** The G1's asset gains are the whole-body trackers' (waist roll and pitch kp 28.5, arm
  14.3). With nothing driving the waist, the torso sags 27–28° under the reaching arm and no scripted grasp lifts the
  beaker; at waist kp 300 and arm kp 60, every one of 40 does. A deployment sets the same gains.
- **Reach decides the layout.** With the pelvis and waist fixed the palm reaches about 35–45 cm forward, so the bench
  is a slab with knee room under it, 20 cm from the pelvis, and the beaker starts anywhere in 12 × 30 cm in front of
  the right shoulder. The arm starts raised behind the bench edge, clear of the beaker, and a quarter of the episodes
  start instead with the open hand around the beaker (`StartAtPregrasp`).
- **A grasp is contact.** `OpposedGrasp` is a metric that holds while the thumb and a finger both press the beaker;
  it is rewarded and gates the lift-height reward. `GraspClosure` pays the fingers for closing at the beaker. The
  distance-based approach terms are worth at most 5 a step against up to 45 for a grasp and lift: at full weight, a
  first version's policy lay the open hand around the beaker's foot and collected them without ever closing. A
  thumb-index pinch counts as a grasp. A bonus for the middle finger pressing too (`DigitsPressing`, now only
  logged) got a policy that rested that finger on the beaker at 2 N against 20 N on the thumb and the index, held
  the beaker 4° more tipped, and lost more heavy, slippery beakers; its predecessor's pinches held as well as its
  three-digit grasps.
- **Lift is clearance.** The lift-height reward pays how far the beaker's bottom clears the bench
  (`cylinder_clearance`), not its centre's height: paid for the centre, a policy gripped the beaker, tipped it 20°
  onto its rim (the centre rises 0.9 cm) and held it there all episode without lifting.
- **The beaker must stay upright.** An `object_tilt` metric (`1 − cos` of the tilt) is penalized once the beaker is
  gripped or off the bench, and ends the episode past 45° wherever it is.
- **Little creep, slow carrying.** At Genesis' default `impratio` (1, with the pyramidal friction cone) the soft
  friction constraints let a held beaker turn in the hand at about 4°/s under a steady 46 N grip, and a policy
  learns to chase it with its wrist; the task sets `impratio=10`, at which it turns 0.45°/s. Stiffer friction
  stops it (0.06°/s at 100), but makes the contact solver's blow-ups far worse (below). The noslip solver stops it
  too but cuts training throughput 4.5×. The IK's per-step joint clamp (0.015 rad, 0.75 rad/s) sets the speed —
  the palm reaches at a median 0.19 m/s — and the held beaker's angular velocity is penalized.
- **Solver blow-ups stay out of the returns.** A hand striking the beaker can make the contact solver fling the
  beaker and the arm: under an early policy's exploration, arm joints reached 7000 rad/s at `impratio=100`, 140 at
  10. An episode whose beaker is flung (past 3 m/s or 50 rad/s) ends, and the beaker-spin and arm-speed penalties
  are capped; uncapped, their squares took a training run's value loss to 3e18 and stalled it.
- **Friction and mass are redrawn every episode.** The hand's friction (0.8–2.0, against the beaker's 0.5) is one
  coefficient for the whole hand (`RandomizeSharedFriction`) and the beaker's mass is 0.12–0.30 kg. Drawn once per
  hand link, the hand was almost never slippery all over: a policy trained that way held its beaker in 99 % of
  episodes with every link at 1.0 and 57 % at 0.8.
- **The beaker is moved before the grasp.** `DisplaceRestingObject` teleports a resting beaker the hand has not
  reached yet to a fresh spot, at random intervals, so the policy re-targets a beaker someone moved.
- **State-based, deployment-shaped observations.** The policy sees the arm and hand encoders, forward kinematics of
  them, the goal, its last action and a noised beaker pose with a 3-frame history (what a perception module would
  supply); the critic also sees the fingertip-to-beaker offsets and the beaker's velocity.

## G1 beaker pour

`manipulation_g1_beaker_pour` goes on from the beaker lift: the same welded G1 picks a beaker of eight glass beads (a
stand-in for a liquid) off the bench and pours them into a second beaker beside it. Its actor observes what the
lift's does and acts the same way, so it trains on from a lift checkpoint, widened first for its critic's extra
inputs:

```bash
uv run python -m eden.tasks.manipulation.g1_beaker_pour.warm_start \
    logs/manipulation_g1_beaker_lift/<run>/model_<iter>.pt pour_init.pt
eden train --task manipulation_g1_beaker_pour --checkpoint pour_init.pt
```

- **Hollow beakers, rigid beads.** Both beakers are open cylinders of 16 wall panels (4 mm) on a 6 mm bottom
  (`HollowBeakerOptions`, generated MJCF), the lift's beaker's size. The beads (`BeadsOptions`) are eight free 8 mm
  spheres of 6 g in one entity, stacked in the source at every reset (`FillBeads`); they leave it past about 100° of
  tilt. Eight, not more: each bead adds six DOFs to every constraint solve, and at 4096 envs 20 beads of 6 mm
  trained at 1400 env-steps/s, eight at 10500 (the lift runs at about 45000).
- **Everything that touches collides.** Beads with each other, both beakers, the bench and the hand; the beakers
  with each other, the hand and the bench. The lift's stiff hand-beaker contact parameters cover the target and the
  beads too. `max_collision_pairs=30` leaves room for 150 contacts and 360 broad-phase pairs an env, about four times
  the most measured (39 and 61) while pouring, spilling and acting with noised actions in 768 envs; a count past the
  room halts the run rather than dropping contacts. In those envs no bead's centre ever entered a beaker's glass,
  and the deepest transient penetrations were beads striking the bench (3.5 mm) or the target (3 mm).
- **The hand pours outward, supinating.** The target starts on the right (y −0.34 to −0.26), the source nearer the
  middle (y −0.14 to −0.04). Turned palm up, the beaker lies in the hand, and a lift policy's grasp held it to 130° in
  a scripted pour; turned palm down to pour inward, the beaker hangs under the hand, and it was dropped by 90° in
  nearly every env.
- **The goal is where the beaker's centre is while it pours.** `AnchoredT3Command` draws it 4–6 cm from the
  target's axis toward the source and 12–14 cm above the target's centre, and the actor, which never observes the
  target, finds it at a fixed offset from the goal. A lift policy holds the beaker's centre on its goal and, tipped,
  turns it about that centre, so from there the lip stays over the target's mouth from upright to about 120°, past
  where the beads pour. Placed
  11–14 cm off instead (where the swing of a beaker tipped about the hand would bring its lip over the target), the
  goal fought the lift policy: it pulled the swinging beaker back, the lip never neared the target, and biasing the
  starting policy's wrist toward the target lost return over an episode, where from the pour position it gains.
  The goal terms (`UprightedGoalDistance`) measure the beaker turned back upright about the palm, the IK's control
  point, so tipping it about the hand neither costs nor pays; a first version widened the deadband with the tilt
  instead, and its policy tipped the beaker 40° sideways for the slack. Once the beaker pours (tipped toward the
  target past 20°, its lip within 6 cm of over it and above its rim), the goal terms pay in full: supinating that
  far takes the G1's elbow and wrist to their limits, where the palm sags off the goal.
- **What pays.** The share of beads in the target pays 30 a step and the share on the bench costs 30; past half
  spilled, or with the target knocked over, the episode ends. `PourTilt` pays the tilt toward the target
  (`tilt_toward`: none across or away), in proportion to its angle, while the source is held and its pour lip (the
  lowest point of its rim) is above the target's rim, weighted by how near the lip is to over the target's mouth
  (σ 10 cm: at the goal the lip is 9–13 cm off), up to 60 a step. Leaning the held beaker across the line to the
  target or away from it costs 20 sin² of the lean (`HeldTiltOffTarget`, the lift's upright cost with the pour's
  direction left free). Paid on the plain tilt, a beaker tipped across earned nearly what one tipped toward the
  target did, and a policy tipped it 20–30° sideways for good; with the direction but no cost on leaning and the
  shaping at 25, a policy still drifted it sideways, and tipping 20° toward the target paid too little (1.5 a step)
  to steer it; at σ 4 cm the term paid nothing short of 60°. Beads leave the lip rolling, so they drop in best with the lip about 1.5 cm short of the target's axis:
  held over the axis, half of them hit the far rim. The lift's tilt termination and its settle-at-goal cost are
  gone, the beaker's angular velocity costs a fifth as much, and the lift-height reward is capped at 10 cm of
  clearance, which a beaker tipped at the goal keeps.
- **The critic counts the beads.** It gets the lift critic's inputs and, after them, the shares in the source, in
  the target and spilled, and the beads' centroid from the target's mouth. `warm_start` appends zero weights for
  those to a lift checkpoint's critic, with the identity in its observation normalizer and zero Adam moments, so the
  critic starts out computing what the lift's did. The actor is the lift's unchanged.
- **Every episode starts from the bench.** No beaker is moved mid-episode, there is no pre-grasp start (it puts the
  source at its home, a beaker's width from the target), and the goal-range curriculum is gone (the goal follows the
  target). The smoothness costs sit at the lift curriculum's final weights; episodes last 10 s.

## In-hand reorientation

A port of wuji-mjlab's `reorient` task: a palm-up, fixed-base 20-DOF Wuji Hand 1 holds a 54 mm cube and rotates it to
a target orientation given in the wrist-tag frame, holds it there, then chases the next target, without dropping it.

![The hand rotating the cube toward the goal orientation, shown by the floating twin above the palm](../assets/reorient/reorient_strip.png)

```bash
eden train --task reorient --hand right --num_envs 4096 --logger wandb
eden inference --task reorient --checkpoint logs/reorient/<run>/model_<iter>.pt --rec
```

`--hand` (`right`, default, or `left`) swaps the chirality; the grasp keyframe is mirrored for the left hand.

What to know before changing it (the module docstring of `eden/tasks/reorient/config.py` has the full list):

- **The goal is a state machine, not a timer.** Reaching a goal opens a success window that pays an escalating hold
  reward before the next goal is drawn. The success state is measured in `InHandReorientCommand.post_physics_refresh`
  and only the goal switch is published in `_update_command`, following the hook rule in
  [Writing terms](../guides/terms.md).
- **Drop detection is one metric term** that the reward, the termination and the critic observation all read.
- **The goal is drawn**: a kinematic twin of the cube, invisible to the physics, floats over the palm at the target
  orientation in a recorded rollout.
- **The policy group is reproducible on hardware**: joint angles, joint tracking error and the cube pose relative to
  the wrist tag. Everything privileged is critic-only.
- **The hand is soft.** Its actuator table is the manufacturer's (kp about 0.2–0.7 N·m/rad), and blanket gains break
  it.

`docs/assets/reorient/reorient.mp4` is a recording of a trained policy. It was trained before the success-window fix
above, so reproducing its numbers needs a retrain.

## Piano

Eden's take on RoboPianist's `PianoWithShadowHands`, with two Wuji Hand 2 (beta2) hands on a simulated 88-key
keyboard. The policy sees the key states of the current control step and the next ten (half a second at 20 Hz
control over 200 Hz physics), and is paid for pressing the keys that are due, and nothing else, at the moment they are
due. Each hand is a free body whose 6-axis wrist pose is driven by `FreeBasePoseController` with its weight carried,
so each hand has 26 action axes (20 finger joints and 6 wrist axes).

![Two Wuji hands at the piano keyboard](../assets/readme/piano.webp){ width="60%" }

```bash
eden train --task piano --song twinkle_twinkle
eden train --task piano --song chpn-p7.mid          # a MIDI file, by name if it is in the scores directory
eden inference --task piano --checkpoint logs/piano/<run>/model_<iter>.pt --rec
```

`--song` takes a built-in song (`five_finger_scale`, `ode_to_joy`, `twinkle_twinkle`, the default) or a `.mid` file:
its path, or its file name alone if the file is in the scores directory `~/.cache/eden/scores` (under
`$XDG_CACHE_HOME` when that is set). The file-name form keeps the saved config portable. Where the MIDI file says
which hand plays what, each hand is placed over its own part. The episode is as long as the score.

The task itself is silent. To hear a policy, add a `PianoAudio` sensor and a microphone at evaluation time; the
example does that:

```bash
uv run python examples/projects/manipulation/piano/main.py play --song twinkle_twinkle --vis --play
uv run python examples/projects/manipulation/piano/main.py play --song twinkle_twinkle --record -o /tmp/twinkle.mp4
```

The module docstring of `eden/tasks/piano/config.py` lists every difference from RoboPianist (the free-flying hands,
actions as offsets from a rest pose, a softer key-press margin, a lighter energy penalty, the rewards dropped and
added, and annotation-free fingering).

## OmniReset

UWLab's OmniReset stage-1 task (`OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0`): a UR5e with a Robotiq 2F-85
assembles an object pair, each episode starting from one of four recorded reset-state families drawn with equal
probability, with UWLab's scene, observations, relative Cartesian OSC controller, rewards, startup randomization and
PPO recipe. The recorded states are downloaded from UWLab's Hugging Face dataset on first use.

```bash
eden train --task omnireset
```

| `--robot` | Rig |
|---|---|
| `uwlab` (default) | UWLab's UR5e + 2F-85 (`OmniResetUR5e2f85`), the rig the released expert drives. |
| `ur5e_2f85` | Eden's own `ur5e_2f85` composite, for training a policy of your own. Run `python -m eden.tasks.omnireset.retarget --pair peg` once first, to retarget UWLab's states onto it. |

UWLab evaluates with every episode from the `ObjectAnywhereEEAnywhere` family. To evaluate a checkpoint that way,
override the reset weights:

```bash
eden inference --task omnireset --checkpoint logs/omnireset/<run>/model_<iter>.pt --num_envs 256 --episodes 256 \
    --override "event_options: {reset_states: {category_weights: {ObjectAnywhereEEAnywhere: 1.0,
    ObjectRestingEEGrasped: 0.0, ObjectAnywhereEEGrasped: 0.0, ObjectPartiallyAssembledEEGrasped: 0.0}}}"
```

The released expert runs with `uv run python examples/policies/omnireset/main.py`. Genesis combines contact friction
as `max` where PhysX averages, so only the insertive object's friction is randomized (0.9–1.6). The config's
docstring reports the released expert at 94 % success from `ObjectAnywhereEEAnywhere` over that range, and 87 %
over UWLab's 0.65–1.3 averaged span.

## FlashDexRetarget

The RL half of FlashDexRetarget on Genesis. One policy, trained across every clip of a retargeted dataset, drives one
or two robot hands (6 actuated forearm joints plus fingers each) so the objects follow their recorded trajectories
while the hands follow the human (MANO) wrists and fingertips. A PD wrench pulls each object toward its reference at
first and is withdrawn as the policy improves (`FDRAssistCurriculum`). The clips a policy completes are physically
valid robot hand-object trajectories — the retargeting's real output.

The task trains on a `motion.pt` that `eden retarget flashdex` builds from a released hand-object dataset:

```bash
# 1. Build the dataset (needs the flashdexretarget extra and the licence-gated MANO models).
eden retarget flashdex import --source taco --raw /data/taco --mano-dir /data/mano_v1_2/models --dataset /data/taco_fdr
eden retarget flashdex kinematic --dataset /data/taco_fdr --robot xhand

# 2. Train. The robot is the one motion.pt was retargeted to.
eden train --task flashdexretarget --motion /data/taco_fdr/xhand/motion.pt
eden train --task flashdexretarget --motion /data/taco_fdr/xhand/motion.pt --learner flashsac

# 3. Roll out with the assist off; prints per-episode completion and tracking errors.
eden inference --task flashdexretarget --checkpoint logs/flashdexretarget/<run>/model_<iter>.pt
```

| Flag | Default | Meaning |
|---|---|---|
| `--motion` | required | The `<dataset>/<robot>/motion.pt` that `eden retarget flashdex kinematic` wrote, next to its object URDFs. |
| `--robot` | the motion's own | `xhand` or `sharpa`. |
| `--learner` | `ppo` | `ppo` (rsl_rl PPO) or `flashsac` (FlashDexRetarget's off-policy learner). |

Install what each step needs:

=== "pip"

    ```bash
    pip install "eden-world[flashdexretarget]"   # the dataset pipeline (eden retarget flashdex)
    pip install "eden-world[flashsac]" "rsl-rl-flashsac @ git+https://github.com/Holiday-Robot/rsl_rl_flashsac@9822b1b7e0b5cce45e7f79d6b3e2c169fa042faa"
    ```

=== "uv (checkout)"

    ```bash
    uv sync --inexact --extra flashdexretarget
    uv sync --inexact --extra flashsac --group flashsac-git
    ```

!!! warning "A held-out evaluation cannot use `--motion` on the run's checkpoint"
    `eden inference` replays the run's saved `config.json`, and a task flag such as `--motion` cannot apply to a saved
    config. To roll a policy out on other clips, copy the checkpoint out of its run directory first, so no
    `config.json` sits beside it:

    ```bash
    cp logs/flashdexretarget/<run>/model_<iter>.pt /tmp/policy.pt
    eden inference --task flashdexretarget --motion /data/held_out/xhand/motion.pt --learner <learner> \
        --checkpoint /tmp/policy.pt --num_envs 512 --episodes 2048
    ```

With FlashSAC the replay buffer holds whole observations in float16: 50M rows with one XHand need about 154 GB of
GPU memory, so set `runner_options.algorithm.replay_buffer_size` for your GPU (the task README gives 10M rows,
31 GB, as fitting a 48 GB card). `eden/tasks/flashdexretarget/README.md` lists every piece of the port and every
difference from the original. Building its dataset (`eden retarget flashdex`) is covered on
[Retargeting](../guides/retargeting.md).

## See also

- [Task index](index.md) — every task, and the ported benchmark pages.
- [Robots](../guides/robots.md) — the arm-and-gripper composites (`Droid`, `ur5e_2f85`, `FrankaXHand`, ...).
- [Actions](../guides/actions.md) — differential IK, binary grippers, OSC and free-base pose control.
- [Sensors and rendering](../guides/sensors-and-rendering.md) — cameras, the batch renderer, `PianoAudio`.
- [Motion-tracking tasks](motion-tracking.md) — whole-body tracking, including the hand-object `contrack`.
