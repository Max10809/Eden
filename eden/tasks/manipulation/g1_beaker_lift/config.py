"""G1 beaker lift (``manipulation_g1_beaker_lift``): a standing Unitree G1 picks a beaker off a lab bench with its
right Dex3 hand, keeps it upright, and holds it at a 3D goal.

The humanoid counterpart of :class:`~eden.tasks.manipulation.dex_lift.config.ManipulationDexLiftConfig`
(``manipulation_dex_lift``): its goal tracking, goal-range curriculum, staged smoothness curricula and contact
stiffening, on a :class:`~eden.options.robots.G1Dofs29Dex3` instead of a table-mounted Franka + XHand. What changes:

* **The pelvis is welded in place.** Balance is out of scope: the task is the upper body at a bench, and a deployment
  stands the robot with a lower-body controller. The G1 MJCF's base carries a free joint, which ``is_fixed_base``
  cannot pin (Eden refuses the mismatch), so the robot is welded to ``pelvis_mount``, an invisible, non-colliding
  fixed box at its standing pelvis pose. Welding it to the ground plane instead fails at build, because Genesis then
  voxelizes the plane as part of the robot's own collision tree.
* **Only the right arm and hand act.** 6-DOF differential IK on the seven right-arm joints, at a control point in
  the palm, and joint PD on the seven right Dex3 joints: a 13-D action. The IK deltas accumulate on the last joint
  target (:class:`AccumulatedDifferentialIKController`): the G1 arm has no gravity compensation, so deltas on the
  measured joints would let it sink onto the bench under a zero action. Every other joint is held at its default by
  :class:`NullJointAction`, which takes no action columns. Only the right forearm, the hand and the torso collide:
  without the torso, a policy tucked its elbow into the body in 15 % of its episodes, and touching it costs.
* **The bench sets the reach.** With the pelvis and waist fixed, the palm reaches about 35 cm in front of the pelvis
  at 0.80 m and 40-45 cm at 0.90-1.05 m (the shoulder is at 1.075 m). The bench top is therefore 0.85 m (lab benches
  are about 0.9 m), and it is a slab with knee room under it, so its front edge can sit 20 cm from the pelvis. The
  beaker starts anywhere in 12 x 30 cm of bench in front of the right shoulder; the goal is 10-25 cm above the bench
  over the same area.
* **The arm starts ready.** The asset's default arm hangs below the bench top. The right arm's default is a ready
  pose instead: the hand raised behind the bench edge, clear of everywhere the beaker can be (a hand above the beaker
  area sags onto it under gravity and tips it). The arm starts every episode there.
* **The waist and the right arm are stiff.** The asset's gains are the whole-body trackers' (waist roll and pitch
  kp 28.5, arm kp 14.3), and with nothing driving the waist the torso sagged 27-28 deg under the reaching arm: no
  scripted grasp could lift the beaker. At kp 300 on the waist and 60 on the arm, every one of 40 could.
* **A grasp is contact, and it is what pays.** The first version kept ``manipulation_dex_lift``'s distance-based
  approach rewards at full weight, and its policy lay the open hand around the beaker's foot, collecting them for
  good without ever closing. Now :class:`OpposedGrasp` (thumb and fingers both pressing the beaker) is a rewarded
  metric that gates the lift-height reward, :class:`GraspClosure` pays the fingers for closing at the beaker, the
  approach terms are worth at most 5 a step against up to 45 for a grasp and lift, and pressing the bench costs.
  :class:`DigitsPressing` pays a little more when the middle finger presses too: a thumb-index pinch, which the
  fourth version's policy used, lets the beaker turn about its two contacts.
* **The beaker barely creeps in the grasp, and it is carried slowly.** Friction constraints are 10x stiffer than
  the normal ones (``impratio``): at the default the soft friction let a held beaker turn in the hand at 4 deg/s,
  and the policy chased it with its wrist; at 10 it turns 0.45 deg/s. The IK's per-step joint clamp is the speed
  limit (0.015 rad, 0.75 rad/s): the palm reaches at a median 0.19 m/s, where the first versions' 0.2 rad gave
  0.61 m/s and grabbed the beaker 0.24 s in. The held beaker's angular velocity costs.
* **The contact solver's blow-ups stay out of the returns.** A hand striking the beaker can make the solver fling
  the beaker and the arm (stiffer friction makes it worse: at ``impratio=100`` arm joints reached 7000 rad/s). An
  episode whose beaker is flung (past 3 m/s or 50 rad/s) ends, and the beaker-spin and arm-speed costs are capped:
  uncapped, the squares of those speeds took the third version's value loss to 3e18 and stalled its training.
* **A quarter of the episodes start at the grasp.** :class:`StartAtPregrasp` puts the open hand around the beaker
  at its home (a pose from which a scripted closure lifts it), so the lift is found early.
* **Lift is clearance.** The lift reward pays how far the beaker's bottom clears the bench
  (:func:`cylinder_clearance`). Paid for its centre's height, the third version's policy gripped the beaker, tipped
  it 20 deg onto its rim, which raises the centre 0.9 cm, and held it there all episode.
* **The beaker must stay upright.** :func:`object_tilt` (``1 - cos`` of the beaker's tilt) is a metric that a
  penalty reads once the beaker is gripped or off the bench (before, the penalty only taught the hand not to
  close; gated on the centre's height, it missed the tip on the rim) and that ends the episode past 45 degrees
  wherever it is: a tipped beaker has spilled.
* **The beaker is moved before it is grasped.** At random intervals :class:`DisplaceRestingObject` teleports the
  beaker to a fresh spot, but only while it rests on the bench and the hand is not at it, so the policy learns to
  re-target a beaker someone moved.
* **The policy observes only what the robot measures,** as in ``manipulation_dex_lift``: arm and Dex3 encoders (the
  Dex3 reports joint velocities, so the policy gets them), forward kinematics of those, the goal, its last action and
  the beaker pose a perception module estimates (noised, three frames of history). The critic also gets the
  fingertip-to-beaker offsets and the beaker's velocity.

The beaker is a solid cylinder the size of a 250 mL beaker (7 cm across, 9.5 cm tall) at 0.18 kg, a glass beaker
with a little liquid in it.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import TYPE_CHECKING

import torch
from genesis.typing import Vec2FType

import eden as en
from eden import EventMode, events, observations, rewards, terminations
from eden.constants import MetricDirection, ReferenceSource
from eden.managers import METRIC_TERM_REGISTRY, REWARD_TERM_REGISTRY, TERMINATION_TERM_REGISTRY
from eden.managers.action_manager import ACTION_TERM_REGISTRY
from eden.managers.event_manager import EVENT_TERM_REGISTRY
from eden.managers.event_manager import EventTerm as EventTermBase
from eden.managers.metric_manager import MetricTerm as MetricTermBase
from eden.managers.modifiers.observations.noise import GaussianNoise, UniformNoise
from eden.managers.reward_manager import RewardTerm as RewardTermBase
from eden.managers.terms.actions.joint_actions import ImplicitPDController, NullJointAction
from eden.managers.terms.actions.task_space_actions import DifferentialIKController
from eden.managers.terms.commands.oracle import UniformT3Command
from eden.managers.terms.curricula.rewards import StageRewardWeightCurriculum
from eden.managers.terms.events import RandomizeFrictionRatio
from eden.managers.terms.events.resets import reset_base_state_uniform
from eden.managers.terms.utils import resolve_link_idx, resolve_metric_manager, soft_dof_pos_violation
from eden.options import ObservationGroupOptions
from eden.options.entities import BoxOptions, CylinderOptions, SphereOptions
from eden.options.learning.rsl_rl import (
    RslRlActorOptions,
    RslRlCriticOptions,
    RslRlGaussianDistributionOptions,
    RslRlOnPolicyRunnerOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.options.materials import KinematicMaterialOptions, RigidMaterialOptions
from eden.options.robots import G1Dofs29Dex3
from eden.options.robots.unitree_g1 import G1_DEX3_RIGHT_DOFS_NAME
from eden.tasks.manipulation import custom_terms
from eden.tasks.manipulation.custom_terms.common import _grasp_point_w
from eden.tasks.manipulation.dex_lift.config import (
    dex_fingertip_to_object_reward,
    dex_fingertips_pos_b,
    dex_fingertips_to_object_obs,
)
from eden.tasks.registry import TASK_REGISTRY
from eden.terms import DoneTerm, EventTerm, MetricTerm, ObsTerm, RewardTerm
from eden.utils.configs import (
    ActionManagerOptions,
    CommandManagerOptions,
    CurriculumManagerOptions,
    EdenRLConfig,
    EnvOptions,
    EventManagerOptions,
    MetricManagerOptions,
    ObservationManagerOptions,
    RewardManagerOptions,
    SceneOptions,
    TerminationManagerOptions,
)
from eden.utils.misc import is_env_mask
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

#: The G1's standing pelvis pose, where ``pelvis_mount`` holds it.
_PELVIS_POS = tuple(G1Dofs29Dex3.model_fields["default_root_pos"].default)

# Bench: a 4 cm slab whose top is at 0.85 m and whose front edge is 20 cm in front of the pelvis, over a cabinet set
# back far enough to leave the knees and feet clear.
_BENCH_TOP = 0.85
_SLAB = 0.04
_EDGE_X = 0.20
_BENCH_DEPTH = 0.60
_BENCH_WIDTH = 1.20
_BENCH_Y = -0.10
_CABINET_X = (0.45, 0.80)

# A 250 mL beaker: 7 cm across, 9.5 cm tall, 0.18 kg.
_BEAKER_RADIUS = 0.035
_BEAKER_HEIGHT = 0.095
_BEAKER_MASS = 0.18
_REST_Z = _BENCH_TOP + _BEAKER_HEIGHT / 2
_BEAKER_HOME = (0.32, -0.18, _REST_Z)
# Where the beaker may start (and is moved to), as offsets from home: x 0.26-0.38, y -0.33 to -0.03.
_PLACE_X = (-0.06, 0.06)
_PLACE_Y = (-0.15, 0.15)
# Beaker and bench friction; the hand's is randomized above it (a contact takes the larger of the two).
_BEAKER_FRICTION = 0.5
#: ``1 - cos(45 deg)``: tipped this far, the beaker has spilled.
_SPILL_TILT = 0.2929

_EPISODE_S = 8.0


# ---------------------------------------------------------------------------
# Robot: the G1 with Dex3 hands, right arm at a ready pose, only the right arm and hand colliding
# ---------------------------------------------------------------------------

_ARM_DOFS = [
    f"right_{joint}_joint"
    for joint in (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    )
]
_HAND_DOFS = list(G1_DEX3_RIGHT_DOFS_NAME)
#: Distal link of each Dex3 digit: their collision-geom centroid is the grasp point the shared lift terms use.
_TIPS = ["right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link"]
#: The palm is merged into the wrist-yaw link (a fixed joint); this point of that link is the palm centre.
_EE_LINK = "right_wrist_yaw_link"
_PALM_IN_WRIST = (0.09, 0.0, 0.0)
#: Every link of the right hand that touches the beaker: the palm's and each phalanx.
_HAND_LINKS = [_EE_LINK] + [
    f"right_hand_{digit}_link"
    for digit in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
]
#: The two sides of an opposed grasp: the thumb, and the two fingers it closes against.
_THUMB_LINKS = ["right_hand_thumb_1_link", "right_hand_thumb_2_link"]
_FINGER_LINKS = [f"right_hand_{digit}_link" for digit in ("index_0", "index_1", "middle_0", "middle_1")]
#: A three-point grasp: each digit by its links.
_DIGITS = {
    "thumb": _THUMB_LINKS,
    "index": ["right_hand_index_0_link", "right_hand_index_1_link"],
    "middle": ["right_hand_middle_0_link", "right_hand_middle_1_link"],
}
#: The right arm's colliding links below the elbow, which can reach the torso.
_FOREARM_AND_HAND = ["right_elbow_link", "right_wrist_.*", "right_hand_.*"]
#: Where the closing joints head when the hand closes on nothing: the thumb's distal joint closes negative, the
#: fingers positive. A grasp on the beaker stops them about two thirds of the way.
_HAND_CLOSED = {
    "right_hand_thumb_2_joint": -1.2,
    "right_hand_index_0_joint": 1.2,
    "right_hand_index_1_joint": 1.2,
    "right_hand_middle_0_joint": 1.2,
    "right_hand_middle_1_joint": 1.2,
}

_WAIST_DOFS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
#: The asset's gains are the motor-inertia heuristic of the whole-body trackers (kp 28.5 on waist roll and pitch,
#: 14.3 on the arm), which leave balance and load to a policy that drives every joint. Here nothing drives the
#: waist, and the upper body is an inverted pendulum on it: with the right arm reaching out, the torso sagged 27 deg
#: in roll and 28 deg in pitch, and the hand could not lift a grasped beaker. A scripted grasp lifted it 10 cm in
#: 40 of 40 hand poses at these gains and in none at the asset's. The robot takes per-joint gains over the SDK.
_WAIST_KP, _WAIST_KD = 300.0, 5.0
_ARM_KP, _ARM_KD = 60.0, 2.0

#: The right arm with the open hand straddling the beaker at its home: wrist z up, palm toward the beaker, the beaker
#: 11 cm along the fingers and 5 cm out from the palm. Solved by IK for that hand pose; from it a scripted closure
#: and a 10 cm rise lift the beaker with 0.4 deg of tilt.
_PREGRASP_ARM = {
    "right_shoulder_pitch_joint": -0.3348,
    "right_shoulder_roll_joint": -0.3934,
    "right_shoulder_yaw_joint": -0.2540,
    "right_elbow_joint": 0.5495,
    "right_wrist_roll_joint": 0.2918,
    "right_wrist_pitch_joint": -0.1135,
    "right_wrist_yaw_joint": 0.8468,
}

#: Hand raised behind the bench edge, clear of every place the beaker can be. Elbow 0 is the forearm level with the
#: upper arm down, and negative bends it up. The hand's collision geoms span x 0.01-0.215 m (the beaker's nearest
#: edge can be at 0.225 m) and start at z 1.02 m (its top is at 0.945 m). The margin is for the 2.6 cm the arm's PD
#: sags under gravity and for the reset jitter: a hand over the beaker area (elbow -0.7 or -1.25) lands on a beaker
#: near the bench edge and tips it.
_RIGHT_ARM_READY = {
    "right_shoulder_pitch_joint": 0.4,
    "right_shoulder_roll_joint": -0.15,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": -1.5,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.3,
    "right_wrist_yaw_joint": 0.0,
}


def _per_joint(field: str, overrides: dict[str, float]) -> dict[str, float]:
    """G1Dofs29Dex3's ``field`` with one key per joint, then ``overrides`` on top.

    The asset keys joints by pattern (``*_shoulder_*_joint``), and a joint two keys match is an error, so an override
    of one joint is only possible once every pattern is resolved to the joints it names.
    """
    patterns = G1Dofs29Dex3.model_fields[field].default
    per_joint = {}
    for name in G1Dofs29Dex3.model_fields["dofs_name"].default:
        matched = [value for pattern, value in patterns.items() if fnmatchcase(name, pattern)]
        if matched:
            per_joint[name] = matched[0]
    return {**per_joint, **overrides}


_ROBOT = G1Dofs29Dex3(
    # The MJCF's base is a free joint, so the flag must say so; the weld to ``pelvis_mount`` pins it.
    is_fixed_base=False,
    default_dofs_pos=_per_joint("default_dofs_pos", _RIGHT_ARM_READY),
    default_dofs_kp=_per_joint(
        "default_dofs_kp", {**dict.fromkeys(_WAIST_DOFS, _WAIST_KP), **dict.fromkeys(_ARM_DOFS, _ARM_KP)}
    ),
    default_dofs_kd=_per_joint(
        "default_dofs_kd", {**dict.fromkeys(_WAIST_DOFS, _WAIST_KD), **dict.fromkeys(_ARM_DOFS, _ARM_KD)}
    ),
    # The torso is what the right forearm reaches: without it, a policy tucked the elbow into the body in 15 % of
    # its episodes, carrying the beaker to a goal near the body. Nothing else reaches the bench, the beaker or the
    # arm, and every pair dropped is collision work saved (the torso costs 10 % of the throughput).
    collision_link_patterns=[*_FOREARM_AND_HAND, "torso_link"],
)
_HELD_DOFS = [name for name in _ROBOT.dofs_name if name not in _ARM_DOFS + _HAND_DOFS]


# ---------------------------------------------------------------------------
# Task terms
# ---------------------------------------------------------------------------


@METRIC_TERM_REGISTRY.register(direction=MetricDirection.LIB, contributes_to_success=False)
def object_tilt(env: EnvBase, *, object_name: str) -> torch.Tensor:
    """``1 - cos`` of the angle between the object's z axis and the world's: 0 upright, 1 on its side, 2 upside down.

    The z-z entry of a wxyz quaternion's rotation matrix is ``1 - 2 (x^2 + y^2)``.
    """
    quat = env.get_entity(object_name).get_quat()
    return 2.0 * (quat[:, 1].square() + quat[:, 2].square())


@METRIC_TERM_REGISTRY.register(direction=MetricDirection.HIB, contributes_to_success=False)
def cylinder_clearance(
    env: EnvBase, *, object_name: str, radius: float, height: float, support_z: float
) -> torch.Tensor:
    """How far the lowest point of a cylinder (its axis along its local z) is above ``support_z``, in m.

    Tilted by ``theta``, that point is ``height / 2 * |cos theta| + radius * sin theta`` below the centre. So a
    beaker tipped onto its rim on the bench reads 0, where its centre has risen: 0.9 cm at 20 deg, 1.15 cm at most
    (at 36 deg).
    """
    entity = env.get_entity(object_name)
    quat = entity.get_quat()
    cos_tilt = 1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square())
    sin_tilt = (1.0 - cos_tilt.square()).clamp(min=0.0).sqrt()
    return entity.get_pos()[:, 2] - height / 2 * cos_tilt.abs() - radius * sin_tilt - support_z


@TERMINATION_TERM_REGISTRY.register()
def object_ejected(env: EnvBase, *, object_name: str, max_lin_vel: float, max_ang_vel: float) -> torch.Tensor:
    """The object moves faster than ``max_lin_vel`` m/s or spins faster than ``max_ang_vel`` rad/s.

    No hand carries or knocks a beaker that fast: a hand striking it lets the contact solver fling it (to 16 m/s and
    490 rad/s in 4096 envs of random actions, and 129 m/s at ``impratio=100``), and what follows is not physics to
    learn from.
    """
    entity = env.get_entity(object_name)
    return (entity.get_vel().norm(dim=-1) > max_lin_vel) | (entity.get_ang().norm(dim=-1) > max_ang_vel)


@REWARD_TERM_REGISTRY.register()
class NamedDofsPosLimits(RewardTermBase):
    """Soft-limit violation summed over ``dofs_name`` alone.

    ``dofs_pos_limits`` sums every DOF of the entity, which on a G1 with Dex3 hands includes fingers that rest open on
    their hard limit, outside the 90 % soft band, and would be penalized for it at every step.
    """

    entity_name: str = "robot"
    dofs_name: list[str] = []

    def build(self) -> None:
        entity = self.resolve_entity(self.entity_name)
        if not self.dofs_name:
            raise ValueError(
                f"{term_subject(self)}: dofs_name is empty, so there is nothing to penalize. Name the joints, e.g. "
                f"dofs_name={list(entity.dofs_name[:2])}; for every joint of the entity use dofs_pos_limits."
            )
        dofs = entity.find_named_dofs(self.dofs_name)
        self._dofs_idx_local = dofs.dofs_idx_local
        self._soft_limits = entity.get_soft_dofs_pos_limits(dofs)  # (n_envs, n_dofs, 2), a copy

    def compute(self) -> torch.Tensor:
        dofs_pos = self.resolve_entity(self.entity_name).get_dofs_pos(dofs_idx_local=self._dofs_idx_local)
        return soft_dof_pos_violation(dofs_pos, self._soft_limits).sum(dim=-1)


@REWARD_TERM_REGISTRY.register()
class NamedDofsVelLimits(RewardTermBase):
    """Quadratic hinge on ``|joint velocity| - max_vel`` over ``dofs_name`` alone, each speed capped at ``cap``
    (negative weight to penalize).

    ``dofs_vel_limits`` over the whole G1 would also slow the Dex3 fingers, which close faster than an arm moves. The
    cap is above anything the arm does (12 rad/s under a policy's exploration noise): faster is a contact-solver
    blow-up, whose square would swamp the returns.
    """

    entity_name: str = "robot"
    dofs_name: list[str] = []
    max_vel: float = 0.5
    cap: float = float("inf")

    def build(self) -> None:
        entity = self.resolve_entity(self.entity_name)
        if not self.dofs_name:
            raise ValueError(
                f"{term_subject(self)}: dofs_name is empty, so there is nothing to penalize. Name the joints, e.g. "
                f"dofs_name={list(entity.dofs_name[:2])}; for every joint of the entity use dofs_vel_limits."
            )
        if not 0.0 < self.max_vel < self.cap:
            raise ValueError(
                f"{term_subject(self)}: need 0 < max_vel < cap rad/s, got max_vel={self.max_vel}, cap={self.cap}."
            )
        self._dofs_idx_local = entity.find_named_dofs(self.dofs_name).dofs_idx_local

    def compute(self) -> torch.Tensor:
        dofs_vel = self.resolve_entity(self.entity_name).get_dofs_vel(dofs_idx_local=self._dofs_idx_local)
        return (dofs_vel.abs().clamp(max=self.cap) - self.max_vel).clamp_min(0.0).square().sum(dim=-1)


@ACTION_TERM_REGISTRY.register()
class AccumulatedDifferentialIKController(DifferentialIKController):
    """Differential IK whose joint deltas accumulate on the last commanded target, not on the measured joints.

    :class:`DifferentialIKController` adds each step's IK delta to the *measured* joint positions. An arm whose PD
    carries no gravity compensation (the G1's, in simulation and on the robot) then sags under a zero action: every
    step re-targets wherever gravity has pulled it, and the G1's hand drops 3 cm in 0.2 s, onto whatever is below it.
    Here the delta is added to the previous target, so a zero action holds the arm where it was commanded.

    The target stays within ``max_target_lead`` (rad) of the measured joints, so a hand pressed against the bench
    cannot wind up a target it never reaches, and inside the joint limits. It restarts from the measured joints on
    the first step after a reset.
    """

    max_target_lead: float = 0.3

    def build(self) -> None:
        super().build()
        if self.max_target_lead <= 0.0:
            raise ValueError(
                f"{term_subject(self)}: max_target_lead must be > 0 rad, got {self.max_target_lead}; it bounds how "
                f"far the accumulated target may lead the measured joints."
            )
        lower, upper = self.entity.get_dofs_limit(dofs_idx_local=self.dofs_idx_local)
        self._lower, self._upper = lower.clone(), upper.clone()
        self._target = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
        self._from_measured = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def compute(self, actions: torch.Tensor) -> None:
        super().compute(actions)  # _processed_action = measured + IK delta
        measured = self.entity.get_dofs_pos(dofs_idx_local=self.genesis_dofs_idx)
        base = torch.where(self._from_measured.unsqueeze(-1), measured, self._target)
        target = base + (self._processed_action - measured)
        target = torch.clamp(target, measured - self.max_target_lead, measured + self.max_target_lead)
        self._target.copy_(torch.clamp(target, self._lower, self._upper))
        self._from_measured.fill_(False)
        self._processed_action.copy_(self._target)

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        super().reset(envs_idx)
        self._from_measured[slice(None) if envs_idx is None else envs_idx] = True


@EVENT_TERM_REGISTRY.register()
class DisplaceRestingObject(EventTermBase):
    """Move a resting object the hand has not reached yet to a fresh spot, as a person moving it would.

    When it fires for an env, and with probability ``probability``, it redraws the object's pose like
    ``reset_base_state_uniform`` (``pos_*_range`` and ``yaw_range`` around its default pose), but only where the
    object still rests within ``rest_tolerance`` of ``rest_z`` and the hand's grasp point (the centroid of
    ``grasp_links``' collision geoms) is farther than ``min_hand_distance`` from it. A held or nudged object is left
    alone.
    """

    mode: EventMode = EventMode.INTERVAL
    object_name: str = "object"
    hand_name: str = "robot"
    grasp_links: list[str] = []
    rest_z: float = 0.0
    rest_tolerance: float = 0.01
    min_hand_distance: float = 0.10
    probability: float = 0.5
    pos_x_range: Vec2FType = (0.0, 0.0)
    pos_y_range: Vec2FType = (0.0, 0.0)
    yaw_range: Vec2FType = (-3.14, 3.14)

    def build(self) -> None:
        self.resolve_entity(self.object_name)
        hand = self.resolve_entity(self.hand_name)
        if not self.grasp_links:
            raise ValueError(
                f"{term_subject(self, self.hand_name)}: grasp_links is empty, so there is no grasp point to keep the "
                f"object away from. Name the fingertip links, e.g. grasp_links={list(hand.links_name[-2:])}."
            )
        resolve_link_idx(hand, self.grasp_links, self.device, term=type(self).__name__)  # a bad name raises here
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(
                f"{term_subject(self, self.object_name)}: probability must be in [0, 1], got {self.probability}."
            )

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        obj_pos = self.resolve_entity(self.object_name).get_pos()
        grasp = _grasp_point_w(self._env, self.hand_name, self.grasp_links)
        resting = (obj_pos[:, 2] - self.rest_z).abs() < self.rest_tolerance
        untouched = torch.norm(grasp - obj_pos, dim=-1) > self.min_hand_distance
        drawn = torch.rand(self.num_envs, device=self.device) < self.probability
        move = resting & untouched & drawn & _env_mask(envs_idx, self.num_envs, self.device)
        # A bool mask with every row is the setters' sync-free form; an all-False mask writes nothing.
        reset_base_state_uniform(
            self._env,
            move,
            entity_name=self.object_name,
            pos_x_range=self.pos_x_range,
            pos_y_range=self.pos_y_range,
            pos_z_range=(0.0, 0.0),
            yaw_range=self.yaw_range,
        )


def _env_mask(envs_idx: slice | torch.Tensor | None, num_envs: int, device) -> torch.Tensor:
    """The envs an event fires for, as a ``(num_envs,)`` bool mask."""
    if envs_idx is not None and is_env_mask(envs_idx):
        return envs_idx
    mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
    mask[slice(None) if envs_idx is None else envs_idx] = True
    return mask


@EVENT_TERM_REGISTRY.register()
class StartAtPregrasp(EventTermBase):
    """Start a share of the episodes with the object already inside the open hand.

    With probability ``probability`` per resetting env, it writes ``dofs_pos`` (joint name -> position) to
    ``entity_name`` and redraws the object around its default pose (``pos_jitter`` m in x and y, ``yaw_range``). Run
    it after the events that place the object and the robot (a higher ``priority``), since it overwrites both.
    Episodes that start at the grasp reach the lift often enough for the policy to find out what it pays.
    """

    mode: EventMode = EventMode.RESET
    probability: float = 0.25
    entity_name: str = "robot"
    dofs_pos: dict[str, float] = {}
    object_name: str = "object"
    pos_jitter: float = 0.01
    yaw_range: Vec2FType = (-3.14, 3.14)

    def build(self) -> None:
        entity = self.resolve_entity(self.entity_name)
        self.resolve_entity(self.object_name)
        if not self.dofs_pos:
            raise ValueError(
                f"{term_subject(self)}: dofs_pos is empty, so there is no pre-grasp pose to start from. Give the "
                f"arm's joints, e.g. dofs_pos={{'{entity.dofs_name[0]}': 0.0, ...}}."
            )
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"{term_subject(self)}: probability must be in [0, 1], got {self.probability}.")
        dofs = entity.find_named_dofs(list(self.dofs_pos))
        self._dofs_idx_local = dofs.dofs_idx_local
        self._pregrasp = torch.tensor([list(self.dofs_pos.values())], device=self.device)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        entity = self.resolve_entity(self.entity_name)
        drawn = torch.rand(self.num_envs, device=self.device) < self.probability
        start = drawn & _env_mask(envs_idx, self.num_envs, self.device)
        dofs_pos = entity.get_dofs_pos(dofs_idx_local=self._dofs_idx_local)
        dofs_pos = torch.where(start.unsqueeze(-1), self._pregrasp, dofs_pos)
        entity.set_dofs_pos(dofs_pos, dofs_idx_local=self._dofs_idx_local, envs_idx=start)
        jitter = (-self.pos_jitter, self.pos_jitter)
        reset_base_state_uniform(
            self._env,
            start,
            entity_name=self.object_name,
            pos_x_range=jitter,
            pos_y_range=jitter,
            pos_z_range=(0.0, 0.0),
            yaw_range=self.yaw_range,
        )


@METRIC_TERM_REGISTRY.register()
class OpposedGrasp(MetricTermBase):
    """1.0 where both sides of the hand press the object: a ``thumb_links`` link and a ``finger_links`` link, each
    with more than ``force_threshold`` N of contact force from it.

    Read from the contacts between the two entities alone, so the fingers pressing the hand's own palm or the bench
    do not count, and an object pushed by one side only is not held.
    """

    entity_name: str = "robot"
    object_name: str = "object"
    thumb_links: list[str] = []
    finger_links: list[str] = []
    force_threshold: float = 0.5

    def build(self) -> None:
        entity = self.resolve_entity(self.entity_name)
        self.resolve_entity(self.object_name)
        if not self.thumb_links or not self.finger_links:
            raise ValueError(
                f"{term_subject(self)}: thumb_links and finger_links must both name links (got {self.thumb_links} and "
                f"{self.finger_links}); a grasp needs a link on each side, e.g. thumb_links={entity.links_name[-1:]}."
            )
        thumb = resolve_link_idx(entity, self.thumb_links, self.device, term=type(self).__name__)
        fingers = resolve_link_idx(entity, self.finger_links, self.device, term=type(self).__name__)
        self._links = torch.cat([thumb, fingers])
        self._n_thumb = len(thumb)

    def compute(self) -> torch.Tensor:
        entity, obj = self.resolve_entity(self.entity_name), self.resolve_entity(self.object_name)
        force = entity.get_grouped_contacts(obj, links_a_idx_local=self._links)["force_norm_sum"].sum(dim=-1)
        thumb = force[:, : self._n_thumb].sum(dim=-1) > self.force_threshold
        fingers = force[:, self._n_thumb :].sum(dim=-1) > self.force_threshold
        return (thumb & fingers).float()


@METRIC_TERM_REGISTRY.register()
class DigitsPressing(MetricTermBase):
    """1.0 where every digit in ``digits`` (name -> its links) presses the object with more than ``force_threshold`` N.

    With the thumb, index and middle finger it is a three-point grasp. :class:`OpposedGrasp` also holds for a
    thumb-index pinch, about whose two contacts the beaker can turn: in scripted grasps a pinch let it turn 6-10x
    more in the hand.
    """

    entity_name: str = "robot"
    object_name: str = "object"
    digits: dict[str, list[str]] = {}
    force_threshold: float = 0.5

    def build(self) -> None:
        entity = self.resolve_entity(self.entity_name)
        self.resolve_entity(self.object_name)
        if len(self.digits) < 2 or not all(self.digits.values()):
            raise ValueError(
                f"{term_subject(self)}: digits must name at least two digits, each with its links (got "
                f"{self.digits}), e.g. digits={{'thumb': {entity.links_name[-2:]}, 'index': [...]}}."
            )
        groups = [
            resolve_link_idx(entity, links, self.device, term=type(self).__name__) for links in self.digits.values()
        ]
        self._links = torch.cat(groups)
        self._digit_of_link = torch.cat(
            [torch.full((len(group),), i, dtype=torch.long, device=self.device) for i, group in enumerate(groups)]
        )
        self._n_digits = len(groups)

    def compute(self) -> torch.Tensor:
        entity, obj = self.resolve_entity(self.entity_name), self.resolve_entity(self.object_name)
        force = entity.get_grouped_contacts(obj, links_a_idx_local=self._links)["force_norm_sum"].sum(dim=-1)
        per_digit = force.new_zeros(force.shape[0], self._n_digits).index_add_(1, self._digit_of_link, force)
        return (per_digit > self.force_threshold).all(dim=-1).float()


@REWARD_TERM_REGISTRY.register()
class GraspClosure(RewardTermBase):
    """Mean flexion of the hand's closing joints, paid while its grasp point is within ``near_distance`` of the
    object: what teaches a dexterous hand to close, which no distance reward does.

    Each joint in ``closed_dofs_pos`` (name -> closed position) scores 0 at its default and 1 at the closed position,
    clipped to [0, 1]. The grasp point is the centroid of ``grasp_links``' collision geoms.
    """

    entity_name: str = "robot"
    object_name: str = "object"
    grasp_links: list[str] = []
    closed_dofs_pos: dict[str, float] = {}
    near_distance: float = 0.08

    def build(self) -> None:
        entity = self.resolve_entity(self.entity_name)
        self.resolve_entity(self.object_name)
        if not self.closed_dofs_pos or not self.grasp_links:
            raise ValueError(
                f"{term_subject(self)}: closed_dofs_pos and grasp_links must both be set (got {self.closed_dofs_pos} "
                f"and {self.grasp_links}): the joints that close the hand, and the links whose centroid is its grasp point."
            )
        resolve_link_idx(entity, self.grasp_links, self.device, term=type(self).__name__)
        dofs = entity.find_named_dofs(list(self.closed_dofs_pos))
        self._dofs_idx_local = dofs.dofs_idx_local
        self._open = entity.get_default_dofs_pos(dofs)  # (n_envs, n), a copy
        closed = torch.tensor([list(self.closed_dofs_pos.values())], device=self.device)
        self._span = closed - self._open
        if (self._span.abs() < 1e-3).any():
            raise ValueError(
                f"{term_subject(self)}: every closed position must differ from the joint's default; "
                f"closed_dofs_pos={self.closed_dofs_pos}, defaults {self._open[0].tolist()}."
            )

    def compute(self) -> torch.Tensor:
        entity = self.resolve_entity(self.entity_name)
        flexion = ((entity.get_dofs_pos(dofs_idx_local=self._dofs_idx_local) - self._open) / self._span).clamp(0.0, 1.0)
        grasp = _grasp_point_w(self._env, self.entity_name, self.grasp_links)
        near = torch.norm(grasp - self.resolve_entity(self.object_name).get_pos(), dim=-1) < self.near_distance
        return flexion.mean(dim=-1) * near


@REWARD_TERM_REGISTRY.register()
class GraspedLiftHeight(RewardTermBase):
    """The ``height_metric`` metric (how far the object clears its support), clipped to [0, ``cap``], paid only while
    the ``grasp_metric`` metric holds.

    Ungated, any upward motion pays. Gated, but read off the object's centre, it paid the third version's policy for
    gripping the beaker and tipping it 20 deg onto its rim (the centre rises 0.9 cm): it held it there all episode and
    never lifted. A :func:`cylinder_clearance` is 0 on the rim.
    """

    height_metric: str = ""
    grasp_metric: str = ""
    cap: float = 0.25

    def build(self) -> None:
        for name in (self.height_metric, self.grasp_metric):
            resolve_metric_manager(self._env, term=type(self).__name__, metric_name=name)

    def compute(self) -> torch.Tensor:
        metrics = self._env.metric_manager
        return metrics.get_value(self.height_metric).clamp(0.0, self.cap) * metrics.get_value(self.grasp_metric)


def _held(env: EnvBase, *, clearance_metric: str, grasp_metric: str, margin: float) -> torch.Tensor:
    """Gripped, or clear of the support by more than ``margin`` (in flight, or carried in a grip the metric missed)."""
    metrics = env.metric_manager
    return (metrics.get_value(grasp_metric) > 0.5) | (metrics.get_value(clearance_metric) > margin)


@REWARD_TERM_REGISTRY.register()
class HeldObjectTilt(RewardTermBase):
    """The ``tilt_metric`` metric while the object is held: gripped (``grasp_metric``) or more than ``margin`` m clear
    of its support (``clearance_metric``). Negative weight.

    Not before: a closing hand nudges the beaker, and a penalty on that only taught the hand not to close. Not only
    off the bench either: gated on the beaker's height, a beaker gripped and tipped on its rim cost nothing.
    """

    tilt_metric: str = ""
    clearance_metric: str = ""
    grasp_metric: str = ""
    margin: float = 0.01

    def build(self) -> None:
        for name in (self.tilt_metric, self.clearance_metric, self.grasp_metric):
            resolve_metric_manager(self._env, term=type(self).__name__, metric_name=name)

    def compute(self) -> torch.Tensor:
        held = _held(
            self._env, clearance_metric=self.clearance_metric, grasp_metric=self.grasp_metric, margin=self.margin
        )
        return self._env.metric_manager.get_value(self.tilt_metric) * held


@REWARD_TERM_REGISTRY.register()
class HeldObjectAngVel(RewardTermBase):
    """The object's angular speed squared, with the speed capped at ``max_ang_vel`` rad/s, while it is held as
    :class:`HeldObjectTilt` gates it (negative weight): a carried beaker that swings or spins slops its contents
    however upright it ends up.

    The cap is what keeps the returns finite. A hand striking the beaker lets the contact solver fling it at up to
    thousands of rad/s, and uncapped, the third version's returns took the square of that: its value loss reached
    3e18, and the adaptive schedule held the learning rate at its floor for 800 iterations.
    """

    object_name: str = "object"
    clearance_metric: str = ""
    grasp_metric: str = ""
    margin: float = 0.01
    max_ang_vel: float = 3.0

    def build(self) -> None:
        self.resolve_entity(self.object_name)
        for name in (self.clearance_metric, self.grasp_metric):
            resolve_metric_manager(self._env, term=type(self).__name__, metric_name=name)

    def compute(self) -> torch.Tensor:
        held = _held(
            self._env, clearance_metric=self.clearance_metric, grasp_metric=self.grasp_metric, margin=self.margin
        )
        speed = self.resolve_entity(self.object_name).get_ang().norm(dim=-1).clamp(max=self.max_ang_vel)
        return speed.square() * held


@REWARD_TERM_REGISTRY.register()
class ContactForceWith(RewardTermBase):
    """Contact force between ``links_name`` of ``entity_name`` and the ``other_name`` entity, in N, capped at
    ``max_force`` (negative weight to penalize).
    """

    entity_name: str = "robot"
    links_name: list[str] = []
    other_name: str = ""
    max_force: float = 30.0

    def build(self) -> None:
        entity = self.resolve_entity(self.entity_name)
        self.resolve_entity(self.other_name)
        if not self.links_name:
            raise ValueError(
                f"{term_subject(self)}: links_name is empty, so no contact is measured. Name the links whose contact "
                f"with '{self.other_name}' counts, e.g. links_name={entity.links_name[-2:]}."
            )
        self._links = resolve_link_idx(entity, self.links_name, self.device, term=type(self).__name__)

    def compute(self) -> torch.Tensor:
        other = self.resolve_entity(self.other_name)
        contacts = self.resolve_entity(self.entity_name).get_grouped_contacts(other, links_a_idx_local=self._links)
        return contacts["force_norm_sum"].sum(dim=(1, 2)).clamp(max=self.max_force)


@REWARD_TERM_REGISTRY.register()
class LinksTouching(RewardTermBase):
    """1.0 where one of ``links_name`` touches one of ``other_links_name``, both links of ``entity_name``
    (negative weight to penalize).

    Self-contact, which ``get_grouped_contacts`` does not report, so it is read off ``get_contacts`` as a presence
    rather than a force. Both sides must carry collision geoms (``collision_link_patterns``), or nothing touches.
    """

    entity_name: str = "robot"
    links_name: list[str] = []
    other_links_name: list[str] = []

    def build(self) -> None:
        entity = self.resolve_entity(self.entity_name)
        if not self.links_name or not self.other_links_name:
            raise ValueError(
                f"{term_subject(self)}: links_name and other_links_name must both name links (got {self.links_name} "
                f"and {self.other_links_name}), e.g. links_name=['right_elbow_link'], other_links_name=['torso_link']."
            )
        self._links = resolve_link_idx(entity, self.links_name, self.device, term=type(self).__name__)
        self._other_links = resolve_link_idx(
            entity, self.other_links_name, self.device, term=type(self).__name__, field="other_links_name"
        )

    def compute(self) -> torch.Tensor:
        entity = self.resolve_entity(self.entity_name)
        # Padded: the trimmed form reads the contact count back to the host every step.
        contacts = entity.get_contacts(with_entity=entity, is_padded=True)
        a, b = contacts["link_a"] - entity.link_start, contacts["link_b"] - entity.link_start
        links, other = self._links, self._other_links
        touching = (torch.isin(a, links) & torch.isin(b, other)) | (torch.isin(a, other) & torch.isin(b, links))
        return (touching & contacts["valid_mask"]).any(dim=-1).float()


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


def _bench_scene() -> SceneOptions:
    return en.scenes.Empty(
        robot=_ROBOT,
        # Holds the pelvis: invisible, never in contact, welded to the robot below.
        pelvis_mount=BoxOptions(
            size=(0.02, 0.02, 0.02),
            fixed=True,
            default_root_pos=_PELVIS_POS,
            visualization=False,
            contype=0,
            conaffinity=0,
        ),
        bench_top=BoxOptions(
            size=(_BENCH_DEPTH, _BENCH_WIDTH, _SLAB),
            fixed=True,
            default_root_pos=(_EDGE_X + _BENCH_DEPTH / 2, _BENCH_Y, _BENCH_TOP - _SLAB / 2),
            material=RigidMaterialOptions(friction=_BEAKER_FRICTION),
            color=(0.85, 0.85, 0.80, 1.0),
        ),
        bench_cabinet=BoxOptions(
            size=(_CABINET_X[1] - _CABINET_X[0], _BENCH_WIDTH, _BENCH_TOP - _SLAB),
            fixed=True,
            default_root_pos=(sum(_CABINET_X) / 2, _BENCH_Y, (_BENCH_TOP - _SLAB) / 2),
            color=(0.35, 0.40, 0.45, 1.0),
        ),
        beaker=CylinderOptions(
            radius=_BEAKER_RADIUS,
            height=_BEAKER_HEIGHT,
            default_root_pos=_BEAKER_HOME,
            material=RigidMaterialOptions(
                rho=_BEAKER_MASS / (torch.pi * _BEAKER_RADIUS**2 * _BEAKER_HEIGHT),
                friction=_BEAKER_FRICTION,
            ),
            color=(0.75, 0.90, 1.0, 0.6),
        ),
        # Visual-only goal marker, moved to the goal during eval and inference.
        goal_marker=SphereOptions(
            radius=0.02,
            material=KinematicMaterialOptions(),
            color=(0.1, 0.9, 0.1, 0.85),
            default_root_pos=(_BEAKER_HOME[0], _BEAKER_HOME[1], _REST_Z + 0.10),
        ),
        attachments_dict={"robot": ("pelvis_mount", "box_baselink", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))},
    )


# ---------------------------------------------------------------------------
# MDP
# ---------------------------------------------------------------------------

_GRASP = {"gripper_name": "robot", "grasp_links": _TIPS, "approach_offset": 0.0}
_LIFT_GATE = {"object_name": "beaker", "rest_z": _REST_Z, "lift_margin": 0.02, "grasp_threshold": 0.06}
_BEAKER_B = {"robot_name": "robot", "object_name": "beaker"}
_TIP_TERMS = {"hand_name": "robot", "object_name": "beaker", "fingertip_links": _TIPS}

# What the robot measures, each with its sensor's error (noise applies to the policy group only).
_Measured = dict(
    arm_dofs_pos=observations.DofsPos.configure(
        entity_name="robot", dofs_name=_ARM_DOFS, noise=UniformNoise.configure(n_min=-0.005, n_max=0.005)
    ),
    arm_dofs_vel=observations.DofsVel.configure(
        entity_name="robot", dofs_name=_ARM_DOFS, noise=UniformNoise.configure(n_min=-0.05, n_max=0.05)
    ),
    hand_dofs_pos=observations.DofsPos.configure(
        entity_name="robot", dofs_name=_HAND_DOFS, noise=UniformNoise.configure(n_min=-0.02, n_max=0.02)
    ),
    hand_dofs_vel=observations.DofsVel.configure(
        entity_name="robot", dofs_name=_HAND_DOFS, noise=UniformNoise.configure(n_min=-0.1, n_max=0.1)
    ),
    # The beaker pose from perception: +/-8 mm per axis and ~3 deg, with 3 frames of history to filter it. The
    # rotation is 6D, so the sign a pose estimator picks for a quaternion does not matter.
    object_position=ObsTerm.configure(
        func=custom_terms.object_position_b,
        params=_BEAKER_B,
        noise=UniformNoise.configure(n_min=-0.008, n_max=0.008),
        history_length=3,
    ),
    object_orientation=ObsTerm.configure(
        func=custom_terms.object_rot6d_b,
        params=_BEAKER_B,
        noise=GaussianNoise.configure(std=0.05),
        history_length=3,
    ),
    # The goal is the operator's command: known exactly.
    target_object_position=ObsTerm.configure(
        func=observations.generated_commands,
        params={"command_name": "lift_target"},
    ),
    # Forward kinematics of the true joints in training, of the real encoders on hardware.
    grasp_point=ObsTerm.configure(func=custom_terms.grasp_point_b, params={"robot_name": "robot", **_GRASP}),
    fingertips_pos=ObsTerm.configure(
        func=dex_fingertips_pos_b,
        params={"robot_name": "robot", "hand_name": "robot", "fingertip_links": _TIPS},
    ),
    last_action=ObsTerm.configure(func=observations.last_action),
)

# What only the simulator knows: the critic's alone.
_Privileged = dict(
    # From the true beaker pose (in the policy group they would bypass the perception noise).
    fingertips_to_object=ObsTerm.configure(func=dex_fingertips_to_object_obs, params=_TIP_TERMS),
    object_lin_vel=ObsTerm.configure(func=observations.base_lin_vel, params={"entity_name": "beaker"}),
    object_ang_vel=ObsTerm.configure(func=observations.base_ang_vel, params={"entity_name": "beaker"}),
)


@TASK_REGISTRY.register()
class ManipulationG1BeakerLiftConfig(EdenRLConfig):
    env_options: EnvOptions = EnvOptions(
        num_envs=4096,
        episode_length_s=_EPISODE_S,
        sim_dt=0.005,
        sim_substeps=2,
        decimation=4,
        max_collision_pairs=30,
        multiplier_collision_broad_phase=12,
        enable_multi_contact=True,
        # RandomizeMassShift writes a per-env beaker mass.
        batch_links_info=True,
        # Friction constraints 10x stiffer than the normal ones. At Genesis' pyramidal-cone default (1) the soft
        # friction let a held beaker creep round in the grasp at a steady 3.5-4 deg/s with the arm still and 46 N of
        # grip, and the policy learned to chase it with the wrist; at 10 it turns 0.45 deg/s, at 100 0.06 deg/s. But
        # 100 lets a hand striking the beaker make the contact solver fling the arm too: under an early policy's
        # exploration, arm joints reached 7000 rad/s at 100, 140 at 10 and 37 at 1, and the third version's value
        # loss reached 3e18. The noslip pass stops the creep as well but cuts training throughput 4.5x.
        impratio=10.0,
    )

    scene_options: SceneOptions = _bench_scene()

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        policy=ObservationGroupOptions(**_Measured, enable_corruption=True),
        critic=ObservationGroupOptions(**_Measured, **_Privileged, enable_corruption=False),
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        # 6-DOF differential IK at the palm centre over the seven right-arm joints, well damped, with a per-step
        # joint clamp, accumulated on the last target so the unassisted arm holds its pose. The clamp is the speed
        # limit; the scales are not, because the damping is large against this arm's J Jt and shrinks every step a
        # lot (0.3 of full +z for 60 steps raised the palm 3 cm), so a policy drives the actions past 1. Measured
        # palm speed during the reach (median): 0.61 m/s at 0.2 rad a step, 0.29 at 0.03, 0.19 at 0.015.
        arm_ik=AccumulatedDifferentialIKController.configure(
            entity_name="robot",
            dofs_name=_ARM_DOFS,
            ee_link_name=_EE_LINK,
            ee_local_pos=_PALM_IN_WRIST,
            position_scale=0.02,
            orientation_scale=0.05,
            damping=0.1,
            max_dq=0.015,
        ),
        # Joint PD on the right Dex3 around its open default; one action unit is half of each joint's range, so the
        # fingers reach closed within the policy's first few standard deviations.
        hand=ImplicitPDController.configure(
            entity_name="robot",
            dofs_name=_HAND_DOFS,
            scale_ratio=0.5,
            reference_source=ReferenceSource.DEFAULT,
        ),
        # Legs, waist, left arm and left hand stay at their defaults; no action columns.
        hold_body=NullJointAction.configure(entity_name="robot", dofs_name=_HELD_DOFS),
    )

    command_options: CommandManagerOptions = CommandManagerOptions(
        lift_target=UniformT3Command.configure(
            # In the pelvis frame: over the beaker's area, 10-25 cm above where it rests.
            x_range=(0.25, 0.40),
            y_range=(-0.30, 0.0),
            z_range=(_REST_Z + 0.10 - _PELVIS_POS[2], _REST_Z + 0.25 - _PELVIS_POS[2]),
            # Longer than the episode: one goal per episode, drawn on reset.
            resampling_time_range=(2 * _EPISODE_S, 2 * _EPISODE_S),
            debug_vis=True,
            marker_frame_entity="robot",
            marker_entity_name="goal_marker",
        )
    )

    metric_options: MetricManagerOptions = MetricManagerOptions(
        beaker_tilt=MetricTerm.configure(func=object_tilt, params={"object_name": "beaker"}),
        # The beaker's bottom above the bench top: 0 resting, and 0 tipped onto its rim.
        beaker_clearance=MetricTerm.configure(
            func=cylinder_clearance,
            params={
                "object_name": "beaker",
                "radius": _BEAKER_RADIUS,
                "height": _BEAKER_HEIGHT,
                "support_z": _BENCH_TOP,
            },
        ),
        beaker_grasped=OpposedGrasp.configure(
            entity_name="robot",
            object_name="beaker",
            thumb_links=_THUMB_LINKS,
            finger_links=_FINGER_LINKS,
            force_threshold=0.5,
            contributes_to_success=False,
        ),
        # Thumb, index and middle finger each pressing it.
        beaker_three_digit=DigitsPressing.configure(
            entity_name="robot",
            object_name="beaker",
            digits=_DIGITS,
            force_threshold=0.5,
            contributes_to_success=False,
        ),
    )

    # The approach terms are worth at most 5 a step, well under a grasp (4 + 2 for closing) and far under a lift
    # (up to 5 + 5 + 35): in the first version they paid 8.5, and a hand lying open around the beaker's foot
    # collected 6.8 of it for 2700 iterations without ever closing.
    reward_options: RewardManagerOptions = RewardManagerOptions(
        reaching_object=RewardTerm.configure(
            func=custom_terms.object_ee_distance,
            weight=2.0,
            params={**_GRASP, "object_name": "beaker", "std": 0.3},
        ),
        # Sharp, unclamped fine reach: gradient across the last few cm into the grasp.
        reaching_object_fine=RewardTerm.configure(
            func=custom_terms.object_ee_distance,
            weight=2.0,
            params={**_GRASP, "object_name": "beaker", "std": 0.05, "clamp_max": 1.0},
        ),
        # The fingertips wrapping the beaker.
        fingertip_to_object=RewardTerm.configure(
            func=dex_fingertip_to_object_reward,
            weight=1.0,
            params={**_TIP_TERMS, "sigma": 0.06},
        ),
        # Close the hand once it is at the beaker.
        hand_closure=GraspClosure.configure(
            weight=2.0,
            entity_name="robot",
            object_name="beaker",
            grasp_links=_TIPS,
            closed_dofs_pos=_HAND_CLOSED,
            near_distance=0.08,
        ),
        # Thumb and fingers both pressing the beaker.
        beaker_grasped=rewards.MetricReward.configure(metric_name="beaker_grasped", weight=4.0),
        # On top of it, the middle finger pressing too: the fourth version's policy held 38 % of its beakers in a pinch
        # and touched the middle finger to the rest with 2 N, against 20 N on the thumb and the index.
        beaker_three_digit=rewards.MetricReward.configure(metric_name="beaker_three_digit", weight=2.0),
        # How far the beaker's bottom clears the bench, while grasped.
        lift_height=GraspedLiftHeight.configure(
            weight=20.0, height_metric="beaker_clearance", grasp_metric="beaker_grasped", cap=0.25
        ),
        # These and the goal terms gate on the centre 2 cm up, which no tipping on the bench reaches (1.15 cm).
        lifting_object=RewardTerm.configure(
            func=custom_terms.object_is_lifted,
            weight=5.0,
            params={**_GRASP, **_LIFT_GATE},
        ),
        object_goal_tracking=RewardTerm.configure(
            func=custom_terms.object_goal_distance,
            weight=25.0,
            params={"robot_name": "robot", **_GRASP, **_LIFT_GATE, "command_name": "lift_target", "std": 0.3},
        ),
        # 2 cm deadband so settle_at_goal can still the arm at the goal.
        object_goal_tracking_fine=RewardTerm.configure(
            func=custom_terms.object_goal_distance,
            weight=10.0,
            params={
                "robot_name": "robot",
                **_GRASP,
                **_LIFT_GATE,
                "command_name": "lift_target",
                "std": 0.05,
                "deadband": 0.02,
            },
        ),
        # Keep it upright once it is gripped or off the bench: ~0.06 at 20 deg; past 45 deg the episode ends wherever
        # it is.
        beaker_upright=HeldObjectTilt.configure(
            weight=-20.0,
            tilt_metric="beaker_tilt",
            clearance_metric="beaker_clearance",
            grasp_metric="beaker_grasped",
            margin=0.01,
        ),
        # Carry it smoothly: the beaker's angular velocity squared, held (1 rad/s costs 0.5 a step, at most 4.5).
        beaker_ang_vel=HeldObjectAngVel.configure(
            weight=-0.5,
            object_name="beaker",
            clearance_metric="beaker_clearance",
            grasp_metric="beaker_grasped",
            margin=0.01,
            max_ang_vel=3.0,
        ),
        # The hand pressing on the bench (the first version's policy lay on it): -0.02 per N, at most -0.6.
        hand_on_bench=ContactForceWith.configure(
            weight=-0.02, entity_name="robot", links_name=_HAND_LINKS, other_name="bench_top", max_force=30.0
        ),
        # The forearm or hand on the torso: it stops the arm now, and the policy must not brace the beaker against it.
        arm_on_torso=LinksTouching.configure(
            weight=-1.0, entity_name="robot", links_name=_FOREARM_AND_HAND, other_links_name=["torso_link"]
        ),
        # Smoothness regularizers, ramped in by the curriculum once the grasp and lift are learned.
        action_rate=RewardTerm.configure(func=rewards.action_rate_l2, weight=-0.05),
        arm_joint_vel=NamedDofsVelLimits.configure(
            entity_name="robot", dofs_name=_ARM_DOFS, max_vel=0.5, cap=15.0, weight=-0.01
        ),
        # Hold still once the beaker is at the goal (goal tracking is flat there).
        settle_at_goal=RewardTerm.configure(
            func=custom_terms.settle_at_goal,
            weight=-6.0,
            params={"robot_name": "robot", "object_name": "beaker", "command_name": "lift_target", "std": 0.05},
        ),
        arm_joint_pos_limits=NamedDofsPosLimits.configure(entity_name="robot", dofs_name=_ARM_DOFS, weight=-1.0),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
        beaker_fallen=DoneTerm.configure(
            func=custom_terms.object_fallen,
            time_out=False,
            params={"object_name": "beaker", "threshold": _BENCH_TOP - 0.04},
        ),
        beaker_spilled=terminations.MetricAbove.configure(metric_name="beaker_tilt", threshold=_SPILL_TILT),
        # Flung by the contact solver. A knocked-over beaker turns at 10-20 rad/s, and the palm moves under 0.5 m/s.
        beaker_ejected=DoneTerm.configure(
            func=object_ejected,
            time_out=False,
            params={"object_name": "beaker", "max_lin_vel": 3.0, "max_ang_vel": 50.0},
        ),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        place_beaker=EventTerm.configure(
            func=events.reset_base_state_uniform,
            mode=EventMode.RESET,
            params={
                "entity_name": "beaker",
                "pos_x_range": _PLACE_X,
                "pos_y_range": _PLACE_Y,
                "pos_z_range": (0.0, 0.0),
                "yaw_range": (-3.14159, 3.14159),
            },
        ),
        # Someone moves the beaker before the hand gets to it.
        move_beaker=DisplaceRestingObject.configure(
            interval_range_s=(1.0, 3.0),
            object_name="beaker",
            hand_name="robot",
            grasp_links=_TIPS,
            rest_z=_REST_Z,
            min_hand_distance=0.10,
            probability=0.5,
            pos_x_range=_PLACE_X,
            pos_y_range=_PLACE_Y,
            yaw_range=(-3.14159, 3.14159),
        ),
        # Every held joint is driven back to its default by NullJointAction; the jitter matters for the right arm.
        # Small: summed over the arm's joints, +/-0.05 rad lowered the hand onto beakers by the bench edge.
        reset_robot_dofs=EventTerm.configure(
            func=events.reset_dofs_by_offset,
            mode=EventMode.RESET,
            params={"entity_name": "robot", "dofs_pos_range": (-0.02, 0.02)},
        ),
        # A quarter of the episodes start with the open hand around the beaker at its home (after the two above).
        pregrasp_start=StartAtPregrasp.configure(
            priority=1,
            probability=0.25,
            entity_name="robot",
            dofs_pos=_PREGRASP_ARM,
            object_name="beaker",
            # The open hand clears the beaker by about 1.5 cm on the palm side.
            pos_jitter=0.005,
            yaw_range=(-3.14159, 3.14159),
        ),
        # Stiffen the right hand's contact with the beaker, so a grasp cannot rest on fingers sunk into it.
        stiffen_contact=EventTerm.configure(
            func=events.set_contact_sol_params,
            mode=EventMode.STARTUP,
            params={
                "sol_params": (0.005, 1.0, 0.95, 0.999, 0.0005, 0.5, 2.0),
                "entity_links": {"beaker": None, "robot": _HAND_LINKS},
            },
        ),
        # Per-env hand friction; it decides every hand-beaker contact (the beaker's is lower, and a pair takes the max).
        hand_friction=RandomizeFrictionRatio.configure(
            mode=EventMode.STARTUP,
            entity_name="robot",
            links_name=_HAND_LINKS,
            friction_range=(0.8, 2.0),
        ),
        # An empty beaker to one with about 120 mL in it.
        beaker_mass=events.RandomizeMassShift.configure(
            mode=EventMode.STARTUP,
            entity_name="beaker",
            links_name=[".*"],
            mass_shift_range=(-0.06, 0.12),
        ),
    )

    curriculum_options: CurriculumManagerOptions = CurriculumManagerOptions(
        # The goal starts as one point 10 cm above the beaker's home and widens as goals are reached.
        goal_range=custom_terms.GoalRangeCurriculum.configure(
            command_name="lift_target",
            object_name="beaker",
            robot_name="robot",
            reach_thresh=0.08,
            advance_thresh=0.7,
            retreat_thresh=0.4,
            step_up=0.05,
            step_down=0.05,
            level_init=0.0,
            easy_center=(_BEAKER_HOME[0], _BEAKER_HOME[1], _REST_Z + 0.10 - _PELVIS_POS[2]),  # pelvis frame
        ),
        # Ramp the smoothness penalties in after the grasp and lift. Stage steps are common_step_counter
        # (iteration * num_steps_per_env, here 24).
        action_rate_ramp=StageRewardWeightCurriculum.configure(
            reward_term_name="action_rate",
            weight_stages=[(0, -0.05), (1200 * 24, -0.2), (1800 * 24, -0.5)],
        ),
        arm_joint_vel_ramp=StageRewardWeightCurriculum.configure(
            reward_term_name="arm_joint_vel",
            weight_stages=[(0, -0.01), (1200 * 24, -0.2), (1800 * 24, -0.5)],
        ),
    )

    runner_options: RslRlOnPolicyRunnerOptions = RslRlOnPolicyRunnerOptions(
        algorithm=RslRlPpoAlgorithmOptions(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        actor=RslRlActorOptions(
            activation="elu",
            hidden_dims=[512, 512, 256, 128],
            distribution_cfg=RslRlGaussianDistributionOptions(init_std=1.0),
        ),
        critic=RslRlCriticOptions(
            activation="elu",
            hidden_dims=[512, 512, 256, 128],
        ),
        max_iterations=3000,
        save_interval=50,
        experiment_name="g1_beaker_lift",
        run_name="v5",
    )
