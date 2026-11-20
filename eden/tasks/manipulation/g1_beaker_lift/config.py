"""G1 beaker lift (``manipulation_g1_beaker_lift``): a standing Unitree G1 picks a beaker off a lab bench with its
right Dex3 hand, keeps it upright, and holds it at a 3D goal.

The humanoid counterpart of :class:`~eden.tasks.manipulation.dex_lift.config.ManipulationDexLiftConfig`
(``manipulation_dex_lift``): the same reach / wrap / lift / goal-tracking reward structure, goal-range curriculum,
staged smoothness curricula and contact stiffening, on a :class:`~eden.options.robots.G1Dofs29Dex3` instead of a
table-mounted Franka + XHand. What changes:

* **The pelvis is welded in place.** Balance is out of scope: the task is the upper body at a bench, and a deployment
  stands the robot with a lower-body controller. The G1 MJCF's base carries a free joint, which ``is_fixed_base``
  cannot pin (Eden refuses the mismatch), so the robot is welded to ``pelvis_mount``, an invisible, non-colliding
  fixed box at its standing pelvis pose. Welding it to the ground plane instead fails at build, because Genesis then
  voxelizes the plane as part of the robot's own collision tree.
* **Only the right arm and hand act.** 6-DOF differential IK on the seven right-arm joints, at a control point in
  the palm, and joint PD on the seven right Dex3 joints: a 13-D action. The IK deltas accumulate on the last joint
  target (:class:`AccumulatedDifferentialIKController`): the G1 arm has no gravity compensation, so deltas on the
  measured joints would let it sink onto the bench under a zero action. Every other joint is held at its default by
  :class:`NullJointAction`, which takes no action columns. Only the right forearm and hand collide.
* **The bench sets the reach.** With the pelvis and waist fixed, the palm reaches about 35 cm in front of the pelvis
  at 0.80 m and 40-45 cm at 0.90-1.05 m (the shoulder is at 1.075 m). The bench top is therefore 0.85 m (lab benches
  are about 0.9 m), and it is a slab with knee room under it, so its front edge can sit 20 cm from the pelvis. The
  beaker starts anywhere in 12 x 30 cm of bench in front of the right shoulder; the goal is 10-25 cm above the bench
  over the same area.
* **The arm starts ready.** The asset's default arm hangs below the bench top. The right arm's default is a ready
  pose instead: the hand raised behind the bench edge, clear of everywhere the beaker can be (a hand above the beaker
  area sags onto it under gravity and tips it). The arm starts every episode there.
* **The beaker must stay upright.** :func:`object_tilt` (``1 - cos`` of the beaker's tilt) is a metric that a
  penalty reads and that ends the episode past 45 degrees: a tipped beaker has spilled.
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

from typing import TYPE_CHECKING

import torch
from genesis.typing import Vec2FType

import eden as en
from eden import EventMode, events, observations, rewards, terminations
from eden.constants import MetricDirection, ReferenceSource
from eden.managers import METRIC_TERM_REGISTRY, REWARD_TERM_REGISTRY
from eden.managers.action_manager import ACTION_TERM_REGISTRY
from eden.managers.event_manager import EVENT_TERM_REGISTRY
from eden.managers.event_manager import EventTerm as EventTermBase
from eden.managers.modifiers.observations.noise import GaussianNoise, UniformNoise
from eden.managers.reward_manager import RewardTerm as RewardTermBase
from eden.managers.terms.actions.joint_actions import ImplicitPDController, NullJointAction
from eden.managers.terms.actions.task_space_actions import DifferentialIKController
from eden.managers.terms.commands.oracle import UniformT3Command
from eden.managers.terms.curricula.rewards import StageRewardWeightCurriculum
from eden.managers.terms.events import RandomizeFrictionRatio
from eden.managers.terms.events.resets import reset_base_state_uniform
from eden.managers.terms.utils import resolve_link_idx, soft_dof_pos_violation
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


def _ready_dofs_pos() -> dict[str, float]:
    """The G1Dofs29Dex3 default joint positions with the right arm at :data:`_RIGHT_ARM_READY`.

    The asset's ``*_shoulder_pitch_joint``-style keys are split into a ``left_`` and a ``right_`` key first, because
    a joint matched by two keys is an error and the right arm's keys override exactly those.
    """
    split = {}
    for key, value in G1Dofs29Dex3.model_fields["default_dofs_pos"].default.items():
        if key.startswith("*_"):
            split[f"left_{key[2:]}"] = value
            split[f"right_{key[2:]}"] = value
        else:
            split[key] = value
    return {**split, **_RIGHT_ARM_READY}


_ROBOT = G1Dofs29Dex3(
    # The MJCF's base is a free joint, so the flag must say so; the weld to ``pelvis_mount`` pins it.
    is_fixed_base=False,
    default_dofs_pos=_ready_dofs_pos(),
    # Nothing else reaches the bench or the beaker, and every pair dropped is collision work saved.
    collision_link_patterns=["right_elbow_link", "right_wrist_.*", "right_hand_.*"],
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
    """Quadratic hinge on ``|joint velocity| - max_vel`` over ``dofs_name`` alone (negative weight to penalize).

    ``dofs_vel_limits`` over the whole G1 would also slow the Dex3 fingers, which close faster than an arm moves.
    """

    entity_name: str = "robot"
    dofs_name: list[str] = []
    max_vel: float = 0.5

    def build(self) -> None:
        entity = self.resolve_entity(self.entity_name)
        if not self.dofs_name:
            raise ValueError(
                f"{term_subject(self)}: dofs_name is empty, so there is nothing to penalize. Name the joints, e.g. "
                f"dofs_name={list(entity.dofs_name[:2])}; for every joint of the entity use dofs_vel_limits."
            )
        if self.max_vel <= 0.0:
            raise ValueError(f"{term_subject(self)}: max_vel must be > 0 rad/s, got {self.max_vel}.")
        self._dofs_idx_local = entity.find_named_dofs(self.dofs_name).dofs_idx_local

    def compute(self) -> torch.Tensor:
        dofs_vel = self.resolve_entity(self.entity_name).get_dofs_vel(dofs_idx_local=self._dofs_idx_local)
        return (dofs_vel.abs() - self.max_vel).clamp_min(0.0).square().sum(dim=-1)


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
        move = resting & untouched & drawn & self._selected(envs_idx)
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

    def _selected(self, envs_idx: slice | torch.Tensor | None) -> torch.Tensor:
        """The envs this fire covers, as a ``(num_envs,)`` bool mask."""
        if envs_idx is not None and is_env_mask(envs_idx):
            return envs_idx
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[slice(None) if envs_idx is None else envs_idx] = True
        return mask


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
    )

    scene_options: SceneOptions = _bench_scene()

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        policy=ObservationGroupOptions(**_Measured, enable_corruption=True),
        critic=ObservationGroupOptions(**_Measured, **_Privileged, enable_corruption=False),
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        # 6-DOF differential IK at the palm centre over the seven right-arm joints, well damped, with a per-step
        # joint clamp, accumulated on the last target so the unassisted arm holds its pose.
        arm_ik=AccumulatedDifferentialIKController.configure(
            entity_name="robot",
            dofs_name=_ARM_DOFS,
            ee_link_name=_EE_LINK,
            ee_local_pos=_PALM_IN_WRIST,
            position_scale=0.05,
            orientation_scale=0.10,
            damping=0.1,
            max_dq=0.2,
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
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        reaching_object=RewardTerm.configure(
            func=custom_terms.object_ee_distance,
            weight=5.0,
            params={**_GRASP, "object_name": "beaker", "std": 0.3},
        ),
        # Sharp, unclamped fine reach: gradient across the last few cm into the grasp.
        reaching_object_fine=RewardTerm.configure(
            func=custom_terms.object_ee_distance,
            weight=3.0,
            params={**_GRASP, "object_name": "beaker", "std": 0.05, "clamp_max": 1.0},
        ),
        # The fingertips wrapping the beaker.
        fingertip_to_object=RewardTerm.configure(
            func=dex_fingertip_to_object_reward,
            weight=2.0,
            params={**_TIP_TERMS, "sigma": 0.06},
        ),
        # Dense height reward: any upward beaker motion is rewarded.
        lift_height=RewardTerm.configure(
            func=custom_terms.lift_height,
            weight=15.0,
            params={"object_name": "beaker", "rest_z": _REST_Z, "cap": 0.25},
        ),
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
        # Keep it upright: 0 while it stands, ~0.06 at 20 deg; past 45 deg the episode ends.
        beaker_upright=rewards.MetricReward.configure(metric_name="beaker_tilt", weight=-20.0),
        # Smoothness regularizers, ramped in by the curriculum once the grasp and lift are learned.
        action_rate=RewardTerm.configure(func=rewards.action_rate_l2, weight=-0.05),
        arm_joint_vel=NamedDofsVelLimits.configure(entity_name="robot", dofs_name=_ARM_DOFS, max_vel=0.5, weight=-0.01),
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
        run_name="v1",
    )
