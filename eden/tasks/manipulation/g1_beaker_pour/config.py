"""G1 beaker pour (``manipulation_g1_beaker_pour``): the standing G1 of ``manipulation_g1_beaker_lift`` picks up a
beaker of beads, pours them into a second beaker on the bench, and puts the beaker back where it stood.

:class:`ManipulationG1BeakerPourConfig` subclasses
:class:`~eden.tasks.manipulation.g1_beaker_lift.config.ManipulationG1BeakerLiftConfig`: the robot, the bench and the
actions are the lift's, and so are the policy's observations but the last (whether the source is poured), so a lift
checkpoint drives this task once widened for it. What changes:

* **Both beakers are hollow, and the beads are rigid spheres.** A beaker is an open polygonal cylinder of wall
  panels on a bottom disc (:class:`HollowBeakerOptions`, generated MJCF), the lift's beaker's size and, with its
  beads, its mass. The beads (:class:`BeadsOptions`) stand in for a liquid: eight free spheres in one entity,
  stacked in the source beaker at every reset (:class:`FillBeads`). They leave it past about 100 degrees of tilt.
  Eight, not more: each bead is six DOFs in every constraint solve, and 20 beads trained 7x slower.
* **Everything that touches collides.** The beads with each other, both beakers, the bench and the hand; the beakers
  with each other, the hand and the bench. The lift's stiff contact on the source covers the target and the beads,
  and ``max_collision_pairs`` has room for twice the most contacts measured (a count past it halts the run): a
  beaker on its rim makes a contact patch on each of its 16 wall panels, 199 contacts with the source upside down on
  the target's rim, where at rest the beads make about 20.
  Pouring, spilling and noised actions in 768 envs never put a bead's centre inside a beaker's glass.
* **So does the arm, with the body.** Besides the lift's right forearm and hand on the torso, the right upper arm and
  shoulder collide with the torso, and the right arm with the left one (``arm_on_torso`` costs every such contact).
  Without them, the first version's policy, pouring outward, sank its upper arm 5-10 mm into the torso in every
  episode. Held there by contact, the upper arm touches the torso only once the shoulder pulls the arm in toward the
  body (roll past 0.15 rad), which no pour on this bench needs.
* **The hand pours inward, pronating, as a person does.** The right hand pours toward +y, the body's middle, from the
  target's right; the left hand's policy is this one mirrored. Pouring outward (supinating), the first version ran
  out of forearm turn (113 degrees, a quarter of it spent by the grasp) and made the rest of 135 degrees up by pulling
  the shoulder in 23 degrees, into the torso, and bending the wrist to its limit, which it reached in 40 % of its
  pours. A scripted inward pour from the lift's grasp reached 137 degrees with no joint at a limit.
* **The finger pads resist spin.** Palm down, the beaker hangs in the fingers, and point contacts let it swing out
  about the line through them: a scripted inward pour from the lift's grasp dropped 59 % of the beakers by 130
  degrees, and closing the fingers harder or turning a third as fast changed nothing. A rubber pad resists that turn
  over its contact patch (torsional friction, ``friction_torsional``, the patch radius): at 1 cm the same pour held
  86 % to 130 degrees, at 0.5 cm 58 %. The hand's is 8 mm, scaled every episode with its friction (0.8-2x). Whether
  the real pads hold as well is the assumption to check on hardware.
* **The target stands toward the middle from the source.** The source is on the right half of the bench (y -0.33 to
  -0.02 m), the target on its left, at least 12 cm across and no further than y +0.10 (:class:`PlaceBeside`): with
  the beaker tipped 120 degrees in the lift's grasp, the IK reaches a pour with the beaker's centre from y -0.35 to
  +0.05, and further left the upper arm meets the torso. The lift's grasp closes round the source from its right, so
  a target on that side is in the hand's way: with the target anywhere either side, the second version's policy
  knocked it over in 64 % of the episodes with it 12-18 cm to the source's right, 45 % at 18-21 cm and 11 % at 25-35
  cm, 0.4 s in, as often after 300 iterations as at the start, and in 0-3 % with it on the left at any distance. So the
  hand on the source's outer side pours, toward the middle; a target further out than the source, on its side of
  the bench, is left to no hand.
* **The goal is where the beaker's centre is while it pours.** :class:`AnchoredT3Command` draws it 4-6 cm from the
  target's axis toward -y and 12-14 cm above the target's centre, and the actor, which never observes the target,
  finds it at a fixed offset from the goal. The lift checkpoint holds a beaker's centre on its goal and, tipped,
  turns it about that centre, so from there the lip stays over the target's mouth from upright to about 120
  degrees, past where the beads pour, and tipping toward the target pays from the first degree. With the goal 11-14
  cm off (where the swing of a beaker tipped about the hand would bring the lip over the target), the lift policy
  pulled the swinging beaker back to the goal, the lip never neared the target, and a bias toward the target on its
  wrist lost return over an episode. The goal terms measure the beaker turned back upright about the palm
  (:class:`UprightedGoalDistance`), so tipping it about the hand neither costs nor pays either, and pay in full once
  it pours (its lip over the target), wherever the palm then is.
* **The beaker may tip toward the target.** The lift's upright cost becomes a cost on leaning across or away from
  the target (:class:`HeldTiltOffTarget`), and its tilt termination and settle-at-goal cost are gone; the held
  beaker's angular velocity costs a fifth as much, and the lift-height reward is capped at 10 cm of clearance,
  which a tipped beaker held at the goal keeps. What pays is the share of beads in the target
  (``beads_in_target``), and what costs is the share spilled onto the bench (``beads_spilled``, which ends the
  episode past half), and so does knocking the target over. Getting there, :class:`PourTilt` pays the beaker's tilt
  toward the target (:func:`tilt_toward`) while its pour lip (the lowest point of its rim, :func:`pour_lip`) is near
  over the target's mouth and above its rim.
* **Then it puts the beaker back.** Poured (at most one bead left in it), the source goes back where it stood when
  the episode started: :class:`PourThenPlaceCommand` sends it 12 cm over that place first, then, once it is there
  and upright, onto it. Sent straight there, the pour policy took the tipped beaker down through the target and
  knocked it over in 34 % of the episodes; sent over it, in none. The terms that reached, held, lifted, carried and
  tipped the source pay only until it is poured (:class:`PourPhaseGate`), since each would pay for holding on; then
  being poured pays as much as the pour paid at most (:class:`PourStage`), the way left back pays, upright
  (:class:`UprightHomeDistance`), and so do the source standing back on its place (:class:`BackAtHome`), the hand
  off it, and then the arm back at its start (:class:`DofsAtDefault`), which lapses if the retreat knocks the
  source off its place. Playing the pick-up back in reverse instead put it back in 18 % of the episodes: the beaker
  shifts and turns in the hand while it pours.
* **The critic counts the beads.** It gets the lift's critic observations and, after them, the shares in the
  source, in the target and spilled, and the beads' centroid from the target's mouth; both it and the actor get,
  last, whether the source is poured (:data:`ACTOR_EXTRA_DIMS` and :data:`CRITIC_EXTRA_DIMS` more inputs than the
  lift's). A lift or earlier pour checkpoint is widened for them with
  :mod:`eden.tasks.manipulation.g1_beaker_pour.warm_start`: the new inputs' weights start at zero, so each network
  starts out as it was.
* **Every episode starts with the hand raised.** No beaker is moved mid-episode, the pre-grasp start is gone (it
  puts the source at its home, a beaker's width from the target), and so is the goal-range curriculum (the goal
  follows the target). The smoothness costs sit at their final weights from the start.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from pydantic import model_validator

from eden import EventMode, observations, rewards, terminations
from eden.constants import MetricDirection
from eden.managers import COMMAND_TERM_REGISTRY, METRIC_TERM_REGISTRY, OBSERVATION_TERM_REGISTRY, REWARD_TERM_REGISTRY
from eden.managers.base import build_term
from eden.managers.event_manager import EVENT_TERM_REGISTRY
from eden.managers.event_manager import EventTerm as EventTermBase
from eden.managers.metric_manager import MetricTerm as MetricTermBase
from eden.managers.reward_manager import RewardTerm as RewardTermBase
from eden.managers.terms.commands.oracle import UniformT3Command
from eden.managers.terms.utils import resolve_link_idx, resolve_metric_manager
from eden.options.entities import EntityOptions
from eden.options.learning.rsl_rl import RslRlOnPolicyRunnerOptions
from eden.options.managers.rewards import RewardTermOptions
from eden.options.materials import RigidMaterialOptions
from eden.tasks.manipulation.custom_terms.common import _base_to_world, _to_base_frame
from eden.tasks.manipulation.custom_terms.rewards import _lifted_and_close
from eden.tasks.manipulation.g1_beaker_lift.config import (
    _ARM_DOFS,
    _BEAKER_HEIGHT,
    _BEAKER_HOME,
    _BEAKER_RADIUS,
    _BENCH_TOP,
    _EE_LINK,
    _FOREARM_AND_HAND,
    _GRASP,
    _HAND_LINKS,
    _LIFT_GATE,
    _PALM_IN_WRIST,
    _REST_Z,
    _ROBOT,
    _SPILL_TILT,
    ManipulationG1BeakerLiftConfig,
    _bench_scene,
    _env_mask,
    _held,
    object_tilt,
)
from eden.tasks.manipulation.g1_beaker_pour.mirror import mirror_config
from eden.tasks.registry import TASK_REGISTRY
from eden.terms import MetricTerm, ObsTerm
from eden.utils.configs import (
    ActionManagerOptions,
    CommandManagerOptions,
    CurriculumManagerOptions,
    EnvOptions,
    EventManagerOptions,
    MetricManagerOptions,
    ObservationManagerOptions,
    RewardManagerOptions,
    SceneOptions,
    TerminationManagerOptions,
)
from eden.utils.geom import inv_transform_by_quat, quat_from_axis_angle, quat_mul, transform_by_quat
from eden.utils.misc import assign_over_envs, get_cache_dir
from eden.utils.procedural_meshes import write_atomically
from eden.utils.sample import sample_uniform_ranges
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

# A glass beaker the lift's size: 4 mm walls, a 6 mm bottom, 16 wall panels.
_WALL = 0.004
_BOTTOM = 0.006
_SIDES = 16
_INNER_RADIUS = _BEAKER_RADIUS - _WALL
#: Beaker-frame z (origin at the centre) of the inner floor and of the rim.
_FLOOR_Z = -_BEAKER_HEIGHT / 2 + _BOTTOM
_RIM_Z = _BEAKER_HEIGHT / 2
_GLASS_MASS = 0.13
_GLASS_FRICTION = 0.4

# Glass beads: with the glass, the lift's 0.18 kg. Eight, because each bead is six more DOFs in every constraint
# solve: at 4096 envs, 20 beads of 6 mm trained at 1400 env-steps/s (70 s an iteration), 8 of 8 mm at 10500.
_BEAD_COUNT = 8
_BEAD_RADIUS = 0.008
_BEAD_MASS = 0.006
_BEAD_FRICTION = 0.4

# The source on the right hand's half of the bench (y -0.33 to -0.02, offsets from its home at -0.18), the target
# toward the middle from it (+y), at least 12 cm across (5 cm of bench between the two), and no further than +0.10,
# where the hand still pours over it (y -0.21 to +0.10). The right hand pours toward +y, pronating, from the target's
# right.
_SOURCE_PLACE_X = (-0.04, 0.04)
_SOURCE_PLACE_Y = (-0.15, 0.16)
_TARGET_HOME = (0.30, -0.10, _REST_Z)
_TARGET_PLACE_X = (-0.04, 0.04)
_TARGET_PLACE_Y = (-0.11, 0.20)
_MIN_GAP_Y = 0.12
#: Where the goal is drawn, from the target's centre: where the source's centre is while it pours. 4-6 cm toward
#: -y, the side the hand pours from, so the lip of a beaker tipped about its centre stays over the target's mouth from
#: upright to about 120 degrees, past where the beads pour, and 12-14 cm up, so the lip stays 1.5 cm or more over the
#: target's rim and the upright source's bottom 2.5 cm or more.
_GOAL_FROM_TARGET_X = (-0.01, 0.01)
_GOAL_FROM_TARGET_Y = (-0.06, -0.04)
_GOAL_FROM_TARGET_Z = (0.12, 0.14)

#: Collide with the body besides the lift's forearm and hand on the torso: the right upper arm and shoulder on the
#: torso, and the right arm on the left one, which hangs at its default.
_UPPER_ARM = ["right_shoulder_roll_link", "right_shoulder_yaw_link"]
_LEFT_ARM = ["left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_.*", "left_hand_.*"]
#: Torsional friction of the hand's links, the radius (m) of a finger pad's contact patch.
_PAD_TORSION = 0.008
_POUR_ROBOT = _ROBOT.updated(
    collision_link_patterns=[*_ROBOT.collision_link_patterns, *_UPPER_ARM, *_LEFT_ARM],
    default_geoms_friction_torsional=dict.fromkeys(_HAND_LINKS, _PAD_TORSION),
)

_EPISODE_S = 10.0


# ---------------------------------------------------------------------------
# Generated assets
# ---------------------------------------------------------------------------


def hollow_beaker_file(radius: float, height: float, wall: float, bottom: float, sides: int, mass: float) -> str:
    """Path of the MJCF of an open beaker, written to Eden's cache (rewritten per call).

    One free body with its origin at the beaker's centre, which is also its inertial frame, so ``get_pos()`` is that
    centre: ``sides`` wall panels ``wall`` thick, their outer faces tangent to a circle of ``radius``, standing on a
    disc ``bottom`` thick. The inertia is a thin-walled cylinder's.
    """
    half_width = radius * math.tan(math.pi / sides)
    panel_r = radius - wall / 2
    wall_half = (height - bottom) / 2
    inner = radius - wall
    ixx = mass * (3 * (radius**2 + inner**2) + height**2) / 12
    izz = mass * (radius**2 + inner**2) / 2
    panels = "\n".join(
        f'      <geom name="wall{i}" type="box" size="{wall / 2:g} {half_width:.6f} {wall_half:g}" '
        f'pos="{panel_r * math.cos(2 * math.pi * i / sides):.6f} {panel_r * math.sin(2 * math.pi * i / sides):.6f} '
        f'{bottom / 2:g}" euler="0 0 {2 * math.pi * i / sides:.6f}" rgba="0.75 0.9 1 0.45"/>'
        for i in range(sides)
    )
    text = f"""<mujoco model="hollow_beaker">
  <compiler angle="radian"/>
  <worldbody>
    <body name="beaker">
      <freejoint name="beaker_free"/>
      <inertial pos="0 0 0" mass="{mass:g}" diaginertia="{ixx:.4e} {ixx:.4e} {izz:.4e}"/>
      <geom name="bottom" type="cylinder" size="{radius:g} {bottom / 2:g}" pos="0 0 {-height / 2 + bottom / 2:g}"
            rgba="0.75 0.9 1 0.45"/>
{panels}
    </body>
  </worldbody>
</mujoco>
"""
    name = f"beaker_r{radius:g}_h{height:g}_w{wall:g}_b{bottom:g}_n{sides}_m{mass:g}.xml"
    path = f"{get_cache_dir('g1_beaker_pour')}/{name}"
    write_atomically(path, text)
    return path


def beads_file(count: int, radius: float, mass: float) -> str:
    """Path of the MJCF of ``count`` free spheres (``bead0``, ``bead1``, ...), written to Eden's cache (rewritten per
    call). They start in a row on the floor out of the way; :class:`FillBeads` places them at every reset.
    """
    bodies = "\n".join(
        f'    <body name="bead{i}" pos="{0.03 * i:g} 2 {radius:g}"><freejoint name="bead{i}_free"/>'
        f'<geom name="bead{i}" type="sphere" size="{radius:g}" mass="{mass:g}" rgba="0.2 0.5 0.95 1"/></body>'
        for i in range(count)
    )
    text = f"""<mujoco model="beads">
  <worldbody>
{bodies}
  </worldbody>
</mujoco>
"""
    path = f"{get_cache_dir('g1_beaker_pour')}/beads_{count}_r{radius:g}_m{mass:g}.xml"
    write_atomically(path, text)
    return path


class HollowBeakerOptions(EntityOptions):
    """An open beaker (:func:`hollow_beaker_file`): ``radius`` and ``height`` outside, ``wall`` and ``bottom`` thick,
    ``sides`` wall panels, ``mass`` kg, a free body.

    Its MJCF is generated when the env builds the entity: ``file`` stays a name and :meth:`resolve_runtime_file`
    hands the builder the written file.
    """

    radius: float = _BEAKER_RADIUS
    height: float = _BEAKER_HEIGHT
    wall: float = _WALL
    bottom: float = _BOTTOM
    sides: int = _SIDES
    mass: float = _GLASS_MASS
    file: str = "hollow_beaker.xml"
    is_fixed_base: bool = False
    default_armature: float = 0.0

    @model_validator(mode="after")
    def _check_shape(self) -> HollowBeakerOptions:
        if not (
            0.0 < self.wall < self.radius and 0.0 < self.bottom < self.height and self.sides >= 6 and self.mass > 0
        ):
            raise ValueError(
                f"HollowBeakerOptions: need 0 < wall < radius, 0 < bottom < height, sides >= 6 and mass > 0; got "
                f"radius={self.radius}, height={self.height}, wall={self.wall}, bottom={self.bottom}, "
                f"sides={self.sides}, mass={self.mass}."
            )
        return self

    def resolve_runtime_file(self) -> str:
        return hollow_beaker_file(self.radius, self.height, self.wall, self.bottom, self.sides, self.mass)


class BeadsOptions(EntityOptions):
    """``count`` free spheres of ``radius`` and ``mass`` in one entity (:func:`beads_file`); its links are the beads.

    Its MJCF is generated when the env builds the entity: ``file`` stays a name and :meth:`resolve_runtime_file`
    hands the builder the written file.
    """

    count: int = _BEAD_COUNT
    radius: float = _BEAD_RADIUS
    mass: float = _BEAD_MASS
    file: str = "beads.xml"
    is_fixed_base: bool = False
    default_armature: float = 0.0

    @model_validator(mode="after")
    def _check_beads(self) -> BeadsOptions:
        if self.count < 1 or self.radius <= 0 or self.mass <= 0:
            raise ValueError(
                f"BeadsOptions: need count >= 1, radius > 0 and mass > 0; got count={self.count}, "
                f"radius={self.radius}, mass={self.mass}."
            )
        return self

    def resolve_runtime_file(self) -> str:
        return beads_file(self.count, self.radius, self.mass)


# ---------------------------------------------------------------------------
# Task terms
# ---------------------------------------------------------------------------


def _layer(inner_radius: float, bead_radius: float, gap: float) -> list[tuple[float, float]]:
    """The densest square grid of bead centres (pitch ``2 bead_radius + gap``) that fits a disc of ``inner_radius``,
    centred on a bead or on a gap, whichever holds more.
    """
    pitch = 2 * bead_radius + gap
    reach = inner_radius - bead_radius - gap
    best: list[tuple[float, float]] = []
    for shift in (0.0, 0.5):
        n = int(reach / pitch) + 1
        points = [
            ((i + shift) * pitch, (j + shift) * pitch)
            for i in range(-n, n + 1)
            for j in range(-n, n + 1)
            if math.hypot((i + shift) * pitch, (j + shift) * pitch) <= reach
        ]
        best = max(best, points, key=len)
    return best


@EVENT_TERM_REGISTRY.register()
class FillBeads(EventTermBase):
    """Stack every bead of ``beads_name`` (one free sphere a link, :class:`BeadsOptions`) in ``container_name`` at
    reset, at rest.

    In layers of a square grid on the container's floor (``floor_z``, its frame's z of the inner bottom), ``gap``
    apart and from the wall (``inner_radius``), each bead nudged by up to ``gap / 2`` so the pile does not settle
    symmetric. Run it after the events that place the container (a higher ``priority``): the beads follow the
    container's pose. The pile must fit under ``rim_z``.
    """

    mode: EventMode = EventMode.RESET
    beads_name: str = "beads"
    container_name: str = "beaker"
    bead_radius: float = 0.0
    inner_radius: float = 0.0
    floor_z: float = 0.0
    rim_z: float = 0.0
    gap: float = 0.001

    def build(self) -> None:
        beads = self.resolve_entity(self.beads_name)
        self.resolve_entity(self.container_name)
        n_beads = beads.num_links
        if beads.n_qs != 7 * n_beads:
            raise ValueError(
                f"{term_subject(self, self.beads_name)}: every link must be a free body (7 qs each), but the entity "
                f"has {beads.num_links} links and {beads.n_qs} qs. Use BeadsOptions for '{self.beads_name}'."
            )
        if not 0.0 < self.bead_radius < self.inner_radius:
            raise ValueError(
                f"{term_subject(self, self.beads_name)}: need 0 < bead_radius < inner_radius, got "
                f"bead_radius={self.bead_radius}, inner_radius={self.inner_radius}."
            )
        layer = _layer(self.inner_radius, self.bead_radius, self.gap)
        pitch = 2 * self.bead_radius + self.gap
        n_layers = math.ceil(n_beads / len(layer))
        top = self.floor_z + self.bead_radius + self.gap + (n_layers - 1) * pitch + self.bead_radius
        if top > self.rim_z:
            raise ValueError(
                f"{term_subject(self, self.beads_name)}: {n_beads} beads stack {n_layers} layers of {len(layer)} up to "
                f"z={top:.4f} in '{self.container_name}', above its rim_z={self.rim_z}. Use fewer or smaller beads."
            )
        local = [
            (x, y, self.floor_z + self.bead_radius + self.gap + k * pitch) for k in range(n_layers) for x, y in layer
        ][:n_beads]
        self._local = torch.tensor(local, device=self.device)
        self._identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self._qs_idx_local = range(beads.n_qs)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        container = self.resolve_entity(self.container_name)
        n_beads = len(self._local)
        nudge = (torch.rand(self.num_envs, n_beads, 3, device=self.device) - 0.5) * self.gap
        nudge[..., 2] = 0.0
        quat = container.get_quat().unsqueeze(1).expand(-1, n_beads, 4)
        pos = container.get_pos().unsqueeze(1) + transform_by_quat(self._local + nudge, quat)
        qpos = torch.cat([pos, self._identity.expand(self.num_envs, n_beads, 4)], dim=-1)
        self.resolve_entity(self.beads_name).set_qpos(
            qpos.reshape(self.num_envs, -1),
            qs_idx_local=self._qs_idx_local,
            envs_idx=_env_mask(envs_idx, self.num_envs, self.device),
            zero_velocity=True,
        )


def sample_beside(
    lo: torch.Tensor, hi: torch.Tensor, other: torch.Tensor, gap: float, side: torch.Tensor | float
) -> torch.Tensor:
    """Per row, a value drawn uniformly from ``[lo, hi]`` at least ``gap`` from ``other`` on its ``side`` (+1: above
    it, -1: below it). Where ``other`` stands too near the band's end for the gap, the value is ``other + side * gap``,
    past the band: the gap is kept.
    """
    side = torch.as_tensor(side, dtype=lo.dtype, device=lo.device).expand_as(lo)
    near = other + side * gap
    start = torch.where(side > 0, torch.maximum(near, lo), torch.minimum(near, hi))
    room = torch.where(side > 0, hi - start, start - lo).clamp(min=0.0)
    return start + side * torch.rand_like(lo) * room


@EVENT_TERM_REGISTRY.register()
class PlaceBeside(EventTermBase):
    """Place ``entity_name`` at reset on a uniformly drawn spot of its patch on one side of ``other_name``: at least
    ``min_gap`` m from the other's centre toward ``side_y`` (+1: toward +y, -1: toward -y), turned about z by a
    ``yaw_range`` draw.

    The patch is ``pos_x_range`` x ``pos_y_range`` of offsets from the entity's default root pose. Where the other
    stands too near the patch's end for the gap, the entity goes ``min_gap`` from it, past the patch (the gap is
    kept). Run it after the event that places ``other_name`` (a higher ``priority``). The entity must be one free body
    whose root qpos is its authored pose, as a :class:`HollowBeakerOptions` beaker is.
    """

    mode: EventMode = EventMode.RESET
    entity_name: str = ""
    other_name: str = ""
    pos_x_range: tuple[float, float] = (0.0, 0.0)
    pos_y_range: tuple[float, float] = (0.0, 0.0)
    min_gap: float = 0.0
    side_y: float = 1.0
    yaw_range: tuple[float, float] = (-math.pi, math.pi)

    def build(self) -> None:
        entity = self.resolve_entity(self.entity_name)
        self.resolve_entity(self.other_name)
        if entity.n_qs != 7 or not entity.root_qpos_is_authored_pose:
            raise ValueError(
                f"{term_subject(self, self.entity_name)}: must be one free body whose root qpos is its authored pose "
                f"(it has {entity.n_qs} qs, root_qpos_is_authored_pose={entity.root_qpos_is_authored_pose}), as a "
                f"HollowBeakerOptions beaker is; place anything else with events.reset_base_state_uniform."
            )
        if self.side_y not in (-1.0, 1.0) or self.min_gap < 0.0 or self.pos_y_range[1] < self.pos_y_range[0]:
            raise ValueError(
                f"{term_subject(self, self.entity_name)}: need side_y of 1.0 (toward +y) or -1.0, min_gap >= 0 (m) and "
                f"pos_y_range as (low, high); got side_y={self.side_y}, min_gap={self.min_gap}, "
                f"pos_y_range={self.pos_y_range}."
            )
        self._z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        entity = self.resolve_entity(self.entity_name)
        default_pos, default_quat = entity.get_default_root_pose()
        lo, hi = default_pos[:, 1] + self.pos_y_range[0], default_pos[:, 1] + self.pos_y_range[1]
        other_y = self.resolve_entity(self.other_name).get_pos()[:, 1]
        y = sample_beside(lo, hi, other_y, self.min_gap, self.side_y)
        x_yaw = sample_uniform_ranges((self.pos_x_range, self.yaw_range), self.num_envs, self.device)
        pos = torch.stack([default_pos[:, 0] + x_yaw[:, 0], y, default_pos[:, 2]], dim=-1)
        quat = quat_mul(quat_from_axis_angle(self._z_axis * x_yaw[:, 1:]), default_quat)
        entity.set_qpos(
            torch.cat([pos, quat], dim=-1),
            qs_idx_local=range(7),
            envs_idx=_env_mask(envs_idx, self.num_envs, self.device),
            zero_velocity=True,
        )


def _beads_in(
    env: EnvBase, beads_pos: torch.Tensor, container_name: str, inner_radius: float, floor_z: float, top_z: float
) -> torch.Tensor:
    """``(n_envs, n_beads)``: the beads whose centres are inside ``container_name``'s cavity, read in its frame: within
    ``inner_radius`` of its axis and between ``floor_z`` and ``top_z``.
    """
    container = env.get_entity(container_name)
    quat = container.get_quat().unsqueeze(1).expand(-1, beads_pos.shape[1], 4)
    local = inv_transform_by_quat(beads_pos - container.get_pos().unsqueeze(1), quat)
    radial = local[..., :2].norm(dim=-1)
    return (radial < inner_radius) & (local[..., 2] > floor_z) & (local[..., 2] < top_z)


@METRIC_TERM_REGISTRY.register(direction=MetricDirection.HIB, contributes_to_success=False)
def beads_inside(
    env: EnvBase, *, beads_name: str, container_name: str, inner_radius: float, floor_z: float, top_z: float
) -> torch.Tensor:
    """The share of ``beads_name``'s beads inside ``container_name`` (see :func:`_beads_in`)."""
    beads_pos = env.get_entity(beads_name).get_links_pos()
    return _beads_in(env, beads_pos, container_name, inner_radius, floor_z, top_z).float().mean(dim=-1)


@METRIC_TERM_REGISTRY.register(direction=MetricDirection.LIB, contributes_to_success=False)
def beads_spilled(
    env: EnvBase,
    *,
    beads_name: str,
    container_names: list[str],
    inner_radius: float,
    floor_z: float,
    top_z: float,
    below_z: float,
) -> torch.Tensor:
    """The share of ``beads_name``'s beads in none of ``container_names`` (all one shape) and lower than ``below_z``:
    on the bench or the floor. A bead in the air, falling, is not spilled yet.
    """
    beads_pos = env.get_entity(beads_name).get_links_pos()
    contained = torch.zeros(beads_pos.shape[:2], dtype=torch.bool, device=beads_pos.device)
    for name in container_names:
        contained |= _beads_in(env, beads_pos, name, inner_radius, floor_z, top_z)
    return (~contained & (beads_pos[..., 2] < below_z)).float().mean(dim=-1)


def pour_lip(env: EnvBase, *, source_name: str, target_name: str, radius: float, height: float) -> torch.Tensor:
    """``(n_envs, 3)``: the point of ``source_name``'s rim that beads leave over, in the world frame.

    That is the rim's lowest point. Upright, every rim point is lowest, and the one toward ``target_name`` is taken:
    a small pull toward the target breaks the tie, and from a few degrees of tilt the lowest point wins wherever it
    is. ``radius`` and ``height`` are the source's outside dimensions, its frame at its centre and its axis its z.
    """
    source, target = env.get_entity(source_name), env.get_entity(target_name)
    pos = source.get_pos()
    up = torch.zeros_like(pos)
    up[:, 2] = 1.0
    axis = transform_by_quat(up, source.get_quat())
    toward = target.get_pos() - pos
    toward[:, 2] = 0.0
    toward = toward / toward.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    pull = -up + 0.1 * toward
    across = pull - (pull * axis).sum(dim=-1, keepdim=True) * axis
    across = across / across.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return pos + axis * (height / 2) + across * radius


def _mouth(env: EnvBase, name: str, height: float) -> torch.Tensor:
    """``(n_envs, 3)``: the centre of ``name``'s rim (a beaker ``height`` tall, centred on its frame)."""
    entity = env.get_entity(name)
    up = torch.zeros(env.num_envs, 3, device=env.device)
    up[:, 2] = height / 2
    return entity.get_pos() + transform_by_quat(up, entity.get_quat())


@METRIC_TERM_REGISTRY.register(direction=MetricDirection.LIB, contributes_to_success=False)
def pour_lip_offset(env: EnvBase, *, source_name: str, target_name: str, radius: float, height: float) -> torch.Tensor:
    """How far the source's :func:`pour_lip` is from over the centre of the target's mouth, horizontally, in m."""
    lip = pour_lip(env, source_name=source_name, target_name=target_name, radius=radius, height=height)
    return (lip - _mouth(env, target_name, height))[:, :2].norm(dim=-1)


@METRIC_TERM_REGISTRY.register(direction=MetricDirection.HIB, contributes_to_success=False)
def pour_lip_height(env: EnvBase, *, source_name: str, target_name: str, radius: float, height: float) -> torch.Tensor:
    """How far the source's :func:`pour_lip` is above the target's rim, in m (negative below it)."""
    lip = pour_lip(env, source_name=source_name, target_name=target_name, radius=radius, height=height)
    return lip[:, 2] - _mouth(env, target_name, height)[:, 2]


@OBSERVATION_TERM_REGISTRY.register()
def beads_from_mouth(env: EnvBase, *, robot_name: str, beads_name: str, container_name: str, height: float):
    """The beads' centroid from the centre of ``container_name``'s rim (a beaker ``height`` tall), in the robot's
    base frame. (N, 3).
    """
    centroid = env.get_entity(beads_name).get_links_pos().mean(dim=1)
    mouth = _mouth(env, container_name, height)
    return _to_base_frame(env, robot_name, centroid) - _to_base_frame(env, robot_name, mouth)


@METRIC_TERM_REGISTRY.register(direction=MetricDirection.HIB, contributes_to_success=False)
def tilt_toward(env: EnvBase, *, object_name: str, toward_name: str) -> torch.Tensor:
    """How far ``object_name``'s axis (its local z) is tipped toward ``toward_name``, in rad: its tilt in the vertical
    plane through the object's centre toward the other entity's. Negative tipped away, 0 upright or tipped across,
    pi upside down.
    """
    obj = env.get_entity(object_name)
    pos = obj.get_pos()
    up = torch.zeros_like(pos)
    up[:, 2] = 1.0
    axis = transform_by_quat(up, obj.get_quat())
    toward = (env.get_entity(toward_name).get_pos() - pos)[:, :2]
    toward = toward / toward.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return torch.atan2((axis[:, :2] * toward).sum(dim=-1), axis[:, 2])


def _pouring(
    env: EnvBase, tilt_metric: str, lip_offset_metric: str, lip_height_metric: str, min_tilt: float, max_offset: float
) -> torch.Tensor:
    """Where the source pours into the target: tipped toward it past ``min_tilt`` rad (``tilt_metric``, a
    :func:`tilt_toward`), its pour lip within ``max_offset`` m of over the target's mouth (``lip_offset_metric``) and
    above its rim (``lip_height_metric``).
    """
    metrics = env.metric_manager
    return (
        (metrics.get_value(tilt_metric) > min_tilt)
        & (metrics.get_value(lip_offset_metric) < max_offset)
        & (metrics.get_value(lip_height_metric) > 0.0)
    )


@REWARD_TERM_REGISTRY.register()
class PourTilt(RewardTermBase):
    """How far the source is tipped toward the target (``tilt_metric``, a :func:`tilt_toward` in rad), as a share of
    upside down and none tipped across or away, paid while the source is held (gripped, ``grasp_metric``, or
    ``margin`` m clear of the bench, ``clearance_metric``) and its pour lip is above the target's rim
    (``lip_height_metric`` > 0), weighted by ``exp(-(offset / sigma)^2)`` of how far the lip is from over the target's
    mouth (``lip_offset_metric``).

    The way into pouring, which pays only once beads leave the source past 90 degrees. On the plain tilt, a beaker
    tipped across paid nearly what one tipped toward the target did, and a policy tipped it 20-30 degrees sideways
    for good. Linear in the angle, so the first 30 degrees pay a sixth of upside down; ``1 - cos`` would pay them a
    fifteenth.
    """

    tilt_metric: str = ""
    lip_offset_metric: str = ""
    lip_height_metric: str = ""
    clearance_metric: str = ""
    grasp_metric: str = ""
    margin: float = 0.01
    sigma: float = 0.03

    def build(self) -> None:
        for name in (
            self.tilt_metric,
            self.lip_offset_metric,
            self.lip_height_metric,
            self.clearance_metric,
            self.grasp_metric,
        ):
            resolve_metric_manager(self._env, term=type(self).__name__, metric_name=name)
        if self.sigma <= 0.0:
            raise ValueError(f"{type(self).__name__}: sigma (m) must be > 0, got {self.sigma}.")

    def compute(self) -> torch.Tensor:
        metrics = self._env.metric_manager
        held = _held(
            self._env, clearance_metric=self.clearance_metric, grasp_metric=self.grasp_metric, margin=self.margin
        )
        over = torch.exp(-(metrics.get_value(self.lip_offset_metric) / self.sigma).square())
        above = metrics.get_value(self.lip_height_metric) > 0.0
        toward = metrics.get_value(self.tilt_metric).clamp(min=0.0)
        return toward / math.pi * over * (held & above)


@REWARD_TERM_REGISTRY.register()
class HeldTiltOffTarget(RewardTermBase):
    """How far the held object leans other than toward the target, squared (negative weight): the horizontal part of
    its axis (its local z) across the line to ``target_name``, and along it away from the target. Paid while the
    object is held, as :class:`~eden.tasks.manipulation.g1_beaker_lift.config.HeldObjectTilt` gates it.

    The lift's upright cost, with the pour's direction left free. With no cost on the tilt at all, a policy drifted the
    beaker 20-30 degrees sideways, where the lip swings away from the target and the pour shaping pays less.
    """

    object_name: str = "object"
    target_name: str = ""
    clearance_metric: str = ""
    grasp_metric: str = ""
    margin: float = 0.01

    def build(self) -> None:
        self.resolve_entity(self.object_name)
        self.resolve_entity(self.target_name)
        for name in (self.clearance_metric, self.grasp_metric):
            resolve_metric_manager(self._env, term=type(self).__name__, metric_name=name)

    def compute(self) -> torch.Tensor:
        obj = self.resolve_entity(self.object_name)
        pos = obj.get_pos()
        up = torch.zeros_like(pos)
        up[:, 2] = 1.0
        lean = transform_by_quat(up, obj.get_quat())[:, :2]
        toward = (self.resolve_entity(self.target_name).get_pos() - pos)[:, :2]
        toward = toward / toward.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        along = (lean * toward).sum(dim=-1)
        across = lean[:, 0] * toward[:, 1] - lean[:, 1] * toward[:, 0]
        held = _held(
            self._env, clearance_metric=self.clearance_metric, grasp_metric=self.grasp_metric, margin=self.margin
        )
        return (across.square() + along.clamp(max=0.0).square()) * held


@REWARD_TERM_REGISTRY.register()
class UprightedGoalDistance(RewardTermBase):
    """:func:`~eden.tasks.manipulation.custom_terms.rewards.object_goal_distance` (``1 - tanh(d / std)``, ``deadband``
    m flat, paid while the object is lifted and in the hand), with ``d`` measured to where the object's centre would be
    turned back upright about the pivot: the point ``pivot_local_pos`` of ``pivot_link_name``, the IK's control point.
    While the object pours, it pays in full.

    A held beaker tips about the hand, and its centre swings with it, about 6 cm per radian with the G1's grasp: to the
    goal as it is, tipping it lost more goal reward up to 60 degrees than tipping it paid. Turned back upright about
    the palm, a beaker tipped about the palm is where it was, so tipping neither costs nor pays; an upright beaker is
    measured as in the lift. A deadband that widened with the tilt instead paid for any tilt: its policy tipped the
    beaker 40 degrees sideways, nowhere near the target, for the slack.

    Pouring is the object tipped toward the target past ``pour_min_tilt`` rad (``tilt_metric``, a :func:`tilt_toward`
    in rad) with its pour lip within
    ``pour_max_offset`` m of over the target's mouth (``lip_offset_metric``) and above its rim (``lip_height_metric``).
    The goal is where a pour starts, not where it must stay: the palm moves as the arm turns the beaker over, and
    an outward pour, whose palm sagged off the goal at the elbow's and wrist's limits, lost a scripted pour 12 of the
    goal's 35 a step by 80 degrees.
    Leave the three metric names empty for no pour credit.
    """

    robot_name: str = "robot"
    gripper_name: str = "robot"
    object_name: str = "object"
    command_name: str = ""
    grasp_links: list[str] = []
    approach_offset: float = 0.0
    rest_z: float = 0.0
    lift_margin: float = 0.02
    grasp_threshold: float = 0.06
    std: float = 0.1
    deadband: float = 0.0
    pivot_link_name: str = ""
    pivot_local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tilt_metric: str = ""
    lip_offset_metric: str = ""
    lip_height_metric: str = ""
    pour_min_tilt: float = 0.35
    pour_max_offset: float = 0.05

    def build(self) -> None:
        robot = self.resolve_entity(self.robot_name)
        self.resolve_entity(self.object_name)
        gripper = self.resolve_entity(self.gripper_name)
        resolve_link_idx(gripper, self.grasp_links, self.device, term=type(self).__name__)
        self._env.command_manager.get_term(self.command_name)
        if not self.pivot_link_name:
            raise ValueError(
                f"{term_subject(self, self.object_name)}: pivot_link_name is empty; name the link the object is tipped about, e.g. "
                f"the IK's ee_link_name, with its control point as pivot_local_pos."
            )
        self._pivot_link = resolve_link_idx(
            robot, [self.pivot_link_name], self.device, term=type(self).__name__, field="pivot_link_name"
        )
        if len(self._pivot_link) != 1:
            raise ValueError(
                f"{term_subject(self, self.object_name)}: pivot_link_name={self.pivot_link_name!r} must name one link of "
                f"'{self.robot_name}', but it matches {len(self._pivot_link)}."
            )
        if self.std <= 0.0 or self.deadband < 0.0:
            raise ValueError(
                f"{term_subject(self, self.object_name)}: need std > 0 and deadband >= 0 (m); got std={self.std}, deadband={self.deadband}."
            )
        pour_metrics = (self.tilt_metric, self.lip_offset_metric, self.lip_height_metric)
        if any(pour_metrics) and not all(pour_metrics):
            raise ValueError(
                f"{term_subject(self, self.object_name)}: tilt_metric, lip_offset_metric and lip_height_metric credit a pour together; "
                f"got {pour_metrics}. Name all three (e.g. 'beaker_tilt', 'pour_lip_offset', 'pour_lip_height'), or "
                f"none for no pour credit."
            )
        for name in pour_metrics if all(pour_metrics) else ():
            resolve_metric_manager(self._env, term=type(self).__name__, metric_name=name)
        self._pour_credit = all(pour_metrics)
        self._pivot_local = torch.tensor([self.pivot_local_pos], device=self.device)
        self._up = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)

    def compute(self) -> torch.Tensor:
        obj, robot = self.resolve_entity(self.object_name), self.resolve_entity(self.robot_name)
        centre = obj.get_pos()
        up = self._up.expand_as(centre)
        axis = transform_by_quat(up, obj.get_quat())
        # The turn that stands the axis back up: about axis x up, by the tilt.
        across = torch.cross(axis, up, dim=-1)
        tilt = torch.acos(axis[:, 2:3].clamp(-1.0, 1.0))
        upright = quat_from_axis_angle(across * (tilt / across.norm(dim=-1, keepdim=True).clamp(min=1e-6)))
        link_quat = robot.get_links_quat(ls_idx_local=self._pivot_link)[:, 0]
        pivot = robot.get_links_pos(ls_idx_local=self._pivot_link)[:, 0] + transform_by_quat(
            self._pivot_local.expand_as(centre), link_quat
        )
        centre_upright = pivot + transform_by_quat(centre - pivot, upright)
        goal = _base_to_world(self._env, self.robot_name, self._env.command_manager.get_term(self.command_name).command)
        dist = ((goal - centre_upright).norm(dim=-1) - self.deadband).clamp(min=0.0)
        gated = _lifted_and_close(
            self._env,
            self.gripper_name,
            self.object_name,
            self.grasp_links,
            self.rest_z,
            self.lift_margin,
            self.grasp_threshold,
            self.approach_offset,
        )
        value = 1.0 - torch.tanh(dist / self.std)
        if self._pour_credit:
            pouring = _pouring(
                self._env,
                self.tilt_metric,
                self.lip_offset_metric,
                self.lip_height_metric,
                self.pour_min_tilt,
                self.pour_max_offset,
            )
            value = torch.where(pouring, torch.ones_like(value), value)
        return gated * value


@COMMAND_TERM_REGISTRY.register()
class AnchoredT3Command(UniformT3Command):
    """:class:`UniformT3Command` drawn around an entity: ``anchor_name``'s position in the frame of
    ``marker_frame_entity`` (the command's frame), plus an offset drawn from ``x_range``, ``y_range`` and ``z_range``.

    Drawn at the anchor's pose when the command resamples, which at a reset follows the events that place the anchor.
    """

    anchor_name: str = ""

    def build(self) -> None:
        if not self.anchor_name or not self.marker_frame_entity:
            raise ValueError(
                f"{type(self).__name__}: anchor_name and marker_frame_entity must both name entities (got "
                f"{self.anchor_name!r} and {self.marker_frame_entity!r}): the goal is drawn around the first, in the "
                f"second's frame, e.g. anchor_name='target', marker_frame_entity='robot'."
            )
        self.resolve_entity(self.anchor_name)
        self.resolve_entity(self.marker_frame_entity)

    def _resample_command(self, envs_idx: torch.Tensor) -> None:
        anchor = _to_base_frame(self._env, self.marker_frame_entity, self.resolve_entity(self.anchor_name).get_pos())
        offset = sample_uniform_ranges((self.x_range, self.y_range, self.z_range), self.num_envs, self.device)
        self.oracle_pos_b.copy_(torch.where(envs_idx.unsqueeze(-1), anchor + offset, self.oracle_pos_b))


# ---------------------------------------------------------------------------
# Putting the source back
# ---------------------------------------------------------------------------


@COMMAND_TERM_REGISTRY.register()
class PourThenPlaceCommand(AnchoredT3Command):
    """:class:`AnchoredT3Command`'s goal by the target until the source is poured, then the way back to the source's
    own place on the bench, where ``object_name`` stood when the episode started (:attr:`home_b`, in the command's
    frame): first ``approach_height`` m over it, then, once the source is there, on it.

    The source is poured once no more than ``max_beads_left`` of ``beads_name``'s beads are inside ``object_name``
    (its cavity, ``inner_radius``, ``floor_z`` and ``top_z``, as :func:`beads_inside` reads it); it is over its home
    once, poured, its centre is within ``approach_tolerance`` m of the point over the home, or of the home itself if
    it got there another way, and it is tilted less than ``approach_max_tilt`` rad. :attr:`poured` and :attr:`over_home` stay set for the rest of the episode. They grade
    the step that set them, so they are set in :meth:`post_physics_refresh`, ahead of the rewards; the goal the
    policy is given switches when the command is published, for the next step.

    Over the home first: sent straight to it, the pour policy took the tipped beaker down through the target and
    knocked it over in 34 % of the episodes, within half a second; sent 12 cm over it, in none, and it carried the
    beaker to 3.4 cm of the point (median), as the lift it started from carries one to a goal.

    Simulation only: :meth:`post_physics_refresh` does not run on hardware, where both stay False, and the goal by
    the target, until something else sets them.
    """

    object_name: str = ""
    beads_name: str = ""
    inner_radius: float = 0.0
    floor_z: float = 0.0
    top_z: float = 0.0
    max_beads_left: int = 1
    approach_height: float = 0.12
    approach_tolerance: float = 0.03
    approach_max_tilt: float = math.radians(15.0)

    def __init__(self, env: EnvBase, options) -> None:
        super().__init__(env, options)
        self.poured = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.over_home = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.home_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_b = torch.zeros(self.num_envs, 3, device=self.device)

    def build(self) -> None:
        super().build()
        self.resolve_entity(self.object_name)
        n_beads = self.resolve_entity(self.beads_name).get_links_pos().shape[1]
        if self.inner_radius <= 0.0 or self.top_z <= self.floor_z:
            raise ValueError(
                f"{type(self).__name__}: the source's cavity needs inner_radius > 0 and top_z > floor_z (m, in its "
                f"frame); got inner_radius={self.inner_radius}, floor_z={self.floor_z}, top_z={self.top_z}."
            )
        if not 0 <= self.max_beads_left < n_beads:
            raise ValueError(
                f"{type(self).__name__}: max_beads_left must be from 0 to {n_beads - 1} ('{self.beads_name}' has "
                f"{n_beads} beads), got {self.max_beads_left}."
            )
        if self.approach_height < 0.0 or self.approach_tolerance <= 0.0 or not 0.0 < self.approach_max_tilt < math.pi:
            raise ValueError(
                f"{type(self).__name__}: need approach_height >= 0 and approach_tolerance > 0 (m), and "
                f"approach_max_tilt in (0, pi) rad; got {self.approach_height}, {self.approach_tolerance}, "
                f"{self.approach_max_tilt}."
            )
        self._up = torch.tensor([[0.0, 0.0, self.approach_height]], device=self.device)
        self._max_lean = 1.0 - math.cos(self.approach_max_tilt)

    @property
    def command(self) -> torch.Tensor:
        return self._goal_b

    @property
    def over_home_b(self) -> torch.Tensor:
        """The point ``approach_height`` over the home, in the command's frame (its z is up)."""
        return self.home_b + self._up

    def distance_to_go(self, pos_w: torch.Tensor) -> torch.Tensor:
        """How far ``pos_w`` (``(n_envs, 3)``, world) is from the home along the way back: down from the point over it
        and to that point until the source has been there (:attr:`over_home`), then straight to the home.
        Continuous where the way turns down.
        """
        home = _base_to_world(self._env, self.marker_frame_entity, self.home_b)
        over = _base_to_world(self._env, self.marker_frame_entity, self.over_home_b)
        via = (pos_w - over).norm(dim=-1) + self.approach_height
        return torch.where(self.over_home, (pos_w - home).norm(dim=-1), via)

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        idx = slice(None) if envs_idx is None else envs_idx
        self.poured[idx] = False
        self.over_home[idx] = False
        return super().reset(envs_idx)

    def post_reset(self, envs_idx: slice | torch.Tensor) -> None:
        home = _to_base_frame(self._env, self.marker_frame_entity, self.resolve_entity(self.object_name).get_pos())
        assign_over_envs(self.home_b, envs_idx, home)
        self._publish()

    def post_physics_refresh(self) -> None:
        beads_pos = self.resolve_entity(self.beads_name).get_links_pos()
        left = _beads_in(self._env, beads_pos, self.object_name, self.inner_radius, self.floor_z, self.top_z).sum(-1)
        pos = self.resolve_entity(self.object_name).get_pos()
        near = torch.minimum(
            (pos - _base_to_world(self._env, self.marker_frame_entity, self.over_home_b)).norm(dim=-1),
            (pos - _base_to_world(self._env, self.marker_frame_entity, self.home_b)).norm(dim=-1),
        )
        upright = object_tilt(self._env, object_name=self.object_name) < self._max_lean
        self.over_home |= self.poured & (near < self.approach_tolerance) & upright
        self.poured |= left <= self.max_beads_left

    def _resample_command(self, envs_idx: torch.Tensor) -> None:
        super()._resample_command(envs_idx)
        self._publish()

    def _update_command(self) -> None:
        self._publish()

    def _publish(self) -> None:
        back = torch.where(self.over_home.unsqueeze(-1), self.home_b, self.over_home_b)
        torch.where(self.poured.unsqueeze(-1), back, self.oracle_pos_b, out=self._goal_b)


def _pour_command(env: EnvBase, term: str, command_name: str) -> PourThenPlaceCommand:
    """The :class:`PourThenPlaceCommand` ``command_name``, for ``term`` (named in the error)."""
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, PourThenPlaceCommand):
        raise TypeError(
            f"{term}: command_name={command_name!r} is a {type(command).__name__}, but it must name a "
            f"PourThenPlaceCommand, whose poured latch and home this term reads (the pour task's 'lift_target')."
        )
    return command


@OBSERVATION_TERM_REGISTRY.register()
def pour_done(env: EnvBase, *, command_name: str) -> torch.Tensor:
    """``(n_envs, 1)``: 1.0 once the source is poured (:attr:`PourThenPlaceCommand.poured`), when the goal it is given
    is its home rather than by the target.
    """
    return _pour_command(env, "pour_done", command_name).poured.float().unsqueeze(-1)


@METRIC_TERM_REGISTRY.register()
class PourStage(MetricTermBase):
    """1.0 once the source has reached ``stage`` of :class:`PourThenPlaceCommand` ``command_name``: ``"poured"``, or
    ``"over_home"`` on its way back.
    """

    command_name: str = ""
    stage: str = "poured"

    def build(self) -> None:
        self._command = _pour_command(self._env, type(self).__name__, self.command_name)
        if self.stage not in ("poured", "over_home"):
            raise ValueError(f"{type(self).__name__}: stage must be 'poured' or 'over_home', got {self.stage!r}.")

    def compute(self) -> torch.Tensor:
        return getattr(self._command, self.stage).float()


def _home_w(env: EnvBase, command: PourThenPlaceCommand) -> torch.Tensor:
    return _base_to_world(env, command.marker_frame_entity, command.home_b)


@METRIC_TERM_REGISTRY.register()
class HomeDistance(MetricTermBase):
    """How far ``object_name``'s centre is from its home (:attr:`PourThenPlaceCommand.home_b`), in m."""

    command_name: str = ""
    object_name: str = ""

    def build(self) -> None:
        self._command = _pour_command(self._env, term_subject(self, self.object_name), self.command_name)
        self.resolve_entity(self.object_name)

    def compute(self) -> torch.Tensor:
        return (self.resolve_entity(self.object_name).get_pos() - _home_w(self._env, self._command)).norm(dim=-1)


@METRIC_TERM_REGISTRY.register()
class BackAtHome(MetricTermBase):
    """1.0 where the source, poured, stands back at its home: its centre within ``max_offset`` m of the home across
    and ``max_height`` m of it up or down (resting on the bench), and tilted less than ``max_tilt`` rad. With
    ``released``, nothing of ``entity_name`` touches it either (under ``max_force`` N of contact).
    """

    command_name: str = ""
    object_name: str = ""
    max_offset: float = 0.02
    max_height: float = 0.01
    max_tilt: float = math.radians(10.0)
    released: bool = False
    entity_name: str = "robot"
    max_force: float = 0.05

    def build(self) -> None:
        self._command = _pour_command(self._env, term_subject(self, self.object_name), self.command_name)
        self.resolve_entity(self.object_name)
        if self.released:
            self.resolve_entity(self.entity_name)
        if min(self.max_offset, self.max_height, self.max_tilt, self.max_force) <= 0.0:
            raise ValueError(
                f"{term_subject(self, self.object_name)}: max_offset, max_height (m), max_tilt (rad) and max_force (N) must be > 0; got "
                f"{self.max_offset}, {self.max_height}, {self.max_tilt}, {self.max_force}."
            )

    def compute(self) -> torch.Tensor:
        command = self._command
        obj = self.resolve_entity(self.object_name)
        off = obj.get_pos() - _home_w(self._env, command)
        upright = object_tilt(self._env, object_name=self.object_name) < 1.0 - math.cos(self.max_tilt)
        back = (
            command.poured & (off[:, :2].norm(dim=-1) < self.max_offset) & (off[:, 2].abs() < self.max_height) & upright
        )
        if self.released:
            force = self.resolve_entity(self.entity_name).get_grouped_contacts(obj)["force_norm_sum"].sum(dim=(1, 2))
            back &= force < self.max_force
        return back.float()


@REWARD_TERM_REGISTRY.register()
class PourPhaseGate(RewardTermBase):
    """``term`` (another reward term's options), paid only before the source is poured, or with ``after_pour`` only
    after (:attr:`PourThenPlaceCommand.poured` of ``command_name``).

    The weight is this term's: ``term`` keeps weight 1.0 and no ``range_s``, which the reward manager would never
    apply to it.
    """

    term: RewardTermOptions | None = None
    command_name: str = ""
    after_pour: bool = False

    def build(self) -> None:
        if not isinstance(self.term, RewardTermOptions):
            raise TypeError(
                f"{type(self).__name__}: term must be a reward term's options (from <Term>.configure(...)), got "
                f"{type(self.term).__name__}."
            )
        if self.term.weight != 1.0 or self.term.range_s is not None:
            raise ValueError(
                f"{type(self).__name__}: the gated term '{self.term.name}' has weight={self.term.weight} and "
                f"range_s={self.term.range_s}, which only the gate's own fields apply. Give the gate the weight and "
                f"the range, and the term weight 1.0 and no range: PourPhaseGate.configure(weight=w, "
                f"term=term.updated(weight=1.0), ...)."
            )
        self._command = _pour_command(self._env, type(self).__name__, self.command_name)
        self._term = build_term(REWARD_TERM_REGISTRY, self._env, self.term, label=f"{type(self).__name__}.term")

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        self._term.reset(envs_idx)

    def post_reset(self, envs_idx: slice | torch.Tensor) -> None:
        self._term.post_reset(envs_idx)

    def compute(self) -> torch.Tensor:
        open_ = self._command.poured if self.after_pour else ~self._command.poured
        return self._term.compute() * open_


@REWARD_TERM_REGISTRY.register()
class UprightHomeDistance(RewardTermBase):
    """``1 - tanh(d / std)`` of how far the source has to go back to its home (``d``,
    :meth:`PourThenPlaceCommand.distance_to_go`: over it, then down), times how upright it is (the cosine of its
    tilt, none past 90 degrees), paid once it is poured.

    Along the way rather than to the home, so it pays the same just before and just after the goal turns down, and
    reaching the point over the home loses nothing. Upright, so a beaker dropped on its side by its home, or put
    down on its rim, earns nothing.
    """

    command_name: str = ""
    object_name: str = ""
    std: float = 0.1

    def build(self) -> None:
        self._command = _pour_command(self._env, term_subject(self, self.object_name), self.command_name)
        self.resolve_entity(self.object_name)
        if self.std <= 0.0:
            raise ValueError(f"{term_subject(self, self.object_name)}: std (m) must be > 0, got {self.std}.")

    def compute(self) -> torch.Tensor:
        command = self._command
        d = command.distance_to_go(self.resolve_entity(self.object_name).get_pos())
        upright = (1.0 - object_tilt(self._env, object_name=self.object_name)).clamp(min=0.0)
        return (1.0 - torch.tanh(d / self.std)) * upright * command.poured


@REWARD_TERM_REGISTRY.register()
class DofsAtDefault(RewardTermBase):
    """``1 - tanh(d / std)`` of how far ``dofs_name``'s joints are from their default positions within their limits,
    where each episode starts them: ``d`` the norm over them, in rad, less ``deadband``. Paid where the metric
    ``gate_metric`` reads 1.

    Within the limits: the G1's default elbow (-1.5 rad) is past its lower limit (-1.05), where the arm starts. The
    deadband is the arm's sag at rest: under no action the elbow sits 0.18 rad off the limit, 0.20 rad in all.
    """

    entity_name: str = "robot"
    dofs_name: list[str] = []
    std: float = 0.5
    deadband: float = 0.0
    gate_metric: str = ""

    def build(self) -> None:
        entity = self.resolve_entity(self.entity_name)
        if not self.dofs_name:
            raise ValueError(
                f"{term_subject(self)}: dofs_name is empty; name the joints to bring back, e.g. "
                f"dofs_name={entity.dofs_name[:2]}."
            )
        if self.std <= 0.0 or self.deadband < 0.0:
            raise ValueError(
                f"{term_subject(self)}: need std > 0 and deadband >= 0 (rad); got std={self.std}, "
                f"deadband={self.deadband}."
            )
        resolve_metric_manager(self._env, term=type(self).__name__, metric_name=self.gate_metric)
        self._dofs = entity.find_named_dofs(self.dofs_name)
        lower, upper = entity.get_dofs_limit(dofs_idx_local=self._dofs.dofs_idx_local)
        self._default = torch.clamp(entity.get_default_dofs_pos(self._dofs), lower, upper)

    def compute(self) -> torch.Tensor:
        q = self.resolve_entity(self.entity_name).get_dofs_pos(dofs_idx_local=self._dofs.dofs_idx_local)
        value = 1.0 - torch.tanh(((q - self._default).norm(dim=-1) - self.deadband).clamp(min=0.0) / self.std)
        return value * self._env.metric_manager.get_value(self.gate_metric)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_LIFT = ManipulationG1BeakerLiftConfig.model_fields
#: The beakers' cavity, for the bead counts: the target's reaches 1 cm over its rim, so a bead falling in counts on
#: its way down.
_CAVITY = {"inner_radius": _INNER_RADIUS, "floor_z": _FLOOR_Z - 0.001, "top_z": _RIM_Z}
_LIP = {"source_name": "beaker", "target_name": "target", "radius": _BEAKER_RADIUS, "height": _BEAKER_HEIGHT}
#: Pouring: tipped toward the target past 20 degrees, the lip within 6 cm of over it and above it. The goal terms pay
#: in full then, and the hand may come near the target.
_POURING = {
    "tilt_metric": "beaker_tilt_toward_target",
    "lip_offset_metric": "pour_lip_offset",
    "lip_height_metric": "pour_lip_height",
    "pour_min_tilt": math.radians(20.0),
    "pour_max_offset": 0.06,
}
#: The inputs the actor gets beyond the lift's: whether the source is poured (:func:`pour_done`).
ACTOR_EXTRA_DIMS = 1
#: The inputs the critic gets beyond the lift's: three bead shares, the beads' centroid from the target's mouth, and
#: whether the source is poured.
CRITIC_EXTRA_DIMS = 7
#: The pour task's goal, by the target and then the source's home.
_COMMAND = "lift_target"


def _pour_scene() -> SceneOptions:
    return _bench_scene(
        robot=_POUR_ROBOT,
        beaker=HollowBeakerOptions(
            default_root_pos=_BEAKER_HOME, material=RigidMaterialOptions(friction=_GLASS_FRICTION)
        ),
        target=HollowBeakerOptions(
            default_root_pos=_TARGET_HOME, material=RigidMaterialOptions(friction=_GLASS_FRICTION)
        ),
        beads=BeadsOptions(material=RigidMaterialOptions(friction=_BEAD_FRICTION)),
    )


def _pour_observations() -> ObservationManagerOptions:
    options = _LIFT["observation_options"].default.clone(unlock=True)
    critic = options.critic
    # Appended after the lift's, in this order, so a lift checkpoint widens by zero columns at the end.
    for name in ("beads_in_source", "beads_in_target", "beads_spilled"):
        setattr(critic, name, observations.MetricObservation.configure(metric_name=name))
    # In decimetres, so it is of the order of the shares.
    critic.beads_from_mouth = ObsTerm.configure(
        func=beads_from_mouth,
        params={"robot_name": "robot", "beads_name": "beads", "container_name": "target", "height": _BEAKER_HEIGHT},
        scale=10.0,
    )
    # Last in both groups: whether the goal is by the target, to pour at, or the source's home, to put it down at.
    for group in (options.policy, critic):
        group.pour_done = ObsTerm.configure(func=pour_done, params={"command_name": _COMMAND})
    return options


def _pour_metrics() -> MetricManagerOptions:
    options = _LIFT["metric_options"].default.clone(unlock=True)
    options.target_tilt = MetricTerm.configure(func=object_tilt, params={"object_name": "target"})
    options.beaker_tilt_toward_target = MetricTerm.configure(
        func=tilt_toward, params={"object_name": "beaker", "toward_name": "target"}
    )
    options.beads_in_source = MetricTerm.configure(
        func=beads_inside, params={"beads_name": "beads", "container_name": "beaker", **_CAVITY}
    )
    options.beads_in_target = MetricTerm.configure(
        func=beads_inside, params={"beads_name": "beads", "container_name": "target", **_CAVITY, "top_z": _RIM_Z + 0.01}
    )
    options.beads_spilled = MetricTerm.configure(
        func=beads_spilled,
        params={
            "beads_name": "beads",
            "container_names": ["beaker", "target"],
            **_CAVITY,
            "below_z": _BENCH_TOP + 2.5 * _BEAD_RADIUS,
        },
    )
    options.pour_lip_offset = MetricTerm.configure(func=pour_lip_offset, params=_LIP)
    options.pour_lip_height = MetricTerm.configure(func=pour_lip_height, params=_LIP)
    home = {"command_name": _COMMAND, "object_name": "beaker", "contributes_to_success": False}
    options.source_poured = PourStage.configure(command_name=_COMMAND, contributes_to_success=False)
    options.source_over_home = PourStage.configure(
        command_name=_COMMAND, stage="over_home", contributes_to_success=False
    )
    options.home_distance = HomeDistance.configure(**home)
    # Within 2 cm of where it stood, resting, and within 10 degrees of upright; then with the hand off it.
    options.source_placed = BackAtHome.configure(**home)
    options.source_released = BackAtHome.configure(**home, released=True, entity_name="robot")
    return options


def _pour_rewards() -> RewardManagerOptions:
    options = _LIFT["reward_options"].default.clone(unlock=True)
    # Pouring tips the beaker and turns the wrist at the goal.
    options.disable_term("beaker_upright")
    options.disable_term("settle_at_goal")
    # The lift's goal tracking, measured to the beaker turned back upright about the palm (the IK's control point).
    goal = {
        "robot_name": "robot",
        **_GRASP,
        **_LIFT_GATE,
        "command_name": "lift_target",
        "pivot_link_name": _EE_LINK,
        "pivot_local_pos": _PALM_IN_WRIST,
        **_POURING,
    }
    options.object_goal_tracking = UprightedGoalDistance.configure(weight=25.0, **goal, std=0.3)
    options.object_goal_tracking_fine = UprightedGoalDistance.configure(weight=10.0, **goal, std=0.05, deadband=0.02)
    # Clearance is to the beaker's lowest point, which drops as it tips: capped at 10 cm, it pays the same carried
    # and pouring, where at the lift's 25 cm tipping the beaker to 100 degrees cost 1.8 a step.
    options.lift_height = options.lift_height.updated(cap=0.10)
    options.beaker_ang_vel = options.beaker_ang_vel.updated(weight=-0.1)
    # Upright, or tipped toward the target: leaning across or away costs 20 sin^2 of it, 3.6 a step at 25 degrees,
    # twice the lift's upright cost there (under which the lift's policy still held the beaker 10 degrees over).
    options.beaker_tilt_off_target = HeldTiltOffTarget.configure(
        weight=-20.0,
        object_name="beaker",
        target_name="target",
        clearance_metric="beaker_clearance",
        grasp_metric="beaker_grasped",
        margin=0.01,
    )
    # The lift's forearm and hand on the torso, and now the upper arm and shoulder on it and the arm on the left one.
    options.arm_on_torso = options.arm_on_torso.updated(
        links_name=[*_UPPER_ARM, *_FOREARM_AND_HAND], other_links_name=["torso_link", *_LEFT_ARM]
    )
    # The lift's smoothness costs at the weights its curriculum ends at.
    options.action_rate = options.action_rate.updated(weight=-0.5)
    options.arm_joint_vel = options.arm_joint_vel.updated(weight=-0.5)
    # Up to 30 a step with every bead in, as much as the lift and the goal pay together.
    options.beads_in_target = rewards.MetricReward.configure(metric_name="beads_in_target", weight=30.0)
    options.beads_spilled = rewards.MetricReward.configure(metric_name="beads_spilled", weight=-30.0)
    # Tipped toward the target only: up to 60 a step upside down over it, 30 on its side, 10 at 30 degrees, against
    # some 40 for holding the beaker at the goal. At 25, tipping 20 degrees toward the target paid 1.5 a step, too
    # little to steer the policy. At the goal upright, the lip is 9-13 cm from over the target: sigma 10 cm lets a
    # first tip toward the target pay. At 4 cm a beaker had to be 60 degrees over before the term paid anything.
    options.pour_tilt = PourTilt.configure(
        weight=60.0,
        tilt_metric="beaker_tilt_toward_target",
        lip_offset_metric="pour_lip_offset",
        lip_height_metric="pour_lip_height",
        clearance_metric="beaker_clearance",
        grasp_metric="beaker_grasped",
        margin=0.01,
        sigma=0.10,
    )

    # Once the source is poured it goes back where it stood: what reached, held, lifted, carried and tipped it pays
    # only before (each would pay for holding on), and the terms below only after.
    for name in _PICK_AND_POUR_TERMS:
        term = getattr(options, name)
        setattr(
            options,
            name,
            PourPhaseGate.configure(weight=term.weight, term=term.updated(weight=1.0), command_name=_COMMAND),
        )
    # Poured: as much a step as the pour paid at most held tipped over the target (pour_tilt, the goal and the
    # lift's terms, about 90), so putting the source back never pays less than holding it there, short of a bead.
    options.source_poured = rewards.MetricReward.configure(metric_name="source_poured", weight=100.0)
    home = {"command_name": _COMMAND, "object_name": "beaker"}
    options.home_tracking = UprightHomeDistance.configure(weight=25.0, **home, std=0.3)
    options.home_tracking_fine = UprightHomeDistance.configure(weight=10.0, **home, std=0.05)
    options.source_placed = rewards.MetricReward.configure(metric_name="source_placed", weight=20.0)
    options.source_released = rewards.MetricReward.configure(metric_name="source_released", weight=20.0)
    # The arm back at its start, only while the source stands where the hand let it go: a retreat that knocks it pays
    # nothing (playing the pick-up back in reverse, the hand pushed the beaker 4.6 cm off its home, median).
    options.arm_at_rest = DofsAtDefault.configure(
        weight=15.0, entity_name="robot", dofs_name=_ARM_DOFS, std=0.5, deadband=0.25, gate_metric="source_released"
    )
    return options


#: The lift's and the pour's terms that pay only until the source is poured (:class:`PourPhaseGate`).
_PICK_AND_POUR_TERMS = (
    "reaching_object",
    "reaching_object_fine",
    "fingertip_to_object",
    "hand_closure",
    "beaker_grasped",
    "lift_height",
    "lifting_object",
    "object_goal_tracking",
    "object_goal_tracking_fine",
    "beaker_tilt_off_target",
    "pour_tilt",
)


def _pour_terminations() -> TerminationManagerOptions:
    options = _LIFT["termination_options"].default.clone(unlock=True)
    # The source may tip: what it must not do is empty onto the bench.
    options.disable_term("beaker_spilled")
    options.beads_spilled = terminations.MetricAbove.configure(metric_name="beads_spilled", threshold=0.5)
    options.target_tipped = terminations.MetricAbove.configure(metric_name="target_tilt", threshold=_SPILL_TILT)
    return options


def _pour_events() -> EventManagerOptions:
    options = _LIFT["event_options"].default.clone(unlock=True)
    # A beaker moved toward the target would land on it, and the pre-grasp start puts the source at its home, a
    # target's width from the target; the lift checkpoint grasps already.
    options.disable_term("move_beaker")
    options.disable_term("pregrasp_start")
    place = options.place_beaker
    options.place_beaker = place.updated(
        params={**place.params, "pos_x_range": _SOURCE_PLACE_X, "pos_y_range": _SOURCE_PLACE_Y}
    )
    # After the source is placed.
    options.place_target = PlaceBeside.configure(
        priority=1,
        entity_name="target",
        other_name="beaker",
        pos_x_range=_TARGET_PLACE_X,
        pos_y_range=_TARGET_PLACE_Y,
        min_gap=_MIN_GAP_Y,
        side_y=1.0,
    )
    # After the source is placed.
    options.fill_beads = FillBeads.configure(
        priority=2,
        beads_name="beads",
        container_name="beaker",
        bead_radius=_BEAD_RADIUS,
        inner_radius=_INNER_RADIUS,
        floor_z=_FLOOR_Z,
        rim_z=_RIM_Z,
        gap=0.001,
    )
    # The glass alone, from a lighter one to a heavier one; the beads are the rest of the load.
    options.beaker_mass = options.beaker_mass.updated(mass_shift_range=(-0.03, 0.05))
    # The lift's stiff contact on the target and the beads too: a pair takes the mean of its geoms' parameters, so
    # beads on glass and on each other are as stiff as the hand on the source.
    stiffen = options.stiffen_contact
    links = stiffen.params["entity_links"]
    options.stiffen_contact = stiffen.updated(
        params={**stiffen.params, "entity_links": {**links, "target": None, "beads": None}}
    )
    return options


@TASK_REGISTRY.register()
class ManipulationG1BeakerPourConfig(ManipulationG1BeakerLiftConfig):
    env_options: EnvOptions = _LIFT["env_options"].default.updated(
        episode_length_s=_EPISODE_S,
        # Room for 5x this many candidate contacts and 12x as many broad-phase pairs an env. Genesis counts the
        # candidates before pruning them, and a count past the room halts the run. The beads at rest in the source
        # make about 20; a beaker on its rim makes far more, 5 for each of its 16 wall panels: 80 for the source
        # put down upside down on the bench, 184 for it upside down on the target's rim (199 in all, the most in
        # 8M env-steps of a trained policy acting with exploration noise). At 30, a run halted after 100M. Room
        # costs memory, not speed: at 4096 envs 5.9 GB at 30 and 9.8 GB at 80, at the same 16.8k env-steps/s.
        max_collision_pairs=80,
        # The finger pads' spin resistance (``_PAD_TORSION`` on the hand's links): one more constraint row a contact.
        enable_torsional_friction=True,
    )

    scene_options: SceneOptions = _pour_scene()

    observation_options: ObservationManagerOptions = _pour_observations()

    command_options: CommandManagerOptions = CommandManagerOptions(
        lift_target=PourThenPlaceCommand.configure(
            anchor_name="target",
            # In the pelvis frame, from the target's centre.
            x_range=_GOAL_FROM_TARGET_X,
            y_range=_GOAL_FROM_TARGET_Y,
            z_range=_GOAL_FROM_TARGET_Z,
            resampling_time_range=(2 * _EPISODE_S, 2 * _EPISODE_S),
            debug_vis=True,
            marker_frame_entity="robot",
            marker_entity_name="goal_marker",
            # Poured with at most one bead left in the source; then the goal is 12 cm over where it stood, and once
            # it is there, within 3 cm and 15 degrees of upright, where it stood.
            object_name="beaker",
            beads_name="beads",
            **_CAVITY,
            max_beads_left=1,
            approach_height=0.12,
            approach_tolerance=0.03,
            approach_max_tilt=math.radians(15.0),
        )
    )

    metric_options: MetricManagerOptions = _pour_metrics()

    reward_options: RewardManagerOptions = _pour_rewards()

    termination_options: TerminationManagerOptions = _pour_terminations()

    event_options: EventManagerOptions = _pour_events()

    curriculum_options: CurriculumManagerOptions = CurriculumManagerOptions()

    runner_options: RslRlOnPolicyRunnerOptions = _LIFT["runner_options"].default.updated(
        experiment_name="g1_beaker_pour", run_name="v3"
    )


_LEFT = mirror_config(ManipulationG1BeakerPourConfig())


@TASK_REGISTRY.register()
class ManipulationG1BeakerPourLeftConfig(ManipulationG1BeakerPourConfig):
    """:class:`ManipulationG1BeakerPourConfig` for the left hand (``manipulation_g1_beaker_pour_left``), reflected
    through the robot's sagittal plane by :func:`~eden.tasks.manipulation.g1_beaker_pour.mirror.mirror_config`, with
    its observations and actions in the right hand's frame: a right-hand checkpoint drives it as it is. Not trained.
    """

    scene_options: SceneOptions = _LEFT.scene_options
    observation_options: ObservationManagerOptions = _LEFT.observation_options
    action_options: ActionManagerOptions = _LEFT.action_options
    command_options: CommandManagerOptions = _LEFT.command_options
    metric_options: MetricManagerOptions = _LEFT.metric_options
    reward_options: RewardManagerOptions = _LEFT.reward_options
    termination_options: TerminationManagerOptions = _LEFT.termination_options
    event_options: EventManagerOptions = _LEFT.event_options
    runner_options: RslRlOnPolicyRunnerOptions = _LEFT.runner_options
