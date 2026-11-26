"""G1 beaker pour (``manipulation_g1_beaker_pour``): the standing G1 of ``manipulation_g1_beaker_lift`` picks up a
beaker of beads and pours them into a second beaker on the bench.

:class:`ManipulationG1BeakerPourConfig` subclasses
:class:`~eden.tasks.manipulation.g1_beaker_lift.config.ManipulationG1BeakerLiftConfig`: the robot, the bench, the
actions and the policy's observations are the lift's, so a lift checkpoint drives this task as it is. What changes:

* **Both beakers are hollow, and the beads are rigid spheres.** A beaker is an open polygonal cylinder of wall
  panels on a bottom disc (:class:`HollowBeakerOptions`, generated MJCF), the lift's beaker's size and, with its
  beads, its mass. The beads (:class:`BeadsOptions`) stand in for a liquid: eight free spheres in one entity,
  stacked in the source beaker at every reset (:class:`FillBeads`). They leave it past about 100 degrees of tilt.
  Eight, not more: each bead is six DOFs in every constraint solve, and 20 beads trained 7x slower.
* **Everything that touches collides.** The beads with each other, both beakers, the bench and the hand; the beakers
  with each other, the hand and the bench. The lift's stiff contact on the source covers the target and the beads,
  and ``max_collision_pairs`` has three times the room of the most contacts measured (a count past it halts the
  run). Pouring, spilling and noised actions in 768 envs never put a bead's centre inside a beaker's glass.
* **The hand pours outward, supinating.** The target starts on the right, the source nearer the middle. Turned palm
  up, the beaker lies in the hand, and the lift checkpoint's grasp held it to 130 degrees in a scripted pour; turned
  palm down (pouring inward), the beaker hangs under the hand, and it dropped every one by 90 degrees.
* **The goal is where the beaker's centre is while it pours.** :class:`AnchoredT3Command` draws it 4-6 cm from the
  target's axis toward the source and 12-14 cm above the target's centre, and the actor, which never observes the
  target, finds it at a fixed offset from the goal. The lift checkpoint holds a beaker's centre on its goal and,
  tipped, turns it about that centre, so from there the lip stays over the target's mouth from upright to about 120
  degrees, past where the beads pour, and tipping toward the target pays from the first degree. With the goal 11-14 cm off (where the swing of a beaker
  tipped about the hand would bring the lip over the target), the lift policy pulled the swinging beaker back to
  the goal, the lip never neared the target, and a bias toward the target on its wrist lost return over an
  episode; from here it gains 25 at a small bias. The goal terms measure the beaker turned back upright about the
  palm (:class:`UprightedGoalDistance`), so tipping it about the hand neither costs nor pays either, and pay in
  full once it pours (its lip over the target), since a supinating arm's palm sags off the goal at its joint limits.
* **The beaker may tip toward the target.** The lift's upright cost becomes a cost on leaning across or away from
  the target (:class:`HeldTiltOffTarget`), and its tilt termination and settle-at-goal cost are gone; the held
  beaker's angular velocity costs a fifth as much, and the lift-height reward is capped at 10 cm of clearance,
  which a tipped beaker held at the goal keeps. What pays is the share of beads in the target
  (``beads_in_target``), and what costs is the share spilled onto the bench (``beads_spilled``, which ends the
  episode past half), and so does knocking the target over. Getting there, :class:`PourTilt` pays the beaker's tilt
  toward the target (:func:`tilt_toward`) while its pour lip (the lowest point of its rim, :func:`pour_lip`) is near
  over the target's mouth and above its rim.
* **The critic counts the beads.** It gets the lift's critic observations and, after them, the shares in the
  source, in the target and spilled, and the beads' centroid from the target's mouth
  (:data:`CRITIC_EXTRA_DIMS` more inputs). A lift checkpoint is widened for them with
  :mod:`eden.tasks.manipulation.g1_beaker_pour.warm_start`: the new inputs' weights start at zero, so the critic
  starts out as the lift's.
* **Every episode starts with the hand raised.** No beaker is moved mid-episode, the pre-grasp start is gone (it
  puts the source at its home, a beaker's width from the target), and so is the goal-range curriculum (the goal
  follows the target). The smoothness costs sit at their final weights from the start.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from pydantic import model_validator

from eden import EventMode, events, observations, rewards, terminations
from eden.constants import MetricDirection
from eden.managers import COMMAND_TERM_REGISTRY, METRIC_TERM_REGISTRY, OBSERVATION_TERM_REGISTRY, REWARD_TERM_REGISTRY
from eden.managers.event_manager import EVENT_TERM_REGISTRY
from eden.managers.event_manager import EventTerm as EventTermBase
from eden.managers.reward_manager import RewardTerm as RewardTermBase
from eden.managers.terms.commands.oracle import UniformT3Command
from eden.managers.terms.utils import resolve_link_idx, resolve_metric_manager
from eden.options.entities import EntityOptions
from eden.options.materials import RigidMaterialOptions
from eden.options.learning.rsl_rl import RslRlOnPolicyRunnerOptions
from eden.tasks.manipulation.custom_terms.common import _base_to_world, _to_base_frame
from eden.tasks.manipulation.custom_terms.rewards import _lifted_and_close
from eden.tasks.manipulation.g1_beaker_lift.config import (
    _BEAKER_HEIGHT,
    _BEAKER_HOME,
    _BEAKER_RADIUS,
    _BENCH_TOP,
    _EE_LINK,
    _GRASP,
    _LIFT_GATE,
    _PALM_IN_WRIST,
    _REST_Z,
    _SPILL_TILT,
    ManipulationG1BeakerLiftConfig,
    _bench_scene,
    _env_mask,
    _held,
    object_tilt,
)
from eden.tasks.registry import TASK_REGISTRY
from eden.terms import EventTerm, MetricTerm, ObsTerm
from eden.utils.configs import (
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
from eden.utils.geom import inv_transform_by_quat, quat_from_axis_angle, transform_by_quat
from eden.utils.misc import get_cache_dir
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

# The source starts nearer the middle (y -0.14 to -0.04, offsets from its home), the target on the right (y -0.34 to
# -0.26): 5 cm of bench at least between them, and the right hand pours toward -y, supinating.
_SOURCE_PLACE_X = (-0.04, 0.04)
_SOURCE_PLACE_Y = (0.04, 0.14)
_TARGET_HOME = (0.30, -0.30, _REST_Z)
_TARGET_PLACE = (-0.04, 0.04)
#: Where the goal is drawn, from the target's centre: where the source's centre is while it pours. 4-6 cm toward
#: the source, so the lip of a beaker tipped about its centre stays over the target's mouth from upright to about
#: 120 degrees, past where the beads pour, and 12-14 cm up, so the lip stays 1.5 cm or more over the target's rim and the upright
#: source's bottom 2.5 cm or more.
_GOAL_FROM_TARGET_X = (-0.01, 0.01)
_GOAL_FROM_TARGET_Y = (0.04, 0.06)
_GOAL_FROM_TARGET_Z = (0.12, 0.14)

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
            raise ValueError(f"{term_subject(self)}: sigma (m) must be > 0, got {self.sigma}.")

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
    The goal is where a pour starts, not where it must stay: supinating that far takes the G1's elbow and wrist to
    their limits, and its palm sags off the goal, which cost a scripted pour 12 of the goal's 35 a step by 80 degrees.
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
                f"{term_subject(self)}: pivot_link_name is empty; name the link the object is tipped about, e.g. "
                f"the IK's ee_link_name, with its control point as pivot_local_pos."
            )
        self._pivot_link = resolve_link_idx(
            robot, [self.pivot_link_name], self.device, term=type(self).__name__, field="pivot_link_name"
        )
        if len(self._pivot_link) != 1:
            raise ValueError(
                f"{term_subject(self)}: pivot_link_name={self.pivot_link_name!r} must name one link of "
                f"'{self.robot_name}', but it matches {len(self._pivot_link)}."
            )
        if self.std <= 0.0 or self.deadband < 0.0:
            raise ValueError(
                f"{term_subject(self)}: need std > 0 and deadband >= 0 (m); got std={self.std}, deadband={self.deadband}."
            )
        pour_metrics = (self.tilt_metric, self.lip_offset_metric, self.lip_height_metric)
        if any(pour_metrics) and not all(pour_metrics):
            raise ValueError(
                f"{term_subject(self)}: tilt_metric, lip_offset_metric and lip_height_metric credit a pour together; "
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
            metrics = self._env.metric_manager
            pouring = (
                (metrics.get_value(self.tilt_metric) > self.pour_min_tilt)
                & (metrics.get_value(self.lip_offset_metric) < self.pour_max_offset)
                & (metrics.get_value(self.lip_height_metric) > 0.0)
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
                f"{term_subject(self)}: anchor_name and marker_frame_entity must both name entities (got "
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
# Config
# ---------------------------------------------------------------------------

_LIFT = ManipulationG1BeakerLiftConfig.model_fields
#: The beakers' cavity, for the bead counts: the target's reaches 1 cm over its rim, so a bead falling in counts on
#: its way down.
_CAVITY = {"inner_radius": _INNER_RADIUS, "floor_z": _FLOOR_Z - 0.001, "top_z": _RIM_Z}
_LIP = {"source_name": "beaker", "target_name": "target", "radius": _BEAKER_RADIUS, "height": _BEAKER_HEIGHT}
#: The inputs the critic gets beyond the lift's: three bead shares and the beads' centroid from the target's mouth.
CRITIC_EXTRA_DIMS = 6


def _pour_scene() -> SceneOptions:
    return _bench_scene(
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
        # Paid in full while pouring: tipped toward the target past 20 degrees, the lip within 6 cm of over it and
        # above it.
        "tilt_metric": "beaker_tilt_toward_target",
        "lip_offset_metric": "pour_lip_offset",
        "lip_height_metric": "pour_lip_height",
        "pour_min_tilt": math.radians(20.0),
        "pour_max_offset": 0.06,
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
    return options


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
    options.place_target = EventTerm.configure(
        func=events.reset_base_state_uniform,
        mode=EventMode.RESET,
        params={
            "entity_name": "target",
            "pos_x_range": _TARGET_PLACE,
            "pos_y_range": _TARGET_PLACE,
            "pos_z_range": (0.0, 0.0),
            "yaw_range": (-3.14159, 3.14159),
        },
    )
    # After the source is placed, and after the pre-grasp start moves it (priority 1).
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
        # Room for 5x this many contacts and 12x as many broad-phase pairs an env. The beads at rest in the source
        # make about 30 contacts; pouring, spilling and noised actions in 768 envs peaked at 39 contacts and 61
        # pairs. A count past the room halts the run, and every pair of room costs constraint rows over all 103
        # DOFs (at 200 with 20 beads, the Jacobian alone took 12 GB at 4096 envs).
        max_collision_pairs=30,
    )

    scene_options: SceneOptions = _pour_scene()

    observation_options: ObservationManagerOptions = _pour_observations()

    command_options: CommandManagerOptions = CommandManagerOptions(
        lift_target=AnchoredT3Command.configure(
            anchor_name="target",
            # In the pelvis frame, from the target's centre.
            x_range=_GOAL_FROM_TARGET_X,
            y_range=_GOAL_FROM_TARGET_Y,
            z_range=_GOAL_FROM_TARGET_Z,
            resampling_time_range=(2 * _EPISODE_S, 2 * _EPISODE_S),
            debug_vis=True,
            marker_frame_entity="robot",
            marker_entity_name="goal_marker",
        )
    )

    metric_options: MetricManagerOptions = _pour_metrics()

    reward_options: RewardManagerOptions = _pour_rewards()

    termination_options: TerminationManagerOptions = _pour_terminations()

    event_options: EventManagerOptions = _pour_events()

    curriculum_options: CurriculumManagerOptions = CurriculumManagerOptions()

    runner_options: RslRlOnPolicyRunnerOptions = _LIFT["runner_options"].default.updated(
        experiment_name="g1_beaker_pour", run_name="v1"
    )
