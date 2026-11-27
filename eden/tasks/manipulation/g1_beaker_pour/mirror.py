"""The G1 beaker pour reflected for the other hand: :func:`mirror_config`, which builds
``manipulation_g1_beaker_pour_left`` (``ManipulationG1BeakerPourLeftConfig`` in
:mod:`eden.tasks.manipulation.g1_beaker_pour.config`) from the right hand's task, driven by the right hand's policy
as it is.

The welded G1, its bench and both beakers are mirror-symmetric about the pelvis frame's xz-plane, so the left hand's
pour is the right hand's reflected (``y -> -y``), and so is its policy. The left task is not trained: its observations
and actions are expressed in the right hand's frame, so a checkpoint of ``manipulation_g1_beaker_pour`` drives it::

    uv run eden inference --task manipulation_g1_beaker_pour_left --checkpoint <a manipulation_g1_beaker_pour model>

:func:`mirror_config` builds it from the right hand's config, field by field:

* **Names.** Every ``right_*`` link, joint and pattern becomes ``left_*`` and the other way round. The robot's own
  ``dofs_name`` keeps its order.
* **Joint values** (the default pose, the grasp closure's targets, wherever a reward term nests them). A joint about x or z (a roll, a yaw, and every
  Dex3 joint but the thumb's base) turns the other way in the mirror, so its value changes sign; one about y (a pitch,
  the knee, the elbow) keeps it (:func:`joint_sign`).
* **Positions and ranges in y.** Every scene entity's default y, every ``y_range`` / ``pos_y_range`` and yaw range,
  and every ``side_y``.
* **Observations.** Each policy and critic term is scaled per component by the reflection's signs
  (:data:`OBSERVATION_SIGNS`, :func:`joint_sign` for joint terms), so the left hand's terms read in a mirrored state
  what the right hand's read in the original. The last action is the policy's own output, already in the right
  hand's frame.
* **Actions.** The IK's y step and its x and z turns change sign (negative axis scales); every Dex3 joint that the
  reflection turns the other way does too (:class:`SignedImplicitPDController`); the elbow is kept clear on the +y side.

A term added to the right task's observations must be added to :data:`OBSERVATION_SIGNS`, or :func:`mirror_config`
raises.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from eden.managers import ACTION_TERM_REGISTRY
from eden.managers.terms.actions.joint_actions import ImplicitPDController
from eden.utils.configs import serialize_obj_with_metadata
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.utils.configs import EdenConfig

#: Joint-name endings of the joints about x or z in their parent's frame: reflected through the xz-plane, they turn
#: the other way. The Dex3's finger joints and the thumb's two outer ones turn about z; the thumb's base about y.
_FLIPPED_JOINT_ENDINGS = (
    "_roll_joint",
    "_yaw_joint",
    "_thumb_1_joint",
    "_thumb_2_joint",
    "_index_0_joint",
    "_index_1_joint",
    "_middle_0_joint",
    "_middle_1_joint",
)

_Y = [1.0, -1.0, 1.0]
#: An angular velocity is an axial vector: the reflection keeps its y and flips its x and z.
_AXIAL = [-1.0, 1.0, -1.0]

#: Per-component signs of each observation term that is not a joint term (those take :func:`joint_sign` per joint),
#: ``None`` for one the reflection leaves as it is.
OBSERVATION_SIGNS: dict[str, list[float] | None] = {
    "object_position": _Y,
    # Rotation 6D, the first two rows of R: the reflection takes R to M R M (M = diag(1, -1, 1)), which flips the
    # entries with exactly one index on y.
    "object_orientation": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
    "target_object_position": _Y,
    "grasp_point": _Y,
    "fingertips_pos": _Y * 3,
    "last_action": None,
    "fingertips_to_object": _Y * 3,
    "object_lin_vel": _Y,
    "object_ang_vel": _AXIAL,
    "beads_in_source": None,
    "beads_in_target": None,
    "beads_spilled": None,
    "beads_from_mouth": _Y,
    "pour_done": None,
}


def mirror_name(name: str) -> str:
    """``right_x`` <-> ``left_x``; any other name as it is."""
    for side, other in (("right_", "left_"), ("left_", "right_")):
        if name.startswith(side):
            return other + name.removeprefix(side)
    return name


def joint_sign(name: str) -> float:
    """-1 for a joint the reflection through the xz-plane turns the other way (about x or z), else 1."""
    return -1.0 if name.endswith(_FLIPPED_JOINT_ENDINGS) else 1.0


@ACTION_TERM_REGISTRY.register()
class SignedImplicitPDController(ImplicitPDController):
    """:class:`ImplicitPDController` that moves each DOF ``dofs_sign`` names (``{pattern: -1.0}``) the other way: its
    scale times the sign. The DOFs no key matches keep theirs.

    For a hand mirrored from the one its policy was trained on: the same action closes it, though its finger joints
    turn the other way.
    """

    dofs_sign: dict[str, float] = {}

    def _build_scale_and_offset(self) -> None:
        super()._build_scale_and_offset()
        bad = {pattern: sign for pattern, sign in self.dofs_sign.items() if sign not in (-1.0, 1.0)}
        if bad:
            raise ValueError(
                f"{term_subject(self)}: dofs_sign values must be -1.0 or 1.0, got {bad}; scale the action with "
                f"scale or scale_ratio instead."
            )
        self._scale = self._scale * self._dofs_dict_to_tensor("dofs_sign", self.dofs_sign, fill=1.0)


def _swap_sides(value: Any) -> Any:
    if isinstance(value, dict):
        return {mirror_name(k) if isinstance(k, str) else k: _swap_sides(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_swap_sides(v) for v in value]
    return mirror_name(value) if isinstance(value, str) else value


def _mirror_ranges(value: Any) -> None:
    """Mirror, in place, every y range and yaw range under ``value`` (``(lo, hi)`` -> ``(-hi, -lo)``) and every side in
    y (``side_y`` -> ``-side_y``).
    """
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("y_range", "pos_y_range", "yaw_range", "roll_range") and isinstance(item, list):
                value[key] = [-item[1], -item[0]]
            elif key == "side_y" and isinstance(item, int | float):
                value[key] = -item
            else:
                _mirror_ranges(item)
    elif isinstance(value, list):
        for item in value:
            _mirror_ranges(item)


def _sign_joint_values(value: Any, key: str) -> None:
    """Sign, in place, every ``key`` (a joint name -> value map) under ``value`` by :func:`joint_sign`, gated terms'
    nested ones too.
    """
    if isinstance(value, dict):
        for name, item in value.items():
            if name == key and isinstance(item, dict):
                value[name] = {dof: joint_sign(dof) * pos for dof, pos in item.items()}
            else:
                _sign_joint_values(item, key)
    elif isinstance(value, list):
        for item in value:
            _sign_joint_values(item, key)


def _mirror_observations(group: dict[str, Any], group_name: str) -> None:
    for name, term in group.items():
        if not isinstance(term, dict) or "name" not in term:
            continue
        if "dofs_name" in term:
            signs = [joint_sign(dof) for dof in term["dofs_name"]]
        elif name in OBSERVATION_SIGNS:
            signs = OBSERVATION_SIGNS[name]
        else:
            raise KeyError(
                f"observation_options.{group_name}.{name}: no mirror signs. Add its per-component signs under the "
                f"reflection y -> -y to OBSERVATION_SIGNS in {__name__} (None if it does not change), e.g. "
                f"OBSERVATION_SIGNS[{name!r}] = [1.0, -1.0, 1.0] for a position."
            )
        if signs is None:
            continue
        scale = term["scale"]
        scales = scale if isinstance(scale, list) else [scale] * len(signs)
        if len(scales) != len(signs):
            raise ValueError(
                f"observation_options.{group_name}.{name}: {len(signs)} mirror signs for its {len(scales)} scales "
                f"({scale}); fix OBSERVATION_SIGNS[{name!r}] in {__name__} to one sign per component."
            )
        term["scale"] = [s * sign for s, sign in zip(scales, signs, strict=True)]


def mirror_config(config: EdenConfig) -> EdenConfig:
    """The pour task ``config`` reflected through the robot's xz-plane, for the other hand, with its observations and
    actions in the original hand's frame (see the module docstring). Of ``config``'s class.
    """
    # JSON's own types (lists, not tuples), with every option's class recorded for the load.
    data = json.loads(json.dumps(serialize_obj_with_metadata(config)))
    robot = data["scene_options"]["robot"]
    # The robot's DOF and kept-link order is its own, not a side's.
    kept = {key: robot[key] for key in ("dofs_name", "links_to_keep")}
    data = _swap_sides(data)
    robot = data["scene_options"]["robot"]
    robot.update(kept)
    robot["default_dofs_pos"] = {dof: joint_sign(dof) * pos for dof, pos in robot["default_dofs_pos"].items()}
    _sign_joint_values(data["reward_options"], "closed_dofs_pos")

    for entity in data["scene_options"].values():
        if isinstance(entity, dict) and "default_root_pos" in entity:
            x, y, z = entity["default_root_pos"]
            entity["default_root_pos"] = [x, -y, z]
            w, qx, qy, qz = entity["default_root_quat"]
            entity["default_root_quat"] = [w, -qx, qy, -qz]
    for section in ("event_options", "command_options"):
        _mirror_ranges(data[section])

    ik = data["action_options"]["arm_ik"]
    position, orientation = ik["position_scale"], ik["orientation_scale"]
    ik["position_scale"] = [position, -position, position]
    ik["orientation_scale"] = [-orientation, orientation, -orientation]
    ik["elbow_side"] = -ik["elbow_side"]
    hand = data["action_options"]["hand"]
    hand["name"] = "signed_implicit_pd_controller"
    hand["dofs_sign"] = {dof: -1.0 for dof in hand["dofs_name"] if joint_sign(dof) < 0}

    for group_name in ("policy", "critic"):
        _mirror_observations(data["observation_options"][group_name], group_name)
    data["runner_options"]["experiment_name"] = f"{data['runner_options']['experiment_name']}_left"
    return type(config).load_from_data(data, source=f"{type(config).__name__}, mirrored")
