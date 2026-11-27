"""Both hands at the bench, the beaker poured toward the middle by the hand on its outer side::

    uv run python -m eden.tasks.manipulation.g1_beaker_pour.bimanual --checkpoint <a manipulation_g1_beaker_pour model> --vis

The right hand's pour policy pours with either hand: the left hand's observations and actions are those of the
mirrored task (``manipulation_g1_beaker_pour_left``), in the right hand's frame, so one policy drives both.
:func:`bimanual_config` puts both on one robot: each hand's IK and Dex3 actions (13 each, the right hand's first),
each hand's actor observations (``policy_right``, and ``policy_left`` mirrored) with its own last action, and each
hand's goal beside the target on the side that hand pours from. Both arms collide with the body and each other, and
both hands' pads resist spin.

Each hand pours inward, toward the middle, so it takes a source that stands on the target's side of it: the rule
(:func:`left_hand_for`) gives the source to the right hand when it stands right of the target (lower y), to the left
hand otherwise. Each episode :class:`ChooseHand` lays the bench out as a single-hand task does, on either half: the
source on that half, the target toward the middle from it (at least 12 cm across, within reach of the hand on that
side), and gives the source to the hand the rule picks. That arm starts at the ready pose and the other hangs at
its default. A target further out than the source, on its side of the bench, is left to no hand: the grasp closes
round the source from its outer side, where such a target is in the way (see the pour task's docstring).
:func:`run` acts with the chosen hand only; the other's zero action holds it still. The runner prints, per hand, the
share of beads poured in, spilled and left in the source, the share of sources put back where they stood and let go
of, and how many episodes ended early.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from typing import TYPE_CHECKING, Any

import torch

from eden import EventMode
from eden.managers.event_manager import EVENT_TERM_REGISTRY
from eden.managers.event_manager import EventTerm as EventTermBase
from eden.tasks.manipulation.g1_beaker_lift.config import _ARM_DOFS, _HAND_DOFS, _env_mask
from eden.tasks.manipulation.g1_beaker_pour.config import (
    _BEAKER_HOME,
    _MIN_GAP_Y,
    _SOURCE_PLACE_X,
    _SOURCE_PLACE_Y,
    _TARGET_HOME,
    _TARGET_PLACE_X,
    _TARGET_PLACE_Y,
    ManipulationG1BeakerPourConfig,
    sample_beside,
)
from eden.tasks.manipulation.g1_beaker_pour.mirror import joint_sign, mirror_config, mirror_name
from eden.utils.configs import serialize_obj_with_metadata
from eden.utils.geom import quat_from_axis_angle, quat_mul
from eden.utils.sample import sample_uniform_ranges
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase

#: The hands, in the order of their action columns.
HANDS = ("right", "left")


def left_hand_for(source_y: torch.Tensor, target_y: torch.Tensor) -> torch.Tensor:
    """Which hand pours (True for the left): the one on the source's outer side, which pours toward the target, inward.
    The right hand for a source right of the target (lower y, pelvis frame), the left one for a source left of it.
    """
    return source_y > target_y


@EVENT_TERM_REGISTRY.register()
class ChooseHand(EventTermBase):
    """At reset, lay the bench out as a single-hand pour task does, on a half drawn at random, and give the source to
    the hand :func:`left_hand_for` picks. :attr:`left` holds each env's hand (True for the left).

    On the right half, ``source_name`` stands at ``source_x_range`` x ``right_source_y_range`` (offsets from its
    default position) and ``target_name`` at ``target_x_range`` x ``right_target_y_range`` (offsets from its own),
    toward +y from the source, at least ``min_gap`` across (:func:`~eden.tasks.manipulation.g1_beaker_pour.config.sample_beside`);
    on the left half, the mirror. Both turn about z at random. The chosen hand's arm (``right_arm_dofs``, or their
    mirror) starts at ``ready_pos`` and the other hangs at ``hang_pos``, both given for the right arm. Run it before
    the event that fills the source (``priority``).
    """

    mode: EventMode = EventMode.RESET
    robot_name: str = "robot"
    source_name: str = ""
    target_name: str = ""
    right_arm_dofs: list[str] = []
    ready_pos: list[float] = []
    hang_pos: list[float] = []
    source_x_range: tuple[float, float] = (0.0, 0.0)
    right_source_y_range: tuple[float, float] = (0.0, 0.0)
    target_x_range: tuple[float, float] = (0.0, 0.0)
    right_target_y_range: tuple[float, float] = (0.0, 0.0)
    min_gap: float = 0.0

    def build(self) -> None:
        robot = self.resolve_entity(self.robot_name)
        n = len(self.right_arm_dofs)
        if not n or len(self.ready_pos) != n or len(self.hang_pos) != n:
            raise ValueError(
                f"{term_subject(self)}: ready_pos and hang_pos need one value per joint of right_arm_dofs; got "
                f"{len(self.ready_pos)} and {len(self.hang_pos)} for {n} joints ({self.right_arm_dofs})."
            )
        for name in (self.source_name, self.target_name):
            entity = self.resolve_entity(name)
            if entity.n_qs != 7 or not entity.root_qpos_is_authored_pose:
                raise ValueError(
                    f"{term_subject(self, name)}: must be one free body whose root qpos is its authored pose (it has "
                    f"{entity.n_qs} qs), as a HollowBeakerOptions beaker is."
                )
        if self.min_gap < 0.0:
            raise ValueError(f"{term_subject(self)}: min_gap (m) must be >= 0, got {self.min_gap}.")
        self._dofs = robot.find_named_dofs([*self.right_arm_dofs, *(mirror_name(dof) for dof in self.right_arm_dofs)])
        signs = torch.tensor([joint_sign(dof) for dof in self.right_arm_dofs], device=self.device)
        ready = torch.tensor(self.ready_pos, device=self.device)
        hang = torch.tensor(self.hang_pos, device=self.device)
        # Both arms' joints (the right arm's, then the left's), a row per chosen hand.
        self._arms = torch.stack([torch.cat([ready, hang * signs]), torch.cat([hang, ready * signs])])
        self._z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)
        self.left = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _place(self, name: str, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> None:
        entity = self.resolve_entity(name)
        default_pos, default_quat = entity.get_default_root_pose()
        yaw = sample_uniform_ranges(((-math.pi, math.pi),), self.num_envs, self.device)
        pos = torch.stack([default_pos[:, 0] + x, default_pos[:, 1] + y, default_pos[:, 2]], dim=-1)
        quat = quat_mul(quat_from_axis_angle(self._z_axis * yaw), default_quat)
        entity.set_qpos(torch.cat([pos, quat], dim=-1), qs_idx_local=range(7), envs_idx=mask, zero_velocity=True)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        mask = _env_mask(envs_idx, self.num_envs, self.device)
        # +1 on the right half, where the target goes toward +y from the source; -1 on the left half, the mirror.
        side = torch.where(torch.rand(self.num_envs, device=self.device) < 0.5, 1.0, -1.0)
        draws = sample_uniform_ranges(
            (self.source_x_range, self.right_source_y_range, self.target_x_range), self.num_envs, self.device
        )
        source_y = side * draws[:, 1]
        low, high = self.right_target_y_range
        lo, hi = torch.where(side > 0, low, -high), torch.where(side > 0, high, -low)
        target_y = sample_beside(lo, hi, source_y, self.min_gap, side)
        self._place(self.source_name, draws[:, 0], source_y, mask)
        self._place(self.target_name, draws[:, 2], target_y, mask)
        self.left = torch.where(mask, left_hand_for(source_y, target_y), self.left)
        self.resolve_entity(self.robot_name).set_dofs_pos(
            self._arms[self.left.long()], dofs_idx_local=self._dofs.dofs_idx_local, envs_idx=mask, zero_velocity=True
        )


def _data(config) -> dict[str, Any]:
    """``config`` as JSON data, every option's class recorded for the load."""
    return json.loads(json.dumps(serialize_obj_with_metadata(config)))


def _is_term(value: Any) -> bool:
    return isinstance(value, dict) and "name" in value


def _hand_group(group: dict[str, Any], hand: str) -> dict[str, Any]:
    """A pour task's actor group for ``hand`` on the bimanual robot: its own goal and pour latch (every term that reads
    a command), and its own last action (its arm's, then its hand's: the columns of the single-hand task's last
    action).
    """
    out = {}
    for name, term in group.items():
        if _is_term(term) and term["name"] == "last_action":
            for key, action in (("last_arm_action", f"arm_ik_{hand}"), ("last_hand_action", f"hand_{hand}")):
                out[key] = copy.deepcopy(term)
                out[key]["params"] = {**(term.get("params") or {}), "action_name": action}
        elif _is_term(term) and "command_name" in (term.get("params") or {}):
            out[name] = copy.deepcopy(term)
            out[name]["params"] = {**term["params"], "command_name": f"lift_target_{hand}"}
        else:
            out[name] = term
    return out


def bimanual_config(right: ManipulationG1BeakerPourConfig) -> ManipulationG1BeakerPourConfig:
    """The pour task ``right`` with both hands on one robot, each beaker poured by the hand on its outer side (see the
    module docstring). Of ``right``'s class; for :func:`run`, not for training.
    """
    left = mirror_config(right)
    r, l = _data(right), _data(left)
    data = copy.deepcopy(r)
    left_arm, left_hand = [mirror_name(dof) for dof in _ARM_DOFS], [mirror_name(dof) for dof in _HAND_DOFS]

    scene = data["scene_options"]
    robot, l_robot = scene["robot"], l["scene_options"]["robot"]
    robot["collision_link_patterns"] = list(
        dict.fromkeys(robot["collision_link_patterns"] + l_robot["collision_link_patterns"])
    )
    robot["default_geoms_friction_torsional"] = {
        **robot["default_geoms_friction_torsional"],
        **l_robot["default_geoms_friction_torsional"],
    }
    for key in ("default_dofs_kp", "default_dofs_kd"):
        robot[key].update({dof: l_robot[key][dof] for dof in left_arm})
    # Both arms hang by default (the right one as the mirror of the left's); ChooseHand raises the chosen one.
    ready = [robot["default_dofs_pos"][dof] for dof in _ARM_DOFS]
    hang = [l_robot["default_dofs_pos"][dof] for dof in _ARM_DOFS]
    robot["default_dofs_pos"].update(dict(zip(_ARM_DOFS, hang, strict=True)))
    # The bench, and both beakers' homes, on the middle; no goal marker (each hand has its own goal).
    for name in ("bench_top", "bench_cabinet", "beaker", "target"):
        scene[name]["default_root_pos"][1] = 0.0
    del scene["goal_marker"]

    obs = data["observation_options"]
    for group in ("policy", "critic"):
        del obs[group]
    for hand, source in zip(HANDS, (r, l), strict=True):
        obs[f"policy_{hand}"] = _hand_group(source["observation_options"]["policy"], hand)

    actions = data["action_options"]
    hold = copy.deepcopy(actions["hold_body"])
    hold["dofs_name"] = [dof for dof in hold["dofs_name"] if dof not in left_arm + left_hand]
    for key in ("arm_ik", "hand", "hold_body"):
        del actions[key]
    for hand, source in zip(HANDS, (r, l), strict=True):
        actions[f"arm_ik_{hand}"] = source["action_options"]["arm_ik"]
        actions[f"hand_{hand}"] = source["action_options"]["hand"]
    actions["hold_body"] = hold

    commands = data["command_options"]
    del commands["lift_target"]
    for hand, source in zip(HANDS, (r, l), strict=True):
        commands[f"lift_target_{hand}"] = {**source["command_options"]["lift_target"], "debug_vis": False}

    events = data["event_options"]
    del events["place_beaker"]
    del events["place_target"]
    events["choose_hand"] = _data(
        ChooseHand.configure(
            priority=1,
            source_name="beaker",
            target_name="target",
            right_arm_dofs=_ARM_DOFS,
            ready_pos=ready,
            hang_pos=hang,
            # The right-hand task's layout, both homes moved to the middle (y 0).
            source_x_range=_SOURCE_PLACE_X,
            right_source_y_range=(_BEAKER_HOME[1] + _SOURCE_PLACE_Y[0], _BEAKER_HOME[1] + _SOURCE_PLACE_Y[1]),
            target_x_range=_TARGET_PLACE_X,
            right_target_y_range=(_TARGET_HOME[1] + _TARGET_PLACE_Y[0], _TARGET_HOME[1] + _TARGET_PLACE_Y[1]),
            min_gap=_MIN_GAP_Y,
        )
    )
    friction = events["hand_friction"]
    friction["links_name"] = list(
        dict.fromkeys(friction["links_name"] + l["event_options"]["hand_friction"]["links_name"])
    )
    stiff = events["stiffen_contact"]["params"]["entity_links"]
    stiff["robot"] = list(
        dict.fromkeys(stiff["robot"] + l["event_options"]["stiffen_contact"]["params"]["entity_links"]["robot"])
    )

    # Measured, not trained: the bead shares pay, and the one hand's grasp metrics go. Both hands' goals latch the
    # same pour and keep the same home (the one source's), so the metrics read the right hand's.
    rewards = data["reward_options"]
    for name in [key for key, term in rewards.items() if _is_term(term) and key != "beads_in_target"]:
        del rewards[name]
    metrics = data["metric_options"]
    for name in ("beaker_grasped", "beaker_three_digit"):
        del metrics[name]
    for term in metrics.values():
        if _is_term(term) and term.get("command_name") == "lift_target":
            term["command_name"] = "lift_target_right"
    data["runner_options"]["experiment_name"] = f"{r['runner_options']['experiment_name']}_bimanual"
    return type(right).load_from_data(data, source=f"{type(right).__name__}, both hands")


def load_actor(checkpoint: str, config: ManipulationG1BeakerPourConfig, obs_dim: int, num_actions: int, device):
    """The actor of an rsl-rl PPO ``checkpoint`` of the pour task, built as ``config``'s runner builds it; it maps a
    hand's actor observations to its 13 actions, deterministically.
    """
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    actor = {k: v for k, v in _data(config)["runner_options"]["actor"].items() if not k.startswith("_")}
    if actor.pop("class_name") != "MLPModel":
        raise ValueError(f"load_actor builds an MLPModel actor; {config.runner_options.actor} is not one.")
    if isinstance(actor.get("distribution_cfg"), dict):
        actor["distribution_cfg"] = {k: v for k, v in actor["distribution_cfg"].items() if not k.startswith("_")}
    obs = TensorDict({"policy": torch.zeros(1, obs_dim, device=device)}, batch_size=[1])
    model = MLPModel(obs, {"actor": ["policy"]}, "actor", num_actions, **actor).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if "actor_state_dict" not in state:
        raise KeyError(
            f"{checkpoint} holds no 'actor_state_dict' (keys: {sorted(state)}); pass an rsl-rl-lib 4.x/5.x PPO "
            f"checkpoint of manipulation_g1_beaker_pour, e.g. logs/manipulation_g1_beaker_pour/<run>/model_<iter>.pt."
        )
    model.load_state_dict(state["actor_state_dict"], strict=True)
    model.eval()

    def act(observations: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return model(TensorDict({"policy": observations}, batch_size=[observations.shape[0]]))

    return act


_REPORTED = ("beads_in_target", "beads_spilled", "beads_in_source", "source_placed", "source_released")


def run(env: RLEnvBase, act, episodes_per_env: int, on_step=None) -> dict[str, dict[str, float]]:
    """Act in ``env`` (a :func:`bimanual_config` env) with the chosen hand of each env, through ``act`` (a hand's
    observations to its actions), until every env has finished ``episodes_per_env`` episodes; ``on_step`` is called
    after each step. Per hand: the episodes, the mean end value of each bead share, and the share ended early.
    """
    chooser = env.event_manager.get_term("choose_hand")
    width = env.action_manager.total_action_dim // len(HANDS)
    obs, _ = env.reset()
    finished = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    sums = {hand: {key: 0.0 for key in (*_REPORTED, "episodes", "ended_early")} for hand in HANDS}
    while (finished < episodes_per_env).any():
        left = chooser.left.clone()
        action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        action[:, :width] = torch.where(~left.unsqueeze(-1), act(obs["policy_right"]), 0.0)
        action[:, width:] = torch.where(left.unsqueeze(-1), act(obs["policy_left"]), 0.0)
        before = {key: env.metric_manager.get_value(key).clone() for key in _REPORTED}
        obs, _, terminated, timeouts, _ = env.step(action)
        done = (terminated | timeouts).bool() & (finished < episodes_per_env)
        for hand, mask in zip(HANDS, (done & ~left, done & left), strict=True):
            sums[hand]["episodes"] += int(mask.sum())
            sums[hand]["ended_early"] += int((mask & terminated.bool() & ~timeouts.bool()).sum())
            for key in _REPORTED:
                sums[hand][key] += float(before[key][mask].sum())
        finished += done.long()
        if on_step is not None:
            on_step()
    return {
        hand: {key: (value if key == "episodes" else value / max(s["episodes"], 1)) for key, value in s.items()}
        for hand, s in sums.items()
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="A manipulation_g1_beaker_pour model (right hand).")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=5, help="At least this many episodes in all.")
    parser.add_argument("--vis", action="store_true", help="Open the Genesis viewer, paced to real time.")
    parser.add_argument("--rec", metavar="MP4", help="Record from a camera over the bench to this mp4 (one env).")
    args = parser.parse_args(argv)
    if args.rec and args.num_envs != 1:
        # The envs share one world, so a camera draws every env's robot and beakers on top of each other.
        parser.error(f"--rec records one env; got --num_envs {args.num_envs}. Record with --num_envs 1.")

    import genesis as gs

    import eden as en

    en.init(backend=gs.gpu)
    from eden.envs.base import env_class_for_config
    from eden.options.camera import CameraOptions, CamerasOptions

    config = bimanual_config(ManipulationG1BeakerPourConfig())
    config.env_options.num_envs = args.num_envs
    if args.rec:
        config.cameras_options = CamerasOptions(
            rec=CameraOptions(
                cam_resolution=(960, 720), cam_pos=(1.05, 0.0, 1.45), cam_lookat=(0.25, 0.0, 0.95), cam_fov=55
            )
        )
    env = env_class_for_config(config).from_config(config, show_viewer=args.vis, eval_mode=True)
    env.build()
    act = load_actor(
        args.checkpoint,
        config,
        env.observation_manager.group_obs_dim["policy_right"][0],
        env.action_manager.total_action_dim // len(HANDS),
        env.device,
    )
    camera = env.get_camera("rec") if args.rec else None
    if camera is not None:
        camera.start_recording(args.rec, fps=1.0 / env.dt)
    # The viewer is watched at real speed: each step waits out what is left of its dt of wall-clock time.
    clock = [time.perf_counter()]

    def on_step() -> None:
        if camera is not None:
            camera.render_rgb()
        if args.vis:
            time.sleep(max(0.0, clock[0] + env.dt - time.perf_counter()))
            clock[0] = time.perf_counter()

    results = run(env, act, -(-args.episodes // args.num_envs), on_step=on_step)
    if camera is not None:
        camera.stop_recording()
        print(f"video: {args.rec}")
    print(f"{'hand':>5} | episodes | beads in target | spilled | left in source | put back | let go | ended early")
    for hand, r in results.items():
        print(
            f"{hand:>5} | {int(r['episodes']):8d} | {r['beads_in_target']:15.2f} | {r['beads_spilled']:7.2f} | "
            f"{r['beads_in_source']:14.2f} | {r['source_placed']:8.2f} | {r['source_released']:6.2f} | "
            f"{r['ended_early']:11.2f}"
        )


if __name__ == "__main__":
    main()
