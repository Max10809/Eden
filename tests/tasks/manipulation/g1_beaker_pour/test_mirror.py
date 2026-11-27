"""Tests for ``eden.tasks.manipulation.g1_beaker_pour.mirror``: the joint signs reflect the whole body, and in mirrored
states the left-hand task observes, rewards and measures what the right-hand task does and moves the same action's
mirror, so the right hand's policy drives the left hand as it is; an observation term with no mirror signs is
refused.
"""

from __future__ import annotations

import pytest
import torch

from eden.tasks import TASK_REGISTRY
from eden.tasks.manipulation.custom_terms.observations import object_position_b
from eden.tasks.manipulation.g1_beaker_pour.mirror import joint_sign, mirror_config, mirror_name
from eden.terms import ObsTerm
from tests.support.envs import build_env

NUM_ENVS = 2
_Y = torch.tensor([1.0, -1.0, 1.0])


def _config(task: str):
    config = TASK_REGISTRY.build(task).with_overrides_from_dict({"env_options": {"num_envs": NUM_ENVS}})
    config.observation_options.policy.enable_corruption = False
    config.event_options.hand_friction.friction_range = (1.5, 1.5)
    config.event_options.beaker_mass.mass_shift_range = (0.0, 0.0)
    return config


@pytest.fixture(scope="module")
def envs():
    """The right-hand task and its mirror, both reset; the arm of each stays at its ready pose under zero actions."""
    return (
        build_env(_config("manipulation_g1_beaker_pour"), reset=True),
        build_env(_config("manipulation_g1_beaker_pour_left"), reset=True),
    )


def _mirror_pose(qpos: torch.Tensor) -> torch.Tensor:
    """Free bodies' ``(..., 7)`` root qpos (position, wxyz quaternion) reflected through the xz-plane: the position's y
    flips, and so do the x and z of the quaternion's vector part, an axial vector.
    """
    y = _Y.to(qpos.device)
    return torch.cat([qpos[..., :3] * y, qpos[..., 3:4], qpos[..., 4:] * -y], dim=-1)


def _mirror_robot(right, left, q: torch.Tensor) -> None:
    """Set the right robot's joints to ``q`` (its dofs_name order) and the left robot's to their mirror."""
    names = list(right.entities["robot"].dofs_name)
    dofs_r = right.entities["robot"].find_named_dofs(names)
    dofs_l = left.entities["robot"].find_named_dofs([mirror_name(name) for name in names])
    signs = torch.tensor([joint_sign(name) for name in names], device=q.device)
    right.entities["robot"].set_dofs_pos(q, dofs_idx_local=dofs_r.dofs_idx_local, zero_velocity=True)
    left.entities["robot"].set_dofs_pos(q * signs, dofs_idx_local=dofs_l.dofs_idx_local, zero_velocity=True)


def _mirror_state(right, left, poured: bool = False) -> None:
    """Put the left task in the mirror of the right task's state: joints, both beakers, the beads, the source's home,
    whether it is poured (``poured``, in both, with the goal over the home) and the goal, and restart both observation
    histories there. Each reset draws its own joint offsets, so two resets are not mirrors.
    """
    names = list(right.entities["robot"].dofs_name)
    q = right.entities["robot"].get_dofs_pos()[..., right.entities["robot"].find_named_dofs(names).columns]
    _mirror_robot(right, left, q)
    for name in ("beaker", "target"):
        entity = right.entities[name]
        left.entities[name].set_qpos(_mirror_pose(entity.get_qpos()), qs_idx_local=range(7), zero_velocity=True)
    beads = right.entities["beads"]
    qpos = beads.get_qpos(qs_idx_local=range(beads.n_qs)).reshape(NUM_ENVS, -1, 7)
    left.entities["beads"].set_qpos(
        _mirror_pose(qpos).reshape(NUM_ENVS, -1), qs_idx_local=range(beads.n_qs), zero_velocity=True
    )
    command_r, command_l = (env.command_manager.get_term("lift_target") for env in (right, left))
    command_l.home_b.copy_(command_r.home_b * _Y.to(command_r.home_b.device))
    for command in (command_r, command_l):
        command.poured.fill_(poured)
        command.over_home.fill_(False)
    if poured:
        command_r.command.copy_(command_r.over_home_b)
    command_l.command.copy_(command_r.command * _Y.to(command_r.command.device))
    for env in (right, left):
        env.observation_manager.reset()


def _links_by_name(env) -> dict[str, torch.Tensor]:
    robot = env.entities["robot"]
    pos = robot.get_links_pos()
    return {link.name: pos[:, i] for i, link in enumerate(robot.links)}


@pytest.mark.requires_genesis
def test_the_joint_signs_reflect_the_whole_body(envs):
    """Every link of the left task's robot, posed by the mirrored joints, is where the right's mirror image is."""
    right, left = envs
    right.reset()
    left.reset()
    robot = right.entities["robot"]
    names = list(robot.dofs_name)
    lower, upper = (bound[..., robot.find_named_dofs(names).columns] for bound in robot.get_dofs_limit())
    generator = torch.Generator(device=lower.device).manual_seed(0)
    q = (lower + (upper - lower) * torch.rand(lower.shape, generator=generator, device=lower.device)).expand(
        NUM_ENVS, -1
    )
    _mirror_robot(right, left, q.contiguous())
    links_r, links_l = _links_by_name(right), _links_by_name(left)
    y = _Y.to(lower.device)
    for name, pos in links_r.items():
        torch.testing.assert_close(links_l[mirror_name(name)], pos * y, atol=1e-4, rtol=0, msg=name)
    right.reset()
    left.reset()


@pytest.mark.requires_genesis
@pytest.mark.parametrize("poured", [False, True])
def test_in_mirrored_states_the_left_task_observes_rewards_and_measures_what_the_right_does(envs, poured):
    """Before the pour and after it, when the goal is over the source's home and the terms that put it back pay."""
    right, left = envs
    right.reset()
    left.reset()
    _mirror_state(right, left, poured)
    for env in envs:
        env.metric_manager.compute()
    obs_r, obs_l = right.observation_manager.compute(), left.observation_manager.compute()
    # To 1 mm and 1e-3: the two hands' meshes differ by a fraction of a millimetre.
    for group in ("policy", "critic"):
        torch.testing.assert_close(obs_l[group], obs_r[group], atol=1e-3, rtol=0, msg=group)
    for name in right.metric_manager.active_terms:
        torch.testing.assert_close(
            left.metric_manager.get_value(name), right.metric_manager.get_value(name), atol=1e-3, rtol=0, msg=name
        )
    for name in right.reward_manager.active_terms:
        torch.testing.assert_close(
            left.reward_manager.get_term(name).compute(),
            right.reward_manager.get_term(name).compute(),
            atol=1e-3,
            rtol=0,
            msg=name,
        )
    right.reset()
    left.reset()


@pytest.mark.requires_genesis
def test_the_same_action_moves_the_left_arm_and_hand_as_the_mirror_of_the_right(envs):
    """A reach forward, out and up, turning the palm, and the fingers closing: the left robot ends up the mirror. Clear
    of the body: the two sides' meshes differ by a fraction of a millimetre, so a contact can come a step apart.
    """
    right, left = envs
    right.reset()
    left.reset()
    _mirror_state(right, left)
    action = torch.zeros(NUM_ENVS, right.action_manager.total_action_dim, device=right.device)
    action[:, :6] = torch.tensor([0.6, -0.4, 0.3, 0.5, -0.3, 0.4], device=right.device)
    action[:, 6:] = 0.5
    for _ in range(30):
        right.step(action)
        left.step(action)
    links_r, links_l = _links_by_name(right), _links_by_name(left)
    y = _Y.to(right.device)
    for name in ("right_wrist_yaw_link", "right_hand_thumb_2_link", "right_hand_index_1_link", "right_elbow_link"):
        torch.testing.assert_close(links_l[mirror_name(name)], links_r[name] * y, atol=1e-3, rtol=0, msg=name)
    right.reset()
    left.reset()


def test_an_observation_term_with_no_mirror_signs_is_refused():
    config = TASK_REGISTRY.build("manipulation_g1_beaker_pour")
    config.observation_options.policy.beaker_height = ObsTerm.configure(
        func=object_position_b, params={"robot_name": "robot", "object_name": "beaker"}
    )
    with pytest.raises(KeyError, match=r"observation_options\.policy\.beaker_height: no mirror signs"):
        mirror_config(config)
