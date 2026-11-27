"""Tests for ``eden.tasks.manipulation.g1_beaker_pour.bimanual``: the target stands toward the middle from the
source and the hand on the source's outer side takes it, its arm ready while the other hangs; a left-hand episode observes and
moves as the mirror of a right-hand one; the runner acts with the chosen hand only and reports both; a checkpoint
without an actor is refused.
"""

from __future__ import annotations

import pytest
import torch

from eden.tasks import TASK_REGISTRY
from eden.tasks.manipulation.g1_beaker_lift.config import _ARM_DOFS
from eden.tasks.manipulation.g1_beaker_pour.bimanual import HANDS, bimanual_config, load_actor, run
from eden.tasks.manipulation.g1_beaker_pour.config import _MIN_GAP_Y
from eden.tasks.manipulation.g1_beaker_pour.mirror import joint_sign, mirror_name
from tests.support.envs import build_env

NUM_ENVS = 8
_Y = torch.tensor([1.0, -1.0, 1.0])


def _config():
    right = TASK_REGISTRY.build("manipulation_g1_beaker_pour").with_overrides_from_dict(
        {"env_options": {"num_envs": NUM_ENVS}}
    )
    right.event_options.hand_friction.friction_range = (1.5, 1.5)
    right.event_options.beaker_mass.mass_shift_range = (0.0, 0.0)
    config = bimanual_config(right)
    for hand in HANDS:
        getattr(config.observation_options, f"policy_{hand}").enable_corruption = False
    return config


@pytest.fixture(scope="module")
def env():
    return build_env(_config(), reset=True)


def _arms(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Each env's right-arm and left-arm joints, in _ARM_DOFS order (the left arm's in its mirrored names)."""
    robot = env.entities["robot"]
    q = robot.get_dofs_pos()
    return (
        q[..., robot.find_named_dofs(_ARM_DOFS).columns],
        q[..., robot.find_named_dofs([mirror_name(dof) for dof in _ARM_DOFS]).columns],
    )


@pytest.mark.requires_genesis
def test_the_target_stands_toward_the_middle_and_the_hand_on_the_sources_outer_side_takes_it(env):
    chooser = env.event_manager.get_term("choose_hand")
    ready, hang = torch.tensor(chooser.ready_pos), torch.tensor(chooser.hang_pos)
    signs = torch.tensor([joint_sign(dof) for dof in _ARM_DOFS])
    seen = {hand: 0 for hand in HANDS}
    for _ in range(25):
        chooser.compute(envs_idx=None)
        source, target = env.entities["beaker"].get_pos(), env.entities["target"].get_pos()
        # The source right of the target goes to the right hand, left of it to the left hand.
        assert torch.equal(chooser.left, source[:, 1] > target[:, 1])
        right_arm, left_arm = (arm.cpu() for arm in _arms(env))
        for i, left in enumerate(chooser.left.tolist()):
            hand = HANDS[int(left)]
            seen[hand] += 1
            # Mirrored into the right hand's frame, each layout is the right-hand task's.
            sign = -1.0 if left else 1.0
            source_y, target_y = sign * source[i, 1].item(), sign * target[i, 1].item()
            assert -0.33 - 1e-5 <= source_y <= -0.02 + 1e-5, (hand, source[i])
            assert target_y - source_y >= _MIN_GAP_Y - 1e-5 and target_y <= 0.10 + 1e-5, (hand, source[i], target[i])
            torch.testing.assert_close(right_arm[i], hang if left else ready, atol=1e-4, rtol=0)
            torch.testing.assert_close(left_arm[i], (ready if left else hang) * signs, atol=1e-4, rtol=0)
    assert min(seen.values()) > 50, seen
    env.reset()


def _mirror_env(env, src: int, dst: int) -> None:
    """Put env ``dst`` (left hand) in the mirror of env ``src``'s state (right hand), goals included, and restart the
    observation histories.
    """
    robot = env.entities["robot"]
    names = list(robot.dofs_name)
    dofs = robot.find_named_dofs(names)
    signs = torch.tensor([joint_sign(name) for name in names], device=env.device)
    q = robot.get_dofs_pos()[..., dofs.columns].clone()
    # Each joint's value goes to its mirror's column in env dst, signed.
    q[dst, robot.find_named_dofs([mirror_name(name) for name in names]).columns] = q[src] * signs
    robot.set_dofs_pos(q, dofs_idx_local=dofs.dofs_idx_local, zero_velocity=True)
    y, axial = _Y.to(env.device), -_Y.to(env.device)
    for name in ("beaker", "target", "beads"):
        entity = env.entities[name]
        qpos = entity.get_qpos(qs_idx_local=range(entity.n_qs)).reshape(env.num_envs, -1, 7).clone()
        qpos[dst, :, :3] = qpos[src, :, :3] * y
        qpos[dst, :, 3] = qpos[src, :, 3]
        qpos[dst, :, 4:] = qpos[src, :, 4:] * axial
        entity.set_qpos(qpos.reshape(env.num_envs, -1), qs_idx_local=range(entity.n_qs), zero_velocity=True)
    right, left = (env.command_manager.get_term(f"lift_target_{hand}").command for hand in HANDS)
    left[dst] = right[src] * y
    chooser = env.event_manager.get_term("choose_hand")
    chooser.left[src], chooser.left[dst] = False, True
    env.observation_manager.reset()


@pytest.mark.requires_genesis
def test_a_left_hand_episode_observes_and_moves_as_the_mirror_of_a_right_hand_one(env):
    env.reset()
    _mirror_env(env, src=0, dst=1)
    obs = env.observation_manager.compute()
    torch.testing.assert_close(obs["policy_left"][1], obs["policy_right"][0], atol=1e-3, rtol=0)

    width = env.action_manager.total_action_dim // 2
    move = torch.tensor([0.6, -0.4, 0.3, 0.5, -0.3, 0.4] + [0.5] * (width - 6), device=env.device)
    action = torch.zeros(env.num_envs, 2 * width, device=env.device)
    action[0, :width], action[1, width:] = move, move
    for _ in range(25):
        env.step(action)
    links = {link.name: i for i, link in enumerate(env.entities["robot"].links)}
    pos = env.entities["robot"].get_links_pos()
    for name in ("right_wrist_yaw_link", "right_hand_thumb_2_link", "right_hand_index_1_link", "right_elbow_link"):
        torch.testing.assert_close(
            pos[1, links[mirror_name(name)]], pos[0, links[name]] * _Y.to(env.device), atol=1e-3, rtol=0, msg=name
        )
    env.reset()


def _actor_checkpoint(config, path, obs_dim: int) -> str:
    """An untrained actor of the pour task's architecture, saved as an rsl-rl checkpoint."""
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    spec = config.runner_options.actor
    model = MLPModel(
        TensorDict({"policy": torch.zeros(1, obs_dim)}, batch_size=[1]),
        {"actor": ["policy"]},
        "actor",
        13,
        hidden_dims=list(spec.hidden_dims),
        activation=spec.activation,
        obs_normalization=spec.obs_normalization,
        distribution_cfg=spec.distribution_cfg.model_dump(),
    )
    torch.save({"actor_state_dict": model.state_dict()}, path)
    return str(path)


@pytest.mark.requires_genesis
def test_the_runner_acts_with_the_chosen_hand_only_and_reports_both(env, tmp_path):
    obs_dim = env.observation_manager.group_obs_dim["policy_right"][0]
    assert env.observation_manager.group_obs_dim["policy_left"][0] == obs_dim
    act = load_actor(_actor_checkpoint(_config(), tmp_path / "actor.pt", obs_dim), _config(), obs_dim, 13, env.device)
    chooser = env.event_manager.get_term("choose_hand")
    hang = torch.tensor(chooser.hang_pos, device=env.device)
    signs = torch.tensor([joint_sign(dof) for dof in _ARM_DOFS], device=env.device)
    idle_drift = []

    def idle_arm_drift() -> None:
        right_arm, left_arm = _arms(env)
        idle = torch.where(chooser.left.unsqueeze(-1), right_arm - hang, left_arm - hang * signs)
        idle_drift.append(idle.abs().max().item())

    results = run(env, act, episodes_per_env=1, on_step=idle_arm_drift)
    assert sum(r["episodes"] for r in results.values()) == NUM_ENVS
    for r in results.values():
        for key in ("beads_in_target", "source_placed", "source_released", "ended_early"):
            assert 0.0 <= r[key] <= 1.0, (key, r[key])
    # The idle arm holds where it hangs, under the reset's joint jitter and its own sag.
    assert max(idle_drift) < 0.1, max(idle_drift)
    env.reset()


def test_a_checkpoint_without_an_actor_is_refused(tmp_path):
    path = tmp_path / "critic_only.pt"
    torch.save({"critic_state_dict": {}}, path)
    with pytest.raises(KeyError, match="holds no 'actor_state_dict'"):
        load_actor(str(path), _config(), 84, 13, "cpu")
