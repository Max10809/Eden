"""Tests for ``eden.tasks.manipulation.g1_beaker_lift.config``: what the policy observes, the welded pelvis, the arm
holding its commanded pose, the stiff waist, the pre-grasp start, the grasp-gated lift reward, a scripted grasp and
lift through the task's own actions, the beaker's tilt, the joint-subset penalties and the beaker displacement, on
the built task.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import pytest
import torch

from eden.tasks import TASK_REGISTRY
from eden.tasks.manipulation.g1_beaker_lift.config import (
    _BEAKER_HOME,
    _HAND_CLOSED,
    _PELVIS_POS,
    _PLACE_X,
    _PLACE_Y,
    _PREGRASP_ARM,
    _REST_Z,
    _WAIST_DOFS,
    ManipulationG1BeakerLiftConfig,
    object_tilt,
)
from eden.utils.geom import axis_angle_from_quat, inv_quat, quat_mul
from tests.support.envs import build_env

NUM_ENVS = 2
# What the real robot measures: arm and Dex3 encoders, FK of them, the perceived beaker pose, the goal, the last action.
MEASURED = {
    "arm_dofs_pos",
    "arm_dofs_vel",
    "hand_dofs_pos",
    "hand_dofs_vel",
    "object_position",
    "object_orientation",
    "target_object_position",
    "grasp_point",
    "fingertips_pos",
    "last_action",
}
# What only the simulator knows.
PRIVILEGED = {"fingertips_to_object", "object_lin_vel", "object_ang_vel"}


def _term_names(group) -> set[str]:
    return {k for k, v in (group.__pydantic_extra__ or {}).items() if not isinstance(v, str)}


def test_the_policy_observes_only_what_the_robot_measures_and_the_critic_also_sees_the_privileged_state():
    obs = ManipulationG1BeakerLiftConfig().observation_options
    assert _term_names(obs.policy) == MEASURED
    assert _term_names(obs.critic) == MEASURED | PRIVILEGED


@pytest.fixture(scope="module")
def env():
    """The task, deterministic: no pre-grasp starts unless a test asks (``_pregrasp``), nominal hand and beaker."""
    config = TASK_REGISTRY.build("manipulation_g1_beaker_lift").with_overrides_from_dict(
        {"env_options": {"num_envs": NUM_ENVS}}
    )
    config.event_options.pregrasp_start.probability = 0.0
    config.event_options.hand_friction.friction_range = (1.5, 1.5)
    config.event_options.beaker_mass.mass_shift_range = (0.0, 0.0)
    return build_env(config, reset=True)


def _zero_action(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)


@contextmanager
def _pregrasp(env):
    """Every reset inside starts at the pre-grasp pose; the env is reset back to the normal start afterwards."""
    start = env.event_manager.get_term("pregrasp_start")
    probability = start.probability
    start.probability = 1.0
    try:
        env.reset()
        yield
    finally:
        start.probability = probability
        env.reset()


def _set_beaker(env, pos: tuple[float, float, float], quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)):
    beaker = env.entities["beaker"]
    beaker.set_quat(torch.tensor([quat] * env.num_envs, device=env.device))
    beaker.set_pos(torch.tensor([pos] * env.num_envs, device=env.device))


@pytest.mark.requires_genesis
def test_the_welded_pelvis_stays_where_the_mount_holds_it(env):
    env.reset()
    for _ in range(5):
        env.step(_zero_action(env))
    pelvis = env.entities["robot"].get_pos()
    torch.testing.assert_close(pelvis, torch.tensor([_PELVIS_POS] * NUM_ENVS), atol=1e-4, rtol=0)


@pytest.mark.requires_genesis
def test_a_zero_ik_action_holds_the_arm_target_instead_of_following_its_sag(env):
    """The arm has no gravity compensation, so a target re-based on the measured joints would sink with them."""
    env.reset()
    arm_ik = env.action_manager.get_term("arm_ik")
    env.step(_zero_action(env))
    first_target = arm_ik.processed_action.clone()
    for _ in range(10):
        env.step(_zero_action(env))
    torch.testing.assert_close(arm_ik.processed_action, first_target, atol=1e-6, rtol=0)


@pytest.mark.requires_genesis
def test_pregrasp_start_puts_the_open_hand_around_the_beaker_at_home(env):
    robot = env.entities["robot"]
    arm = robot.find_named_dofs(list(_PREGRASP_ARM))
    with _pregrasp(env):
        torch.testing.assert_close(
            robot.get_dofs_pos(dofs_idx_local=arm.dofs_idx_local),
            torch.tensor([list(_PREGRASP_ARM.values())] * NUM_ENVS),
            atol=1e-4,
            rtol=0,
        )
        offset = env.entities["beaker"].get_pos() - torch.tensor(_BEAKER_HOME)
        assert (offset[:, :2].abs() <= 0.005 + 1e-5).all()
        env.step(_zero_action(env))
        # The open hand does not hold it: an opposed grasp needs the hand to close.
        torch.testing.assert_close(env.metric_manager.get_value("beaker_grasped"), torch.zeros(NUM_ENVS))


@pytest.mark.requires_genesis
def test_the_stiff_waist_holds_the_torso_up_with_the_arm_reaching_out(env):
    """At the asset's waist gains the torso sagged 0.48 rad in roll and in pitch under this arm pose."""
    robot = env.entities["robot"]
    waist = robot.find_named_dofs(_WAIST_DOFS)
    with _pregrasp(env):
        for _ in range(20):
            env.step(_zero_action(env))
        assert robot.get_dofs_pos(dofs_idx_local=waist.dofs_idx_local).abs().max() < 0.05


@pytest.mark.requires_genesis
def test_lift_height_pays_nothing_for_a_beaker_off_the_bench_that_the_hand_does_not_hold(env):
    env.reset()
    # Away from home, where 5 cm up reaches the raised hand's fingers.
    _set_beaker(env, (0.36, -0.28, _REST_Z + 0.05))
    env.step(_zero_action(env))
    assert (env.entities["beaker"].get_pos()[:, 2] > _REST_Z + 0.03).all()
    torch.testing.assert_close(env.reward_manager.get_term("lift_height").compute(), torch.zeros(NUM_ENVS))
    env.reset()


def _beaker_in_hand(env) -> torch.Tensor:
    """The beaker's orientation in the palm (wrist-yaw link) frame, wxyz."""
    robot = env.entities["robot"]
    _, wrist = robot.find_named_links_idx_local(["right_wrist_yaw_link"], preserve_order=True)
    return quat_mul(inv_quat(robot.get_links_quat(ls_idx_local=wrist)[:, 0]), env.entities["beaker"].get_quat())


@pytest.mark.requires_genesis
def test_closing_and_raising_the_hand_from_the_pregrasp_lifts_the_beaker_and_holds_it_without_creep(env):
    """The task's own actions can do it: a hand closure over 15 steps, then full +z on the palm's IK.

    Held still afterwards, the beaker must not turn in the hand: at the default ``impratio`` the soft friction let
    it creep round at about 4 deg/s.
    """
    hand = env.action_manager.get_term("hand")
    closed = torch.tensor([[_HAND_CLOSED.get(name, 0.0) for name in hand.dofs_name]])
    hand_action = closed / hand.effective_scale
    action = _zero_action(env)
    columns = {name: i for i, name in enumerate(env.action_manager.active_terms)}
    assert list(columns) == ["arm_ik", "hand", "hold_body"]  # arm_ik's 6 columns, then the hand's 7
    with _pregrasp(env):
        for step in range(30):
            action[:, 6:13] = hand_action * min(1.0, (step + 1) / 15)
            env.step(action)
        torch.testing.assert_close(env.metric_manager.get_value("beaker_grasped"), torch.ones(NUM_ENVS))
        action[:, 2] = 1.0  # full +z, in the IK's (x, y, z, rx, ry, rz)
        grasped_steps = torch.zeros(NUM_ENVS)
        for _ in range(60):
            env.step(action)
            grasped_steps += env.metric_manager.get_value("beaker_grasped")
        beaker = env.entities["beaker"]
        assert (beaker.get_pos()[:, 2] - _REST_Z > 0.08).all()
        assert (object_tilt(env, object_name="beaker") < 1.0 - math.cos(math.radians(15.0))).all()
        # Held through the raise. Not on every single step: the thumb's contact hops between its two phalanges.
        assert (grasped_steps / 60 > 0.9).all()

        action[:, 2] = 0.0
        before = _beaker_in_hand(env)
        for _ in range(100):  # 2 s
            env.step(action)
        turned = axis_angle_from_quat(quat_mul(_beaker_in_hand(env), inv_quat(before))).norm(dim=-1)
        assert (torch.rad2deg(turned) < 1.0).all()


@pytest.mark.requires_genesis
@pytest.mark.parametrize("tilt_deg", [0.0, 45.0, 90.0])
def test_object_tilt_is_one_minus_the_cosine_of_the_tilt(env, tilt_deg):
    half = math.radians(tilt_deg) / 2
    _set_beaker(env, _BEAKER_HOME, (math.cos(half), math.sin(half), 0.0, 0.0))  # about world x
    expected = torch.full((NUM_ENVS,), 1.0 - math.cos(math.radians(tilt_deg)))
    torch.testing.assert_close(object_tilt(env, object_name="beaker"), expected, atol=1e-5, rtol=0)
    env.reset()


@pytest.mark.requires_genesis
def test_arm_joint_limit_penalty_counts_the_arm_alone(env):
    """The Dex3 fingers rest open on their hard limit, outside the 90 % soft band, and must not count."""
    env.reset()
    penalty = env.reward_manager.get_term("arm_joint_pos_limits")
    torch.testing.assert_close(penalty.compute(), torch.zeros(NUM_ENVS), atol=1e-6, rtol=0)

    robot = env.entities["robot"]
    elbow = robot.find_named_dofs(["right_elbow_joint"])
    _, upper = robot.get_dofs_limit(dofs_idx_local=elbow.dofs_idx_local)
    robot.set_dofs_pos(upper, dofs_idx_local=elbow.dofs_idx_local)
    # Range [-1.0472, 2.0944]: the soft band ends 5 % of the range (0.15708 rad) inside the hard limit.
    torch.testing.assert_close(penalty.compute(), torch.full((NUM_ENVS,), 0.15708), atol=1e-4, rtol=0)
    env.reset()


@pytest.mark.requires_genesis
def test_move_beaker_redraws_a_resting_beaker_but_leaves_a_lifted_one(env):
    env.reset()
    move = env.event_manager.get_term("move_beaker")
    probability = move.probability
    move.probability = 1.0
    try:
        beaker = env.entities["beaker"]
        all_envs = torch.ones(NUM_ENVS, dtype=torch.bool)

        # Resting at home with the hand raised behind the bench edge: moved, within the placement area.
        _set_beaker(env, _BEAKER_HOME)
        move.compute(all_envs)
        pos = beaker.get_pos()
        offset = pos[:, :2] - torch.tensor(_BEAKER_HOME[:2])
        assert (offset.norm(dim=-1) > 1e-4).all()
        assert ((offset[:, 0] >= _PLACE_X[0] - 1e-5) & (offset[:, 0] <= _PLACE_X[1] + 1e-5)).all()
        assert ((offset[:, 1] >= _PLACE_Y[0] - 1e-5) & (offset[:, 1] <= _PLACE_Y[1] + 1e-5)).all()
        torch.testing.assert_close(pos[:, 2], torch.full((NUM_ENVS,), _REST_Z), atol=1e-5, rtol=0)

        # Off the bench (held): left where it is.
        lifted = (_BEAKER_HOME[0], _BEAKER_HOME[1], _REST_Z + 0.05)
        _set_beaker(env, lifted)
        move.compute(all_envs)
        torch.testing.assert_close(beaker.get_pos(), torch.tensor([lifted] * NUM_ENVS), atol=1e-5, rtol=0)
    finally:
        move.probability = probability
        env.reset()
