"""Tests for ``eden.tasks.manipulation.g1_beaker_lift.config``: what the policy observes, the welded pelvis, the arm
holding its commanded pose, the IK keeping the elbow off the torso, the stiff waist, the pre-grasp start, the
grasp-gated lift reward, the beaker's clearance, the gates and cap of its tilt and spin costs and the flung-beaker
termination, a scripted grasp and lift through the task's own actions, the beaker's tilt, the joint-subset penalties,
the per-episode friction and mass draws and the beaker displacement, on the built task.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import pytest
import torch

from eden.tasks import TASK_REGISTRY
from eden.tasks.manipulation.g1_beaker_lift.config import (
    _BEAKER_HOME,
    _EE_LINK,
    _HAND_CLOSED,
    _HAND_LINKS,
    _PALM_IN_WRIST,
    _PELVIS_POS,
    _PLACE_X,
    _PLACE_Y,
    _PREGRASP_ARM,
    _REST_Z,
    _RIGHT_ARM_READY,
    _WAIST_DOFS,
    AccumulatedDifferentialIKController,
    ManipulationG1BeakerLiftConfig,
    object_tilt,
)
from eden.utils.geom import axis_angle_from_quat, inv_quat, quat_mul, transform_by_quat
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


def _quat_about_x(deg: float) -> tuple[float, float, float, float]:
    half = math.radians(deg) / 2
    return (math.cos(half), math.sin(half), 0.0, 0.0)


#: Tipped 20 deg onto its rim on the bench, the beaker's centre is 0.035 sin 20 + 0.0475 cos 20 - 0.0475 = 0.0091 m up:
#: where the third version's policy held it, paid for the lift.
_ON_RIM_20 = (_BEAKER_HOME[0], _BEAKER_HOME[1], _REST_Z + 0.0091)


@pytest.mark.requires_genesis
@pytest.mark.parametrize(
    ("tilt_deg", "pos", "clearance"),
    [
        (0.0, _BEAKER_HOME, 0.0),
        (20.0, _ON_RIM_20, 0.0),
        (0.0, (_BEAKER_HOME[0], _BEAKER_HOME[1], _REST_Z + 0.05), 0.05),
        (20.0, (_ON_RIM_20[0], _ON_RIM_20[1], _ON_RIM_20[2] + 0.05), 0.05),
    ],
)
def test_beaker_clearance_is_its_bottom_above_the_bench_so_tipping_onto_the_rim_is_no_lift(
    env, tilt_deg, pos, clearance
):
    _set_beaker(env, pos, _quat_about_x(tilt_deg))
    env.metric_manager.compute()
    torch.testing.assert_close(
        env.metric_manager.get_value("beaker_clearance"), torch.full((NUM_ENVS,), clearance), atol=1e-4, rtol=0
    )
    env.reset()


@pytest.mark.requires_genesis
def test_tilt_and_spin_cost_once_the_beaker_is_off_the_bench_and_the_spin_cost_is_capped(env):
    upright = env.reward_manager.get_term("beaker_upright")
    ang_vel = env.reward_manager.get_term("beaker_ang_vel")
    ejected = env.termination_manager.get_term("beaker_ejected")
    beaker = env.entities["beaker"]
    tilt_20 = torch.full((NUM_ENVS,), 1.0 - math.cos(math.radians(20.0)))

    def spin(rad_s: float) -> None:
        beaker.set_dofs_vel(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, rad_s]] * NUM_ENVS))
        env.metric_manager.compute()

    # On its rim and not gripped: a nudge from a closing hand, which costs nothing.
    _set_beaker(env, _ON_RIM_20, _quat_about_x(20.0))
    spin(1.0)
    torch.testing.assert_close(upright.compute(), torch.zeros(NUM_ENVS))
    torch.testing.assert_close(ang_vel.compute(), torch.zeros(NUM_ENVS))

    # Off the bench.
    _set_beaker(env, (_ON_RIM_20[0], _ON_RIM_20[1], _ON_RIM_20[2] + 0.05), _quat_about_x(20.0))
    spin(1.0)
    torch.testing.assert_close(upright.compute(), tilt_20, atol=1e-5, rtol=0)
    torch.testing.assert_close(ang_vel.compute(), torch.ones(NUM_ENVS), atol=1e-4, rtol=0)
    assert not ejected.compute().any()

    # Flung by the contact solver: the cost stops at 3 rad/s squared, and the episode ends.
    spin(60.0)
    torch.testing.assert_close(ang_vel.compute(), torch.full((NUM_ENVS,), 9.0), atol=1e-4, rtol=0)
    assert ejected.compute().all()
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
        torch.testing.assert_close(env.metric_manager.get_value("beaker_three_digit"), torch.ones(NUM_ENVS))
        # Gripped but still on the bench: its tilt already costs, so tipping it onto the rim is no way around it.
        assert (env.metric_manager.get_value("beaker_clearance") < 0.01).all()
        tilt = env.metric_manager.get_value("beaker_tilt")
        assert (tilt > 0).all()
        torch.testing.assert_close(env.reward_manager.get_term("beaker_upright").compute(), tilt, atol=0, rtol=0)
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
def test_a_thumb_index_pinch_is_a_grasp_but_not_a_three_digit_one(env):
    """From the beaker's exact home: with the pre-grasp's 5 mm jitter about one pinch in eight misses it."""
    start = env.event_manager.get_term("pregrasp_start")
    jitter, start.pos_jitter = start.pos_jitter, 0.0
    hand = env.action_manager.get_term("hand")
    pinch = {k: v for k, v in _HAND_CLOSED.items() if "middle" not in k}
    hand_action = torch.tensor([[pinch.get(name, 0.0) for name in hand.dofs_name]]) / hand.effective_scale
    action = _zero_action(env)
    grasped, three_digit = torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS)
    try:
        with _pregrasp(env):
            for step in range(30):
                action[:, 6:13] = hand_action * min(1.0, (step + 1) / 15)
                env.step(action)
                if step >= 20:
                    grasped += env.metric_manager.get_value("beaker_grasped")
                    three_digit += env.metric_manager.get_value("beaker_three_digit")
    finally:
        start.pos_jitter = jitter
    # Held on most steps (the thumb's contact hops between its phalanges), never with the middle finger.
    assert (grasped >= 7).all(), grasped
    torch.testing.assert_close(three_digit, torch.zeros(NUM_ENVS))


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
def test_a_full_ik_action_moves_no_arm_joint_target_more_than_the_clamp_a_step(env):
    """The per-step joint clamp is the arm's speed limit (0.015 rad a 50 Hz step)."""
    env.reset()
    arm_ik = env.action_manager.get_term("arm_ik")
    env.step(_zero_action(env))
    before = arm_ik.processed_action.clone()
    action = _zero_action(env)
    action[:, :6] = 1.0
    env.step(action)
    step = (arm_ik.processed_action - before).abs()
    assert step.max() <= 0.015 + 1e-6
    torch.testing.assert_close(step.max(), torch.tensor(0.015), atol=1e-6, rtol=0)  # the clamp binds
    env.reset()


#: Shoulder rolled in and yawed: the elbow 11.7 cm from the torso's midline, 4 cm inside the IK's clearance, the forearm
#: not touching the torso and the palm 18 cm above the bench (at 5 cm, the reset's finger jitter pressed it in).
_TUCKED = dict(
    _RIGHT_ARM_READY,
    right_shoulder_pitch_joint=-0.2,
    right_shoulder_roll_joint=0.15,
    right_shoulder_yaw_joint=0.4,
    right_elbow_joint=-0.6,
)


def _elbow_offset_and_palm(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The elbow's distance from the torso's midline (m), and the palm centre's position and orientation."""
    robot = env.entities["robot"]
    _, links = robot.find_named_links_idx_local(["right_elbow_link", "torso_link", _EE_LINK], preserve_order=True)
    pos = robot.get_links_pos(ls_idx_local=links)
    quat = robot.get_links_quat(ls_idx_local=links)[:, 2]
    palm = pos[:, 2] + transform_by_quat(torch.tensor([_PALM_IN_WRIST] * env.num_envs), quat)
    return pos[:, 1, 1] - pos[:, 0, 1], palm, quat


@contextmanager
def _tucked_elbow(env):
    """The arm set at ``_TUCKED`` and held there by the IK until it has settled under gravity, the elbow push off.

    The tucked hand reaches over the bench's near end of the placement area, so the beaker waits at the far corner and
    is not moved there.
    """
    robot = env.entities["robot"]
    arm = robot.find_named_dofs(list(_RIGHT_ARM_READY))
    arm_ik = env.action_manager.get_term("arm_ik")
    move = env.event_manager.get_term("move_beaker")
    clearance, probability = arm_ik.elbow_clearance, move.probability
    env.reset()
    try:
        arm_ik.elbow_clearance, move.probability = 1e-6, 0.0
        _set_beaker(env, (_BEAKER_HOME[0] + _PLACE_X[1], _BEAKER_HOME[1] + _PLACE_Y[0], _REST_Z))
        robot.set_dofs_pos(torch.tensor([list(_TUCKED.values())] * NUM_ENVS), dofs_idx_local=arm.dofs_idx_local)
        arm_ik.reset()
        for _ in range(40):
            env.step(_zero_action(env))
        arm_ik.elbow_clearance = clearance
        yield arm_ik
    finally:
        arm_ik.elbow_clearance, move.probability = clearance, probability
        env.reset()


@pytest.mark.requires_genesis
def test_the_ik_moves_a_tucked_elbow_out_to_its_clearance_and_leaves_the_palm_where_it_was(env):
    """The fifth version's policy, which has no way to place the elbow, rested the forearm on the torso a quarter of
    the time. The IK's nullspace moves the elbow out to 16 cm, and the palm by a tenth of that: the IK solves at the
    measured joints, which the unassisted arm sags below the target the step is added to.
    """
    touching = env.reward_manager.get_term("arm_on_torso")
    with _tucked_elbow(env):
        before, palm_before, quat_before = _elbow_offset_and_palm(env)
        assert (before < 0.12).all(), before
        for _ in range(80):
            env.step(_zero_action(env))
            torch.testing.assert_close(touching.compute(), torch.zeros(NUM_ENVS))
        after, palm_after, quat_after = _elbow_offset_and_palm(env)
    torch.testing.assert_close(after, torch.full((NUM_ENVS,), 0.16), atol=2e-3, rtol=0)
    assert ((palm_after - palm_before).norm(dim=-1) < 6e-3).all(), palm_after - palm_before
    turned = axis_angle_from_quat(quat_mul(quat_after, inv_quat(quat_before))).norm(dim=-1)
    assert (torch.rad2deg(turned) < 2.0).all(), torch.rad2deg(turned)


@pytest.mark.requires_genesis
def test_the_elbow_push_shortens_a_full_action_instead_of_passing_the_joint_clamp(env):
    with _tucked_elbow(env) as arm_ik:
        assert (_elbow_offset_and_palm(env)[0] < arm_ik.elbow_clearance).all()  # the push is on
        before = arm_ik.processed_action.clone()
        action = _zero_action(env)
        action[:, :6] = 1.0
        env.step(action)
        step = (arm_ik.processed_action - before).abs()
    assert step.max() <= 0.015 + 1e-6


@pytest.mark.requires_genesis
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"elbow_link_name": "right_elbow_link", "body_link_name": ""}, "needs body_link_name"),
        ({"elbow_link_name": "", "body_link_name": "torso_link"}, "do nothing without elbow_link_name"),
        ({"elbow_side": 0.5}, "elbow_side must be -1.0"),
        ({"elbow_max_dq": 0.015}, "must be below max_dq"),
    ],
)
def test_a_half_configured_elbow_clearance_is_refused_at_build(env, change, message):
    options = ManipulationG1BeakerLiftConfig().action_options.arm_ik.model_copy(update=change)
    with pytest.raises(ValueError, match=message):
        AccumulatedDifferentialIKController(env, options).build()


@pytest.mark.requires_genesis
def test_the_torso_pushes_the_forearm_out_and_touching_it_costs(env):
    """Shoulder yawed in 1.3 rad, elbow straight: the forearm lies in the belly. At the ready pose nothing touches."""
    robot = env.entities["robot"]
    arm = robot.find_named_dofs(list(_RIGHT_ARM_READY))
    yaw = arm.columns[list(_RIGHT_ARM_READY).index("right_shoulder_yaw_joint")]
    touching = env.reward_manager.get_term("arm_on_torso")
    env.reset()
    env.step(_zero_action(env))
    torch.testing.assert_close(touching.compute(), torch.zeros(NUM_ENVS))

    across = dict(_RIGHT_ARM_READY, right_shoulder_pitch_joint=0.0, right_shoulder_roll_joint=0.0)
    across.update(right_shoulder_yaw_joint=1.3, right_elbow_joint=0.0)
    robot.set_dofs_pos(torch.tensor([list(across.values())] * NUM_ENVS), dofs_idx_local=arm.dofs_idx_local)
    env.scene.step()  # one physics step, for the contacts at this pose
    torch.testing.assert_close(touching.compute(), torch.ones(NUM_ENVS))
    # Held there by the IK (re-based on the pose just set), the forearm is still pushed out within a control step.
    env.action_manager.get_term("arm_ik").reset()
    env.step(_zero_action(env))
    assert (robot.get_dofs_pos()[:, yaw] < 1.0).all()
    env.reset()


@pytest.mark.requires_genesis
@pytest.mark.parametrize(("elbow_vel", "expected"), [(2.0, 1.5**2), (1000.0, (15.0 - 0.5) ** 2)])
def test_arm_speed_penalty_is_a_hinge_past_half_a_rad_per_s_capped_at_15(env, elbow_vel, expected):
    """1000 rad/s is a contact-solver blow-up: the cap keeps its square out of the returns."""
    env.reset()
    robot = env.entities["robot"]
    elbow = robot.find_named_dofs(["right_elbow_joint"])
    robot.set_dofs_vel(torch.full((NUM_ENVS, 1), elbow_vel), dofs_idx_local=elbow.dofs_idx_local)
    penalty = env.reward_manager.get_term("arm_joint_vel")
    torch.testing.assert_close(penalty.compute(), torch.full((NUM_ENVS,), expected), atol=1e-3, rtol=0)
    env.reset()


@pytest.mark.requires_genesis
def test_the_hand_friction_is_one_draw_for_the_whole_hand_and_it_and_the_beaker_mass_are_redrawn_every_episode(env):
    """Drawn per link, the hand was almost never slippery all over; drawn once, an env kept its hand and beaker."""
    friction = env.event_manager.get_term("hand_friction")
    mass = env.event_manager.get_term("beaker_mass")
    robot, beaker = env.entities["robot"], env.entities["beaker"]
    _, hand = robot.find_named_links_idx_local(_HAND_LINKS, preserve_order=True)
    fixed = friction.friction_range, mass.mass_shift_range
    friction.friction_range, mass.mass_shift_range = (0.8, 2.0), (-0.06, 0.12)
    draws = []
    try:
        for _ in range(2):
            env.reset()
            draws.append((robot.get_friction_ratio(ls_idx_local=hand), beaker.get_mass_shift()))
    finally:
        friction.friction_range, mass.mass_shift_range = fixed
        env.reset()
    for ratio, shift in draws:
        assert ratio.shape == (NUM_ENVS, len(_HAND_LINKS))
        torch.testing.assert_close(ratio, ratio[:, :1].expand_as(ratio))
        assert ((ratio >= 0.8) & (ratio <= 2.0)).all()
        assert ((shift >= -0.06 - 1e-6) & (shift <= 0.12 + 1e-6)).all()
    assert not torch.allclose(draws[0][0], draws[1][0])
    assert not torch.allclose(draws[0][1], draws[1][1])


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
