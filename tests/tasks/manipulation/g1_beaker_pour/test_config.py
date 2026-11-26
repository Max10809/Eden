"""Tests for ``eden.tasks.manipulation.g1_beaker_pour.config``: the actor's observations are the lift's and the
critic's add the bead counts last, the beads start at rest in the source, the goal is drawn beside the target, beads
poured over the target land in it and beads poured beside it spill and end the episode, the pour lip, the goal
terms measured to the beaker turned back upright about the palm and paid in full while it pours, the tilt toward the
target, the pour-tilt gate and the cost of leaning across or away from the target, on the built task.
"""

from __future__ import annotations

import math

import pytest
import torch

from eden.tasks import TASK_REGISTRY
from eden.tasks.manipulation.custom_terms.common import _grasp_point_w
from eden.tasks.manipulation.g1_beaker_lift.config import ManipulationG1BeakerLiftConfig
from eden.tasks.manipulation.g1_beaker_lift.config import _BENCH_TOP, _EE_LINK, _PALM_IN_WRIST
from eden.tasks.manipulation.g1_beaker_pour.config import (
    _BEAD_COUNT,
    _BEAD_RADIUS,
    _BEAKER_HEIGHT,
    _BEAKER_RADIUS,
    _GOAL_FROM_TARGET_X,
    _GOAL_FROM_TARGET_Y,
    _GOAL_FROM_TARGET_Z,
    _INNER_RADIUS,
    CRITIC_EXTRA_DIMS,
    ManipulationG1BeakerPourConfig,
    pour_lip,
)
from eden.utils.geom import transform_by_quat
from tests.support.envs import build_env

NUM_ENVS = 2
#: The critic's inputs beyond the lift's, last and in this order (a lift checkpoint is widened for them at the end).
BEAD_TERMS = ["beads_in_source", "beads_in_target", "beads_spilled", "beads_from_mouth"]
#: The lift's actor and critic input sizes, which a lift checkpoint is built for.
LIFT_ACTOR_DIMS, LIFT_CRITIC_DIMS = 83, 98


def _term_names(group) -> list[str]:
    return [k for k, v in (group.__pydantic_extra__ or {}).items() if not isinstance(v, str)]


def test_the_actor_observes_what_the_lift_actor_does_and_the_critic_adds_the_bead_counts_last():
    lift = ManipulationG1BeakerLiftConfig().observation_options
    pour = ManipulationG1BeakerPourConfig().observation_options
    assert _term_names(pour.policy) == _term_names(lift.policy)
    assert _term_names(pour.critic) == _term_names(lift.critic) + BEAD_TERMS


@pytest.fixture(scope="module")
def env():
    """The task with the nominal hand and source; the arm stays at its raised ready pose under zero actions."""
    config = TASK_REGISTRY.build("manipulation_g1_beaker_pour").with_overrides_from_dict(
        {"env_options": {"num_envs": NUM_ENVS}}
    )
    config.event_options.hand_friction.friction_range = (1.5, 1.5)
    config.event_options.beaker_mass.mass_shift_range = (0.0, 0.0)
    return build_env(config, reset=True)


def _zero_action(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)


def _rows(env, values) -> torch.Tensor:
    return torch.tensor([values] * env.num_envs, device=env.device, dtype=torch.float32)


def _quat_about_x(env, deg: float) -> torch.Tensor:
    half = math.radians(deg) / 2
    return _rows(env, (math.cos(half), math.sin(half), 0.0, 0.0))


@pytest.mark.requires_genesis
def test_the_inputs_are_the_lifts_sizes_and_the_bead_counts_add_the_widened_columns(env):
    obs = env.observation_manager
    names, dims = obs.group_term_names["critic"], obs.group_obs_term_dim["critic"]
    assert names[-len(BEAD_TERMS) :] == BEAD_TERMS
    assert sum(math.prod(dim) for dim in dims[-len(BEAD_TERMS) :]) == CRITIC_EXTRA_DIMS
    assert obs.group_obs_dim["policy"] == (LIFT_ACTOR_DIMS,)
    assert obs.group_obs_dim["critic"] == (LIFT_CRITIC_DIMS + CRITIC_EXTRA_DIMS,)


@pytest.mark.requires_genesis
def test_every_bead_starts_at_rest_in_the_source_and_the_goal_is_beside_the_target(env):
    env.reset()
    for _ in range(25):
        env.step(_zero_action(env))
    metrics = env.metric_manager
    torch.testing.assert_close(metrics.get_value("beads_in_source"), torch.ones(NUM_ENVS, device=env.device))
    torch.testing.assert_close(metrics.get_value("beads_in_target"), torch.zeros(NUM_ENVS, device=env.device))
    torch.testing.assert_close(metrics.get_value("beads_spilled"), torch.zeros(NUM_ENVS, device=env.device))
    # Settled on the floor and on each other: none sunk into another by a millimetre.
    beads = env.entities["beads"].get_links_pos()
    gaps = torch.cdist(beads, beads) + torch.eye(_BEAD_COUNT, device=env.device) - 2 * _BEAD_RADIUS
    assert gaps.min() > -0.001

    goal = env.command_manager.get_term("lift_target").command
    offset = goal - (env.entities["target"].get_pos() - env.entities["robot"].get_pos())
    for axis, (lo, hi) in enumerate((_GOAL_FROM_TARGET_X, _GOAL_FROM_TARGET_Y, _GOAL_FROM_TARGET_Z)):
        assert ((offset[:, axis] >= lo - 1e-4) & (offset[:, axis] <= hi + 1e-4)).all(), (axis, offset)


#: Tipped 135 deg about +x, the source's mouth faces down toward -y, and its rim's lowest point is 0.9 cm toward -y and
#: 5.8 cm below its centre: from here, that point is 1.5 cm short of the target's axis, 2 cm above its rim. The beads
#: leave rolling toward -y, so short of the axis is where they drop in: over the axis, half of them hit the far rim.
_TIPPED = 135.0
_OVER_TARGET = (0.0, 0.024, 0.126)
_LIP_SHORT_OF_AXIS = 0.024 - (_BEAKER_HEIGHT / 2 - _BEAKER_RADIUS) * math.sin(math.radians(_TIPPED - 90.0))


def _hold_source(
    env, offset_from_target: tuple[float, float, float], steps: int
) -> tuple[torch.Tensor, dict[str, int]]:
    """Hold the source tipped at ``offset_from_target`` (put back there, at rest, every step): the most of the
    spilled share seen, and how many episodes each termination ended.
    """
    source = env.entities["beaker"]
    at = env.entities["target"].get_pos() + _rows(env, offset_from_target)
    # The beads stay where the source was: refill it where it now is.
    source.set_quat(_quat_about_x(env, _TIPPED))
    source.set_pos(at)
    env.event_manager.get_term("fill_beads").compute()
    most_spilled = torch.zeros(env.num_envs, device=env.device)
    ended: dict[str, int] = {}
    for _ in range(steps):
        source.set_quat(_quat_about_x(env, _TIPPED))
        source.set_pos(at)
        source.zero_all_dofs_velocity()
        extras = env.step(_zero_action(env))[4]
        for key, count in extras["log"].items():
            if key.startswith("Episode_Termination/") and int(count):
                name = key.removeprefix("Episode_Termination/")
                ended[name] = ended.get(name, 0) + int(count)
        most_spilled = torch.maximum(most_spilled, env.metric_manager.get_value("beads_spilled"))
    return most_spilled, ended


@pytest.mark.requires_genesis
def test_beads_poured_over_the_target_land_in_it(env):
    env.reset()
    most_spilled, ended = _hold_source(env, _OVER_TARGET, steps=60)
    metrics = env.metric_manager
    assert (metrics.get_value("beads_in_source") == 0.0).all()
    assert (metrics.get_value("beads_in_target") >= 0.5).all(), metrics.get_value("beads_in_target")
    assert (most_spilled <= 0.25).all(), most_spilled
    assert not ended, ended
    # The target stood: still within 5 degrees of upright.
    assert (metrics.get_value("target_tilt") < 1.0 - math.cos(math.radians(5.0))).all()
    env.reset()


@pytest.mark.requires_genesis
def test_beads_poured_beside_the_target_spill_and_end_the_episode(env):
    env.reset()
    # 15 cm toward the source: over the bench between the two.
    _, ended = _hold_source(env, (_OVER_TARGET[0], _OVER_TARGET[1] + 0.15, _OVER_TARGET[2]), steps=60)
    assert ended.get("beads_spilled", 0) >= NUM_ENVS, ended
    env.reset()


_LIP = {"source_name": "beaker", "target_name": "target", "radius": _BEAKER_RADIUS, "height": _BEAKER_HEIGHT}


@pytest.mark.requires_genesis
def test_the_pour_lip_is_the_rim_point_toward_the_target_upright_and_the_lowest_one_tipped(env):
    env.reset()
    source = env.entities["beaker"]
    # 15 cm on the target's +y side and 20 cm up.
    centre = env.entities["target"].get_pos() + _rows(env, (0.0, 0.15, 0.20))
    source.set_pos(centre)

    # Upright, every rim point is lowest: the one toward the target (-y) is the lip.
    source.set_quat(_quat_about_x(env, 0.0))
    upright = centre + _rows(env, (0.0, -_BEAKER_RADIUS, _BEAKER_HEIGHT / 2))
    torch.testing.assert_close(pour_lip(env, **_LIP), upright, atol=1e-4, rtol=0)

    # Tipped 60 deg away from the target (mouth toward +y): the lowest rim point, on the far side.
    source.set_quat(_quat_about_x(env, -60.0))
    s, c = math.sin(math.radians(60.0)), math.cos(math.radians(60.0))
    tipped = centre + _rows(
        env, (0.0, _BEAKER_HEIGHT / 2 * s + _BEAKER_RADIUS * c, _BEAKER_HEIGHT / 2 * c - _BEAKER_RADIUS * s)
    )
    torch.testing.assert_close(pour_lip(env, **_LIP), tipped, atol=1e-4, rtol=0)
    env.reset()


def _palm(env) -> torch.Tensor:
    """The IK's control point, which the goal terms turn a tipped beaker back upright about."""
    robot = env.entities["robot"]
    _, link = robot.find_named_links_idx_local([_EE_LINK], preserve_order=True)
    link = list(link)
    quat = robot.get_links_quat(ls_idx_local=link)[:, 0]
    return robot.get_links_pos(ls_idx_local=link)[:, 0] + transform_by_quat(_rows(env, _PALM_IN_WRIST), quat)


def _beaker_in_hand_at_goal(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Put the source upright on the open hand's grasp point (the lift gate wants it within 6 cm, and off the bench),
    and the goal there; its centre and the palm.
    """
    term = env.reward_manager.get_term("object_goal_tracking_fine")
    centre = _grasp_point_w(env, "robot", term.grasp_links)
    env.entities["beaker"].set_quat(_quat_about_x(env, 0.0))
    env.entities["beaker"].set_pos(centre)
    env.command_manager.get_term("lift_target").command.copy_(centre - env.entities["robot"].get_pos())
    return centre, _palm(env)


@pytest.mark.requires_genesis
def test_tipping_the_beaker_about_the_palm_keeps_it_at_the_goal_and_tipping_it_in_place_does_not(env):
    env.reset()
    term = env.reward_manager.get_term("object_goal_tracking_fine")
    source = env.entities["beaker"]
    centre, palm = _beaker_in_hand_at_goal(env)
    env.metric_manager.compute()
    torch.testing.assert_close(term.compute(), torch.ones(NUM_ENVS, device=env.device), atol=1e-4, rtol=0)

    # 45 deg about the palm: the centre swings, but turned back upright about the palm it is at the goal.
    tip = _quat_about_x(env, 45.0)
    source.set_quat(tip)
    source.set_pos(palm + transform_by_quat(centre - palm, tip))
    env.metric_manager.compute()
    torch.testing.assert_close(term.compute(), torch.ones(NUM_ENVS, device=env.device), atol=1e-4, rtol=0)

    # 90 deg about its own centre: turned back upright about the palm (a turn about x), it lands 2 r sin(45 deg) off
    # the goal, r the palm-to-centre distance across x, past the 2 cm deadband.
    source.set_quat(_quat_about_x(env, 90.0))
    source.set_pos(centre)
    env.metric_manager.compute()
    off = 2 * (centre - palm)[:, 1:].norm(dim=-1) * math.sin(math.radians(45.0))
    expected = 1.0 - torch.tanh((off - 0.02).clamp(min=0.0) / 0.05)
    assert (expected < 0.9).all(), off
    torch.testing.assert_close(term.compute(), expected, atol=1e-3, rtol=0)
    env.reset()


@pytest.mark.requires_genesis
def test_the_goal_terms_pay_in_full_while_the_beaker_pours_wherever_the_goal_is(env):
    env.reset()
    terms = [env.reward_manager.get_term(name) for name in ("object_goal_tracking", "object_goal_tracking_fine")]
    centre, _ = _beaker_in_hand_at_goal(env)
    env.command_manager.get_term("lift_target").command.add_(_rows(env, (0.0, 0.0, 0.20)))  # the goal 20 cm up
    env.entities["beaker"].set_quat(_quat_about_x(env, _TIPPED))
    target = env.entities["target"]
    for under_lip, pays_in_full in ((True, True), (False, False)):
        # The target under the tipped beaker's lip, as _OVER_TARGET has it, or 30 cm away.
        shift = (0.0, 0.0, 0.0) if under_lip else (0.0, -0.30, 0.0)
        target.set_pos(centre - _rows(env, _OVER_TARGET) + _rows(env, shift))
        env.metric_manager.compute()
        for term in terms:
            if pays_in_full:
                torch.testing.assert_close(term.compute(), torch.ones(NUM_ENVS, device=env.device), atol=1e-4, rtol=0)
            else:
                assert (term.compute() < 0.6).all(), (term, term.compute())
    env.reset()


def _quat_about_y(env, deg: float) -> torch.Tensor:
    half = math.radians(deg) / 2
    return _rows(env, (math.cos(half), 0.0, math.sin(half), 0.0))


#: Where the goal puts the source (mid-range): 5 cm from the target's axis toward the source (+y), 13 cm up.
_AT_GOAL = (0.0, 0.05, 0.13)


@pytest.mark.requires_genesis
@pytest.mark.parametrize("deg", [0.0, 45.0, 90.0, 120.0])
def test_from_the_goal_a_beaker_tipped_toward_the_target_about_its_centre_keeps_its_lip_over_the_mouth(env, deg):
    """How the lift policy tips a beaker it holds on its goal: about the beaker's centre."""
    env.reset()
    source = env.entities["beaker"]
    source.set_pos(env.entities["target"].get_pos() + _rows(env, _AT_GOAL))
    source.set_quat(_quat_about_x(env, deg))
    env.metric_manager.compute()
    assert (env.metric_manager.get_value("pour_lip_offset") < _INNER_RADIUS).all()
    assert (env.metric_manager.get_value("pour_lip_height") > 0.015).all()
    env.reset()


@pytest.mark.requires_genesis
@pytest.mark.parametrize(
    ("quat", "deg", "toward_deg"),
    [
        # About +x, the mouth turns toward -y, where the target is.
        (_quat_about_x, 40.0, 40.0),
        (_quat_about_x, -40.0, -40.0),
        # About +y, the mouth turns toward +x: across the line to the target.
        (_quat_about_y, 40.0, 0.0),
        (_quat_about_x, 180.0, 180.0),
    ],
)
def test_the_tilt_toward_the_target_is_the_tilt_in_its_direction(env, quat, deg, toward_deg):
    env.reset()
    source = env.entities["beaker"]
    source.set_pos(env.entities["target"].get_pos() + _rows(env, _AT_GOAL))
    source.set_quat(quat(env, deg))
    env.metric_manager.compute()
    toward = torch.rad2deg(env.metric_manager.get_value("beaker_tilt_toward_target"))
    torch.testing.assert_close(
        toward.abs() if toward_deg == 180.0 else toward,
        torch.full((NUM_ENVS,), toward_deg, device=env.device),
        atol=0.05,
        rtol=0,
    )
    env.reset()


@pytest.mark.requires_genesis
def test_pour_tilt_pays_the_tilt_with_the_lip_over_the_target_and_not_below_its_rim(env):
    env.reset()
    source, target = env.entities["beaker"], env.entities["target"]
    term = env.reward_manager.get_term("pour_tilt")
    source.set_quat(_quat_about_x(env, _TIPPED))
    for offset, pays in (
        # The lip 1.5 cm short of the target's axis, 2 cm over its rim: 135 / 180 of upside down, times
        # exp(-(1.5 / 10)^2).
        (_OVER_TARGET, 0.75 * math.exp(-((_LIP_SHORT_OF_AXIS / 0.10) ** 2))),
        # 4 cm lower, the lip is 2 cm below the target's rim: nothing.
        ((_OVER_TARGET[0], _OVER_TARGET[1], _OVER_TARGET[2] - 0.04), 0.0),
    ):
        source.set_pos(target.get_pos() + _rows(env, offset))
        env.metric_manager.compute()
        torch.testing.assert_close(term.compute(), torch.full((NUM_ENVS,), pays, device=env.device), atol=1e-3, rtol=0)
    env.reset()


@pytest.mark.requires_genesis
def test_pour_tilt_pays_a_first_tip_toward_the_target_and_nothing_for_one_across_or_away(env):
    """From where the goal puts the beaker, 40 deg each way."""
    env.reset()
    source, target = env.entities["beaker"], env.entities["target"]
    term = env.reward_manager.get_term("pour_tilt")
    source.set_pos(target.get_pos() + _rows(env, _AT_GOAL))
    pays = {}
    for name, quat in (
        ("toward", _quat_about_x(env, 40.0)),
        ("away", _quat_about_x(env, -40.0)),
        ("across", _quat_about_y(env, 40.0)),
    ):
        source.set_quat(quat)
        env.metric_manager.compute()
        pays[name] = term.compute()
    assert (pays["toward"] > 0.05).all(), pays
    assert (pays["away"].abs() < 1e-4).all() and (pays["across"].abs() < 1e-3).all(), pays
    env.reset()


@pytest.mark.requires_genesis
@pytest.mark.parametrize(
    ("quat", "deg", "costs"),
    [
        (_quat_about_x, 40.0, 0.0),  # toward the target
        (_quat_about_x, -40.0, math.sin(math.radians(40.0)) ** 2),  # away from it
        (_quat_about_y, 40.0, math.sin(math.radians(40.0)) ** 2),  # across the line to it
    ],
)
def test_leaning_across_or_away_from_the_target_costs_and_toward_it_does_not(env, quat, deg, costs):
    env.reset()
    source = env.entities["beaker"]
    term = env.reward_manager.get_term("beaker_tilt_off_target")
    source.set_quat(quat(env, deg))
    # In the air (held: clear of the bench), where the goal puts it.
    source.set_pos(env.entities["target"].get_pos() + _rows(env, _AT_GOAL))
    env.metric_manager.compute()
    torch.testing.assert_close(term.compute(), torch.full((NUM_ENVS,), costs, device=env.device), atol=1e-4, rtol=0)
    # Tipped on the bench where it rests, unheld: a knock, not a carry, and no cost.
    rest = env.entities["beaker"].get_pos().clone()
    rest[:, 2] = (
        _BENCH_TOP
        + _BEAKER_HEIGHT / 2 * math.cos(math.radians(deg))
        + _BEAKER_RADIUS * abs(math.sin(math.radians(deg)))
    )
    source.set_pos(rest)
    env.metric_manager.compute()
    torch.testing.assert_close(term.compute(), torch.zeros(NUM_ENVS, device=env.device), atol=1e-6, rtol=0)
    env.reset()
