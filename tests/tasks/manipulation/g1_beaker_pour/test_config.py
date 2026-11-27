"""Tests for ``eden.tasks.manipulation.g1_beaker_pour.config``: the actor's observations are the lift's and the
critic's add the bead counts last, the beads start at rest in the source, the target is placed toward the middle
from the source and the goal is drawn on its right, the upper arm touching the torso is a contact and a cost and the
ready arm touches nothing, the hand's pads resist spin, beads poured over the target land in it and beads poured beside it spill and end the episode, the pour lip, the goal terms
measured to the beaker turned back upright about the palm and paid in full while it pours, the tilt toward the
target, the pour-tilt gate and the cost of leaning across or away from the target; once the source is poured, the
goal over its home and then on it, the pour's terms giving way to putting it back, the source placed and released,
and a gated term's own weight refused; on the built task. The right hand pours inward: toward +y, from the target's
right.
"""

from __future__ import annotations

import math

import pytest
import torch

from eden.managers import REWARD_TERM_REGISTRY
from eden.managers.base import build_term
from eden.tasks import TASK_REGISTRY
from eden.tasks.manipulation.custom_terms.common import _base_to_world, _grasp_point_w, _to_base_frame
from eden.tasks.manipulation.g1_beaker_lift.config import (
    _ARM_DOFS,
    _BENCH_TOP,
    _EE_LINK,
    _GRASP,
    _HAND_LINKS,
    _PALM_IN_WRIST,
    ManipulationG1BeakerLiftConfig,
)
from eden.tasks.manipulation.g1_beaker_pour.config import (
    _BEAD_COUNT,
    _BEAD_RADIUS,
    _BEAKER_HEIGHT,
    _BEAKER_HOME,
    _BEAKER_RADIUS,
    _GOAL_FROM_TARGET_X,
    _GOAL_FROM_TARGET_Y,
    _GOAL_FROM_TARGET_Z,
    _INNER_RADIUS,
    _MIN_GAP_Y,
    _PAD_TORSION,
    _SOURCE_PLACE_Y,
    _TARGET_HOME,
    _TARGET_PLACE_X,
    _TARGET_PLACE_Y,
    ACTOR_EXTRA_DIMS,
    CRITIC_EXTRA_DIMS,
    ManipulationG1BeakerPourConfig,
    PourPhaseGate,
    pour_lip,
)
from eden.utils.geom import transform_by_quat
from tests.support.envs import build_env

NUM_ENVS = 2
#: The critic's inputs beyond the lift's, last and in this order (a lift checkpoint is widened for them at the end).
BEAD_TERMS = ["beads_in_source", "beads_in_target", "beads_spilled", "beads_from_mouth"]
#: Both groups' last input: whether the source is poured (an earlier pour checkpoint is widened for it).
PHASE_TERMS = ["pour_done"]
#: The lift's actor and critic input sizes, which a lift checkpoint is built for.
LIFT_ACTOR_DIMS, LIFT_CRITIC_DIMS = 83, 98


def _term_names(group) -> list[str]:
    return [k for k, v in (group.__pydantic_extra__ or {}).items() if not isinstance(v, str)]


def test_the_actor_observes_what_the_lift_actor_does_the_critic_adds_the_bead_counts_and_both_the_pour_last():
    lift = ManipulationG1BeakerLiftConfig().observation_options
    pour = ManipulationG1BeakerPourConfig().observation_options
    assert _term_names(pour.policy) == _term_names(lift.policy) + PHASE_TERMS
    assert _term_names(pour.critic) == _term_names(lift.critic) + BEAD_TERMS + PHASE_TERMS


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


def _quat_toward(env, deg: float) -> torch.Tensor:
    """Tipped ``deg`` about -x: the mouth turns toward +y, the way the right hand pours (negative: toward -y)."""
    return _quat_about_x(env, -deg)


@pytest.mark.requires_genesis
def test_the_inputs_are_the_lifts_sizes_and_the_added_terms_the_widened_columns(env):
    obs = env.observation_manager
    for group, added, extra in (
        ("policy", PHASE_TERMS, ACTOR_EXTRA_DIMS),
        ("critic", BEAD_TERMS + PHASE_TERMS, CRITIC_EXTRA_DIMS),
    ):
        names, dims = obs.group_term_names[group], obs.group_obs_term_dim[group]
        assert names[-len(added) :] == added
        assert sum(math.prod(dim) for dim in dims[-len(added) :]) == extra
    assert obs.group_obs_dim["policy"] == (LIFT_ACTOR_DIMS + ACTOR_EXTRA_DIMS,)
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


@pytest.mark.requires_genesis
def test_the_target_is_placed_toward_the_middle_from_the_source_and_clear_of_it(env):
    events = env.event_manager
    sources, targets = [], []
    for _ in range(60):
        events.get_term("place_beaker").compute(envs_idx=None)
        events.get_term("place_target").compute(envs_idx=None)
        sources.append(env.entities["beaker"].get_pos())
        targets.append(env.entities["target"].get_pos())
    sources, targets = torch.cat(sources), torch.cat(targets)
    # Always on the source's +y side (the middle, where the right hand pours toward), 12 cm across or more.
    gaps = targets[:, 1] - sources[:, 1]
    assert (gaps >= _MIN_GAP_Y - 1e-5).all(), gaps.min()
    # Near the source and far from it, both often.
    assert (gaps < 0.20).float().mean() > 0.2 and (gaps > 0.25).float().mean() > 0.1, gaps
    assert (sources[:, 1] <= _BEAKER_HOME[1] + _SOURCE_PLACE_Y[1] + 1e-5).all()
    for axis, (lo, hi) in ((0, _TARGET_PLACE_X), (1, _TARGET_PLACE_Y)):
        assert (targets[:, axis] >= _TARGET_HOME[axis] + lo - 1e-5).all()
        assert (targets[:, axis] <= _TARGET_HOME[axis] + hi + 1e-5).all()
    torch.testing.assert_close(targets[:, 2], torch.full_like(targets[:, 2], _TARGET_HOME[2]), atol=1e-5, rtol=0)
    env.reset()


@pytest.mark.requires_genesis
def test_the_upper_arm_pulled_into_the_torso_touches_it_and_costs_and_the_ready_arm_touches_nothing(env):
    env.reset()
    robot = env.entities["robot"]
    arm = robot.find_named_dofs(_ARM_DOFS)
    q = robot.get_dofs_pos()[..., arm.columns].clone()
    # Env 0: the shoulder rolled 0.45 rad in toward the body (the ready pose has it 0.15 rad out), pitch at 0. Env 1
    # stays ready.
    q[0, 0], q[0, 1], q[0, 2] = 0.0, 0.45, 0.0
    robot.set_dofs_pos(q, dofs_idx_local=arm.dofs_idx_local, zero_velocity=True)
    env.step(_zero_action(env))
    contacts = robot.get_contacts(with_entity=robot)
    names = [link.name for link in robot.links]
    a, b = contacts["link_a"] - robot.link_start, contacts["link_b"] - robot.link_start
    valid = contacts.get("valid_mask", torch.ones_like(a, dtype=torch.bool))
    upper, torso = names.index("right_shoulder_yaw_link"), names.index("torso_link")
    pressed = (valid & (((a == upper) & (b == torso)) | ((a == torso) & (b == upper)))).any(-1)
    assert pressed.tolist() == [True, False]
    cost = env.reward_manager.get_term("arm_on_torso").compute()
    torch.testing.assert_close(cost, torch.tensor([1.0, 0.0], device=env.device))
    env.reset()


@pytest.mark.requires_genesis
def test_the_hands_links_and_only_they_resist_spin(env):
    robot = env.entities["robot"]
    torsional = env.rigid_solver.dyn_info.geoms.friction_torsional.to_numpy()
    by_link: dict[str, float] = {}
    for geom in robot.geoms:
        by_link[geom.link.name] = max(by_link.get(geom.link.name, 0.0), float(torsional[geom.idx]))
    assert env.env_options.enable_torsional_friction
    assert {link: by_link[link] for link in _HAND_LINKS} == dict.fromkeys(_HAND_LINKS, pytest.approx(_PAD_TORSION))
    assert all(value == 0.0 for link, value in by_link.items() if link not in _HAND_LINKS), by_link


#: Tipped 135 deg toward +y (about -x), the source's mouth faces down toward +y, and its rim's lowest point is 0.9 cm
#: toward +y and 5.8 cm below its centre: from here, that point is 1.5 cm short of the target's axis, 2 cm above its
#: rim. The beads leave rolling toward +y, so short of the axis is where they drop in: over the axis, half of them hit
#: the far rim.
_TIPPED = 135.0
_OVER_TARGET = (0.0, -0.024, 0.126)
_LIP_SHORT_OF_AXIS = 0.024 - (_BEAKER_HEIGHT / 2 - _BEAKER_RADIUS) * math.sin(math.radians(_TIPPED - 90.0))


def _hold_source(
    env, offset_from_target: tuple[float, float, float], steps: int
) -> tuple[torch.Tensor, dict[str, int]]:
    """Hold the source tipped at ``offset_from_target`` (put back there, at rest, every step), the target at the far
    end of its band (y +0.10): the most of the spilled share seen, and how many episodes each termination ended. Held
    over a target near y -0.13, the source would stand against the raised hand and be flung.
    """
    source, target = env.entities["beaker"], env.entities["target"]
    target.set_pos(_rows(env, (_TARGET_HOME[0], _TARGET_HOME[1] + _TARGET_PLACE_Y[1], _TARGET_HOME[2])))
    target.zero_all_dofs_velocity()
    at = target.get_pos() + _rows(env, offset_from_target)
    # The beads stay where the source was: refill it where it now is.
    source.set_quat(_quat_toward(env, _TIPPED))
    source.set_pos(at)
    env.event_manager.get_term("fill_beads").compute()
    most_spilled = torch.zeros(env.num_envs, device=env.device)
    ended: dict[str, int] = {}
    for _ in range(steps):
        source.set_quat(_quat_toward(env, _TIPPED))
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
    # 15 cm further toward -y, the side it pours from: over the bench.
    _, ended = _hold_source(env, (_OVER_TARGET[0], _OVER_TARGET[1] - 0.15, _OVER_TARGET[2]), steps=60)
    assert ended.get("beads_spilled", 0) >= NUM_ENVS, ended
    env.reset()


_LIP = {"source_name": "beaker", "target_name": "target", "radius": _BEAKER_RADIUS, "height": _BEAKER_HEIGHT}


@pytest.mark.requires_genesis
def test_the_pour_lip_is_the_rim_point_toward_the_target_upright_and_the_lowest_one_tipped(env):
    env.reset()
    source = env.entities["beaker"]
    # 15 cm on the target's -y side and 20 cm up.
    centre = env.entities["target"].get_pos() + _rows(env, (0.0, -0.15, 0.20))
    source.set_pos(centre)

    # Upright, every rim point is lowest: the one toward the target (+y) is the lip.
    source.set_quat(_quat_toward(env, 0.0))
    upright = centre + _rows(env, (0.0, _BEAKER_RADIUS, _BEAKER_HEIGHT / 2))
    torch.testing.assert_close(pour_lip(env, **_LIP), upright, atol=1e-4, rtol=0)

    # Tipped 60 deg away from the target (mouth toward -y): the lowest rim point, on the far side.
    source.set_quat(_quat_toward(env, -60.0))
    s, c = math.sin(math.radians(60.0)), math.cos(math.radians(60.0))
    tipped = centre + _rows(
        env, (0.0, -(_BEAKER_HEIGHT / 2 * s + _BEAKER_RADIUS * c), _BEAKER_HEIGHT / 2 * c - _BEAKER_RADIUS * s)
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
    centre = _grasp_point_w(env, "robot", _GRASP["grasp_links"])
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
    # The target 30 cm on, on the bench: no tilt here pours into it (which the goal terms pay in full).
    target = env.entities["target"]
    away = target.get_pos().clone()
    away[:, 1] = centre[:, 1] + 0.30
    target.set_pos(away)
    env.metric_manager.compute()
    torch.testing.assert_close(term.compute(), torch.ones(NUM_ENVS, device=env.device), atol=1e-4, rtol=0)

    # 45 deg about the palm: the centre swings, but turned back upright about the palm it is at the goal.
    tip = _quat_toward(env, 45.0)
    source.set_quat(tip)
    source.set_pos(palm + transform_by_quat(centre - palm, tip))
    env.metric_manager.compute()
    torch.testing.assert_close(term.compute(), torch.ones(NUM_ENVS, device=env.device), atol=1e-4, rtol=0)

    # 90 deg about its own centre: turned back upright about the palm (a turn about x), it lands 2 r sin(45 deg) off
    # the goal, r the palm-to-centre distance across x, past the 2 cm deadband.
    source.set_quat(_quat_toward(env, 90.0))
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
    env.entities["beaker"].set_quat(_quat_toward(env, _TIPPED))
    target = env.entities["target"]
    for under_lip, pays_in_full in ((True, True), (False, False)):
        # The target under the tipped beaker's lip, as _OVER_TARGET has it, or 30 cm away.
        shift = (0.0, 0.0, 0.0) if under_lip else (0.0, 0.30, 0.0)
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


#: Where the goal puts the source (mid-range): 5 cm from the target's axis toward -y, 13 cm up.
_AT_GOAL = (0.0, -0.05, 0.13)


@pytest.mark.requires_genesis
@pytest.mark.parametrize("deg", [0.0, 45.0, 90.0, 120.0])
def test_from_the_goal_a_beaker_tipped_toward_the_target_about_its_centre_keeps_its_lip_over_the_mouth(env, deg):
    """How the lift policy tips a beaker it holds on its goal: about the beaker's centre."""
    env.reset()
    source = env.entities["beaker"]
    source.set_pos(env.entities["target"].get_pos() + _rows(env, _AT_GOAL))
    source.set_quat(_quat_toward(env, deg))
    env.metric_manager.compute()
    assert (env.metric_manager.get_value("pour_lip_offset") < _INNER_RADIUS).all()
    assert (env.metric_manager.get_value("pour_lip_height") > 0.015).all()
    env.reset()


@pytest.mark.requires_genesis
@pytest.mark.parametrize(
    ("quat", "deg", "toward_deg"),
    [
        # About -x, the mouth turns toward +y, where the target is.
        (_quat_toward, 40.0, 40.0),
        (_quat_toward, -40.0, -40.0),
        # About +y, the mouth turns toward +x: across the line to the target.
        (_quat_about_y, 40.0, 0.0),
        (_quat_toward, 180.0, 180.0),
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
    source.set_quat(_quat_toward(env, _TIPPED))
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
        ("toward", _quat_toward(env, 40.0)),
        ("away", _quat_toward(env, -40.0)),
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
        (_quat_toward, 40.0, 0.0),  # toward the target
        (_quat_toward, -40.0, math.sin(math.radians(40.0)) ** 2),  # away from it
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


def _upright(env) -> torch.Tensor:
    return _quat_about_x(env, 0.0)


@pytest.mark.requires_genesis
def test_once_poured_the_goal_goes_over_the_sources_home_then_onto_it_and_a_reset_brings_it_back_by_the_target(env):
    env.reset()
    command, source = env.command_manager.get_term("lift_target"), env.entities["beaker"]
    torch.testing.assert_close(command.home_b, _to_base_frame(env, "robot", source.get_pos()), atol=1e-4, rtol=0)
    assert not command.poured.any() and not command.over_home.any()
    torch.testing.assert_close(command.command, command.oracle_pos_b)

    # Poured over the target: the goal is 12 cm over where the source stood.
    _hold_source(env, _OVER_TARGET, steps=60)
    assert command.poured.all() and not command.over_home.any()
    torch.testing.assert_close(command.command, command.home_b + _rows(env, (0.0, 0.0, 0.12)))
    torch.testing.assert_close(
        env.observation_manager.compute()["policy"][:, -1], torch.ones(NUM_ENVS, device=env.device)
    )

    # The source upright there: the goal turns down, onto the home.
    source.set_quat(_upright(env))
    source.set_pos(_base_to_world(env, "robot", command.over_home_b))
    source.zero_all_dofs_velocity()
    env.step(_zero_action(env))
    assert command.over_home.all()
    torch.testing.assert_close(command.command, command.home_b)

    env.reset()
    assert not command.poured.any() and not command.over_home.any()
    torch.testing.assert_close(command.command, command.oracle_pos_b)
    torch.testing.assert_close(
        env.observation_manager.compute()["policy"][:, -1], torch.zeros(NUM_ENVS, device=env.device)
    )


@pytest.mark.requires_genesis
def test_the_pour_pays_only_until_the_source_is_poured_and_putting_it_back_only_after(env):
    env.reset()
    command, source, target = (
        env.command_manager.get_term("lift_target"),
        env.entities["beaker"],
        env.entities["target"],
    )
    pour_tilt, home_tracking = (env.reward_manager.get_term(name) for name in ("pour_tilt", "home_tracking"))
    source.set_quat(_quat_toward(env, _TIPPED))
    source.set_pos(target.get_pos() + _rows(env, _OVER_TARGET))
    env.metric_manager.compute()
    assert (pour_tilt.compute() > 0.5).all() and (home_tracking.compute() == 0.0).all()
    command.poured.fill_(True)
    env.metric_manager.compute()
    assert (pour_tilt.compute() == 0.0).all()
    # Tipped past 90 degrees, it is no nearer being put back.
    assert (home_tracking.compute() == 0.0).all()
    source.set_quat(_upright(env))
    assert (home_tracking.compute() > 0.0).all()
    env.reset()


@pytest.mark.requires_genesis
def test_the_source_upright_on_its_home_is_placed_released_once_the_hand_is_off_it_and_then_the_arm_pays_at_rest(env):
    env.reset()
    command, source, metrics = (
        env.command_manager.get_term("lift_target"),
        env.entities["beaker"],
        env.metric_manager,
    )
    rewards = env.reward_manager
    # Not poured: standing on its home is where it started, not put back.
    env.step(_zero_action(env))
    assert (metrics.get_value("source_placed") == 0.0).all()
    assert (rewards.get_term("arm_at_rest").compute() == 0.0).all()

    # Poured, standing on its home, the raised hand nowhere near it: placed and released, by the direct way.
    command.poured.fill_(True)
    env.step(_zero_action(env))
    assert command.over_home.all()
    for name in ("source_placed", "source_released"):
        assert (metrics.get_value(name) == 1.0).all(), name
    assert (rewards.get_term("home_tracking_fine").compute() > 0.9).all()
    # The arm where it starts: at its default pose within the limits, sagging into the deadband.
    assert (rewards.get_term("arm_at_rest").compute() > 0.95).all()

    home = source.get_pos().clone()
    for pos, quat in (
        (home + _rows(env, (0.0, 0.03, 0.0)), _upright(env)),  # 3 cm off its home
        (home, _quat_about_x(env, 20.0)),  # tilted 20 degrees
    ):
        source.set_pos(pos)
        source.set_quat(quat)
        metrics.compute()
        for name in ("source_placed", "source_released"):
            assert (metrics.get_value(name) == 0.0).all(), name
        assert (rewards.get_term("arm_at_rest").compute() == 0.0).all()
    env.reset()


@pytest.mark.requires_genesis
def test_a_gated_term_with_a_weight_of_its_own_is_refused(env):
    lifting = ManipulationG1BeakerPourConfig().reward_options.lifting_object
    with pytest.raises(ValueError, match=r"the gated term 'object_is_lifted' has weight=5.0"):
        build_term(
            REWARD_TERM_REGISTRY,
            env,
            PourPhaseGate.configure(weight=5.0, term=lifting.term.updated(weight=5.0), command_name="lift_target"),
            label="test",
        )
