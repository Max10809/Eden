"""Tests for ``eden.tasks.manipulation.g1_beaker_pour.warm_start``: a lift checkpoint widened for the pour critic's
extra inputs loads into the wider rsl-rl models and optimizer, its critic computes what it did whatever the new
inputs read, and a checkpoint of the wrong layout is refused.
"""

from __future__ import annotations

from itertools import chain

import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from eden.tasks.manipulation.g1_beaker_pour.warm_start import widen_critic_input

N_ACTOR, N_CRITIC, EXTRA = 5, 7, 2


def _models(n_critic: int) -> tuple[MLPModel, MLPModel, torch.optim.Adam]:
    """An actor and a critic as rsl-rl's PPO builds them (normalized inputs, a Gaussian actor), and its optimizer."""
    obs = TensorDict({"policy": torch.zeros(4, N_ACTOR), "critic": torch.zeros(4, n_critic)}, batch_size=[4])
    groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = MLPModel(
        obs,
        groups,
        "actor",
        3,
        hidden_dims=[16, 8],
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[16, 8], obs_normalization=True)
    return actor, critic, torch.optim.Adam(list(chain(actor.parameters(), critic.parameters())), lr=1e-3)


def _trained_checkpoint() -> tuple[dict, MLPModel]:
    """A checkpoint after an update: normalizer statistics and Adam moments that are not their initial values."""
    torch.manual_seed(0)
    actor, critic, optimizer = _models(N_CRITIC)
    critic.obs_normalizer.update(torch.randn(64, N_CRITIC) * 3 + 1)
    obs = TensorDict({"policy": torch.randn(32, N_ACTOR), "critic": torch.randn(32, N_CRITIC)}, batch_size=[32])
    (actor(obs).square().mean() + critic(obs).square().mean()).backward()
    optimizer.step()
    checkpoint = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iter": 7,
        "infos": None,
    }
    return checkpoint, critic


def test_the_widened_critic_loads_into_the_wider_models_and_computes_what_it_did():
    checkpoint, critic = _trained_checkpoint()
    actor_before = {k: v.clone() for k, v in checkpoint["actor_state_dict"].items()}
    widened = widen_critic_input(checkpoint, EXTRA)

    actor_w, critic_w, optimizer_w = _models(N_CRITIC + EXTRA)
    actor_w.load_state_dict(widened["actor_state_dict"], strict=True)
    critic_w.load_state_dict(widened["critic_state_dict"], strict=True)
    optimizer_w.load_state_dict(widened["optimizer_state_dict"])
    for key, value in actor_before.items():
        torch.testing.assert_close(widened["actor_state_dict"][key], value)

    critic.eval()
    critic_w.eval()
    x = torch.randn(16, N_CRITIC)
    narrow = TensorDict({"policy": torch.zeros(16, N_ACTOR), "critic": x}, batch_size=[16])
    for new in (torch.zeros(16, EXTRA), torch.randn(16, EXTRA) * 10):
        wide = TensorDict({"policy": torch.zeros(16, N_ACTOR), "critic": torch.cat([x, new], -1)}, batch_size=[16])
        torch.testing.assert_close(critic_w(wide), critic(narrow))

    # The new inputs' statistics are the identity, and their weights' Adam moments start at zero; a step runs.
    normalizer = critic_w.obs_normalizer
    torch.testing.assert_close(normalizer._mean[:, N_CRITIC:], torch.zeros(1, EXTRA))
    torch.testing.assert_close(normalizer._std[:, N_CRITIC:], torch.ones(1, EXTRA))
    moments = optimizer_w.state[critic_w.mlp[0].weight]
    assert moments["exp_avg"][:, :N_CRITIC].abs().sum() > 0
    assert (moments["exp_avg"][:, N_CRITIC:] == 0).all() and (moments["exp_avg_sq"][:, N_CRITIC:] == 0).all()
    critic_w(wide).sum().backward()
    optimizer_w.step()


def test_a_checkpoint_whose_optimizer_does_not_hold_the_actor_then_the_critic_is_refused():
    checkpoint, _ = _trained_checkpoint()
    state = checkpoint["optimizer_state_dict"]["state"]
    # Drop the actor's last parameter's entry: the critic's first-layer weight is no longer where the actor ends.
    n_actor = sum(not key.startswith("obs_normalizer.") for key in checkpoint["actor_state_dict"])
    del state[n_actor]
    with pytest.raises(ValueError, match=r"optimizer parameter \d+ should be the critic's first-layer weight"):
        widen_critic_input(checkpoint, EXTRA)


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    [
        (lambda ckpt: ckpt.pop("critic_state_dict"), KeyError, "rsl-rl-lib 4.x/5.x PPO checkpoint"),
        (lambda ckpt: ckpt["critic_state_dict"].pop("obs_normalizer._std"), KeyError, "observation normalizer"),
        (lambda ckpt: None, ValueError, "extra_inputs must be >= 1"),
    ],
)
def test_a_checkpoint_of_the_wrong_layout_is_refused(mutate, error, match):
    checkpoint, _ = _trained_checkpoint()
    mutate(checkpoint)
    with pytest.raises(error, match=match):
        widen_critic_input(checkpoint, 0 if error is ValueError else EXTRA)
