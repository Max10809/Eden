"""Tests for ``eden.tasks.manipulation.g1_beaker_pour.warm_start``: a checkpoint widened for inputs appended to its
actor's and critic's loads into the wider rsl-rl models and optimizer, each network computes what it did whatever the
new inputs read, and a checkpoint of the wrong layout is refused.
"""

from __future__ import annotations

from itertools import chain

import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from eden.tasks.manipulation.g1_beaker_pour.warm_start import input_size, widen_input

N_ACTOR, N_CRITIC, ACTOR_EXTRA, CRITIC_EXTRA = 5, 7, 1, 2


def _models(n_actor: int, n_critic: int) -> tuple[MLPModel, MLPModel, torch.optim.Adam]:
    """An actor and a critic as rsl-rl's PPO builds them (normalized inputs, a Gaussian actor), and its optimizer."""
    obs = TensorDict({"policy": torch.zeros(4, n_actor), "critic": torch.zeros(4, n_critic)}, batch_size=[4])
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


def _trained_checkpoint() -> tuple[dict, MLPModel, MLPModel]:
    """A checkpoint after an update: normalizer statistics and Adam moments that are not their initial values."""
    torch.manual_seed(0)
    actor, critic, optimizer = _models(N_ACTOR, N_CRITIC)
    actor.obs_normalizer.update(torch.randn(64, N_ACTOR) * 2 - 1)
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
    return checkpoint, actor, critic


def _obs(policy: torch.Tensor, critic: torch.Tensor) -> TensorDict:
    return TensorDict({"policy": policy, "critic": critic}, batch_size=[policy.shape[0]])


def test_the_widened_networks_load_into_the_wider_models_and_compute_what_they_did():
    checkpoint, actor, critic = _trained_checkpoint()
    widen_input(checkpoint, "actor", ACTOR_EXTRA)
    widen_input(checkpoint, "critic", CRITIC_EXTRA)
    assert (input_size(checkpoint, "actor"), input_size(checkpoint, "critic")) == (
        N_ACTOR + ACTOR_EXTRA,
        N_CRITIC + CRITIC_EXTRA,
    )

    actor_w, critic_w, optimizer_w = _models(N_ACTOR + ACTOR_EXTRA, N_CRITIC + CRITIC_EXTRA)
    actor_w.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    critic_w.load_state_dict(checkpoint["critic_state_dict"], strict=True)
    optimizer_w.load_state_dict(checkpoint["optimizer_state_dict"])

    for model in (actor, critic, actor_w, critic_w):
        model.eval()
    p, c = torch.randn(16, N_ACTOR), torch.randn(16, N_CRITIC)
    narrow = _obs(p, c)
    for scale in (0.0, 10.0):
        wide = _obs(
            torch.cat([p, torch.randn(16, ACTOR_EXTRA) * scale], -1),
            torch.cat([c, torch.randn(16, CRITIC_EXTRA) * scale], -1),
        )
        torch.testing.assert_close(actor_w(wide), actor(narrow))
        torch.testing.assert_close(critic_w(wide), critic(narrow))

    # The new inputs' statistics are the identity, and their weights' Adam moments start at zero; a step runs.
    for model, n, extra in ((actor_w, N_ACTOR, ACTOR_EXTRA), (critic_w, N_CRITIC, CRITIC_EXTRA)):
        normalizer = model.obs_normalizer
        torch.testing.assert_close(normalizer._mean[:, n:], torch.zeros(1, extra))
        torch.testing.assert_close(normalizer._std[:, n:], torch.ones(1, extra))
        moments = optimizer_w.state[model.mlp[0].weight]
        assert moments["exp_avg"][:, :n].abs().sum() > 0
        assert (moments["exp_avg"][:, n:] == 0).all() and (moments["exp_avg_sq"][:, n:] == 0).all()
    (actor_w(wide).sum() + critic_w(wide).sum()).backward()
    optimizer_w.step()


def test_a_checkpoint_whose_optimizer_does_not_hold_the_actor_then_the_critic_is_refused():
    checkpoint, _, _ = _trained_checkpoint()
    state = checkpoint["optimizer_state_dict"]["state"]
    # Drop the actor's last parameter's entry: the critic's first-layer weight is no longer where the actor ends.
    n_actor = sum(not key.startswith("obs_normalizer.") for key in checkpoint["actor_state_dict"])
    del state[n_actor]
    with pytest.raises(ValueError, match=r"optimizer parameter \d+ should be the critic's first-layer weight"):
        widen_input(checkpoint, "critic", CRITIC_EXTRA)


@pytest.mark.parametrize(
    ("mutate", "model", "extra", "error", "match"),
    [
        (lambda ckpt: ckpt.pop("critic_state_dict"), "actor", 1, KeyError, "rsl-rl-lib 4.x/5.x PPO checkpoint"),
        (lambda ckpt: ckpt["critic_state_dict"].pop("obs_normalizer._std"), "critic", 1, KeyError, "normalizer"),
        (lambda ckpt: ckpt["actor_state_dict"].pop("mlp.0.weight"), "actor", 1, KeyError, "MLPModel"),
        (lambda ckpt: None, "critic", 0, ValueError, "extra_inputs must be >= 1"),
        (lambda ckpt: None, "policy", 1, ValueError, "model must be one of"),
    ],
)
def test_a_checkpoint_of_the_wrong_layout_is_refused(mutate, model, extra, error, match):
    checkpoint, _, _ = _trained_checkpoint()
    mutate(checkpoint)
    with pytest.raises(error, match=match):
        widen_input(checkpoint, model, extra)
