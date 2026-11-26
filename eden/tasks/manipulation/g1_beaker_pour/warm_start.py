"""Start ``manipulation_g1_beaker_pour`` from a ``manipulation_g1_beaker_lift`` checkpoint.

The pour task's actor observes what the lift's does, and its critic gets the lift's critic inputs and, after them,
:data:`~eden.tasks.manipulation.g1_beaker_pour.config.CRITIC_EXTRA_DIMS` more (the bead counts). A lift checkpoint's
critic is widened for them by :func:`widen_critic_input`: zero weights for the new inputs in its first layer, the
identity (mean 0, variance 1) in its observation normalizer, and zero Adam moments for the new weights. The widened
critic computes what the lift's did whatever the new inputs read, and learns to use them from there.

usage::

    uv run python -m eden.tasks.manipulation.g1_beaker_pour.warm_start <lift run>/model_<N>.pt <out>.pt
    uv run eden train --task manipulation_g1_beaker_pour --checkpoint <out>.pt
"""

from __future__ import annotations

import argparse
import os

import torch

#: The critic's first layer and its observation normalizer's per-input statistics, in an rsl-rl-lib 4.x/5.x
#: ``critic_state_dict``: the normalizer's ``_mean``, ``_var`` and ``_std`` are ``(1, n_inputs)``.
_FIRST_LAYER = "mlp.0.weight"
_NORMALIZER_FILL = {"obs_normalizer._mean": 0.0, "obs_normalizer._var": 1.0, "obs_normalizer._std": 1.0}
_NORMALIZER_PREFIX = "obs_normalizer."


def widen_critic_input(checkpoint: dict, extra_inputs: int) -> dict:
    """``checkpoint`` (an rsl-rl-lib 4.x/5.x PPO checkpoint, as ``torch.load`` returns it) with ``extra_inputs``
    critic inputs appended after its own, changed in place and returned.

    The optimizer's parameters are the actor's and then the critic's (rsl-rl's PPO builds it that way), so the
    critic's first-layer weight is the parameter after the actor's last; its Adam moments get zero columns too.
    """
    if extra_inputs < 1:
        raise ValueError(f"widen_critic_input: extra_inputs must be >= 1, got {extra_inputs}.")
    missing = [
        key for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict") if key not in checkpoint
    ]
    if missing:
        raise KeyError(
            f"widen_critic_input: the checkpoint has no {missing} (its keys: {sorted(checkpoint)}); it must be an "
            f"rsl-rl-lib 4.x/5.x PPO checkpoint, which splits actor_state_dict and critic_state_dict."
        )
    critic = checkpoint["critic_state_dict"]
    needed = [_FIRST_LAYER, *_NORMALIZER_FILL]
    if any(key not in critic for key in needed):
        raise KeyError(
            f"widen_critic_input: critic_state_dict lacks {[key for key in needed if key not in critic]} (its keys: "
            f"{sorted(critic)}); the critic must be an MLP with an observation normalizer (critic obs_normalization)."
        )
    weight = critic[_FIRST_LAYER]
    hidden, n_inputs = weight.shape
    critic[_FIRST_LAYER] = torch.cat([weight, weight.new_zeros(hidden, extra_inputs)], dim=1)
    for key, fill in _NORMALIZER_FILL.items():
        stat = critic[key]
        critic[key] = torch.cat([stat, stat.new_full((*stat.shape[:-1], extra_inputs), fill)], dim=-1)

    # The actor's parameters are its state dict's entries less the normalizer's buffers.
    index = sum(not key.startswith(_NORMALIZER_PREFIX) for key in checkpoint["actor_state_dict"])
    state = checkpoint["optimizer_state_dict"]["state"].get(index)
    if state is None or tuple(state["exp_avg"].shape) != (hidden, n_inputs):
        found = None if state is None else tuple(state["exp_avg"].shape)
        raise ValueError(
            f"widen_critic_input: optimizer parameter {index} should be the critic's first-layer weight, of shape "
            f"{(hidden, n_inputs)}, but its Adam moments are {found}. The optimizer must hold the actor's parameters "
            f"and then the critic's, as rsl-rl's PPO does."
        )
    for moment in ("exp_avg", "exp_avg_sq"):
        state[moment] = torch.cat([state[moment], state[moment].new_zeros(hidden, extra_inputs)], dim=1)
    return checkpoint


def main(argv: list[str] | None = None) -> None:
    from eden.tasks.manipulation.g1_beaker_pour.config import CRITIC_EXTRA_DIMS

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", help="a manipulation_g1_beaker_lift checkpoint (model_<N>.pt)")
    parser.add_argument("out", help="where to write the widened checkpoint (a new file)")
    args = parser.parse_args(argv)
    if os.path.abspath(args.out) == os.path.abspath(args.checkpoint):
        parser.error(f"out must be a new file, not the checkpoint itself ({args.checkpoint}).")
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    n_inputs = checkpoint.get("critic_state_dict", {}).get(_FIRST_LAYER, torch.empty(0, 0)).shape[-1]
    torch.save(widen_critic_input(checkpoint, CRITIC_EXTRA_DIMS), args.out)
    print(f"{args.out}: critic inputs {n_inputs} -> {n_inputs + CRITIC_EXTRA_DIMS}, actor unchanged.")


if __name__ == "__main__":
    main()
