"""Start ``manipulation_g1_beaker_pour`` from a checkpoint whose actor and critic observe less: a
``manipulation_g1_beaker_lift`` checkpoint, or one of an earlier pour task that did not yet put the source back.

The pour task's observations are its predecessors' with inputs appended at the end of each group: the actor's are the
lift's and :data:`~eden.tasks.manipulation.g1_beaker_pour.config.ACTOR_EXTRA_DIMS` more (whether the source is
poured), the critic's the lift's and :data:`~eden.tasks.manipulation.g1_beaker_pour.config.CRITIC_EXTRA_DIMS` more
(the bead counts, then whether the source is poured). :func:`widen_input` widens a network for inputs appended at the
end: zero weights for them in its first layer, the identity (mean 0, variance 1) in its observation normalizer, and
zero Adam moments for the new weights. The widened network computes what it did whatever the new inputs read, and
learns to use them from there. :func:`main` widens the actor and the critic to the task's observation sizes.

usage::

    uv run python -m eden.tasks.manipulation.g1_beaker_pour.warm_start <run>/model_<N>.pt <out>.pt
    uv run eden train --task manipulation_g1_beaker_pour --checkpoint <out>.pt
"""

from __future__ import annotations

import argparse
import os

import torch

#: A network's first layer and its observation normalizer's per-input statistics, in an rsl-rl-lib 4.x/5.x
#: ``actor_state_dict`` / ``critic_state_dict``: the normalizer's ``_mean``, ``_var`` and ``_std`` are ``(1, n_inputs)``.
_FIRST_LAYER = "mlp.0.weight"
_NORMALIZER_FILL = {"obs_normalizer._mean": 0.0, "obs_normalizer._var": 1.0, "obs_normalizer._std": 1.0}
_NORMALIZER_PREFIX = "obs_normalizer."
_MODELS = ("actor", "critic")


def _parameter_keys(state_dict: dict) -> list[str]:
    """A network's parameters, in the order its optimizer holds them: its state dict's entries less the normalizer's
    buffers.
    """
    return [key for key in state_dict if not key.startswith(_NORMALIZER_PREFIX)]


def input_size(checkpoint: dict, model: str) -> int:
    """How many inputs ``model`` (``"actor"`` or ``"critic"``) of ``checkpoint`` takes."""
    return checkpoint[f"{model}_state_dict"][_FIRST_LAYER].shape[-1]


def widen_input(checkpoint: dict, model: str, extra_inputs: int) -> dict:
    """``checkpoint`` (an rsl-rl-lib 4.x/5.x PPO checkpoint, as ``torch.load`` returns it) with ``extra_inputs``
    inputs appended after ``model``'s own (``"actor"`` or ``"critic"``), changed in place and returned.

    The optimizer's parameters are the actor's and then the critic's (rsl-rl's PPO builds it that way), so the
    critic's are numbered after the actor's last; the first layer's Adam moments get zero columns too.
    """
    if model not in _MODELS:
        raise ValueError(f"widen_input: model must be one of {_MODELS}, got {model!r}.")
    if extra_inputs < 1:
        raise ValueError(f"widen_input: extra_inputs must be >= 1, got {extra_inputs}.")
    missing = [
        key for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict") if key not in checkpoint
    ]
    if missing:
        raise KeyError(
            f"widen_input: the checkpoint has no {missing} (its keys: {sorted(checkpoint)}); it must be an "
            f"rsl-rl-lib 4.x/5.x PPO checkpoint, which splits actor_state_dict and critic_state_dict."
        )
    network = checkpoint[f"{model}_state_dict"]
    if _FIRST_LAYER not in network:
        raise KeyError(
            f"widen_input: {model}_state_dict has no {_FIRST_LAYER!r} (its keys: {sorted(network)}); the {model} must "
            f"be an rsl-rl MLPModel."
        )
    stats = [key for key in _NORMALIZER_FILL if key in network]
    if stats and len(stats) != len(_NORMALIZER_FILL):
        raise KeyError(
            f"widen_input: {model}_state_dict has the observation normalizer's {stats} but not "
            f"{[key for key in _NORMALIZER_FILL if key not in stats]}; an rsl-rl EmpiricalNormalization holds all three."
        )
    weight = network[_FIRST_LAYER]
    hidden, n_inputs = weight.shape
    network[_FIRST_LAYER] = torch.cat([weight, weight.new_zeros(hidden, extra_inputs)], dim=1)
    for key, fill in _NORMALIZER_FILL.items():
        if key in network:
            stat = network[key]
            network[key] = torch.cat([stat, stat.new_full((*stat.shape[:-1], extra_inputs), fill)], dim=-1)

    before = len(_parameter_keys(checkpoint["actor_state_dict"])) if model == "critic" else 0
    index = before + _parameter_keys(network).index(_FIRST_LAYER)
    state = checkpoint["optimizer_state_dict"]["state"].get(index)
    if state is None or tuple(state["exp_avg"].shape) != (hidden, n_inputs):
        found = None if state is None else tuple(state["exp_avg"].shape)
        raise ValueError(
            f"widen_input: optimizer parameter {index} should be the {model}'s first-layer weight, of shape "
            f"{(hidden, n_inputs)}, but its Adam moments are {found}. The optimizer must hold the actor's parameters "
            f"and then the critic's, as rsl-rl's PPO does."
        )
    for moment in ("exp_avg", "exp_avg_sq"):
        state[moment] = torch.cat([state[moment], state[moment].new_zeros(hidden, extra_inputs)], dim=1)
    return checkpoint


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", help="a manipulation_g1_beaker_lift or earlier pour checkpoint (model_<N>.pt)")
    parser.add_argument("out", help="where to write the widened checkpoint (a new file)")
    args = parser.parse_args(argv)
    if os.path.abspath(args.out) == os.path.abspath(args.checkpoint):
        parser.error(f"out must be a new file, not the checkpoint itself ({args.checkpoint}).")
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location="cpu")

    import genesis as gs

    import eden as en

    en.init(backend=gs.gpu)
    from eden.envs.base import env_class_for_config
    from eden.tasks.manipulation.g1_beaker_pour.config import ManipulationG1BeakerPourConfig

    # The task's input sizes, as its observation manager builds them.
    config = ManipulationG1BeakerPourConfig()
    config.env_options.num_envs = 1
    env = env_class_for_config(config).from_config(config)
    env.build()
    sizes = dict(zip(_MODELS, (env.observation_manager.group_obs_dim[group][0] for group in ("policy", "critic"))))
    report = []
    for model, size in sizes.items():
        have = input_size(checkpoint, model)
        if have > size:
            raise ValueError(
                f"{args.checkpoint}: its {model} takes {have} inputs, more than the pour task's {size}; it is not a "
                f"checkpoint of the lift or of an earlier pour, which observe a prefix of the pour's inputs."
            )
        if have < size:
            widen_input(checkpoint, model, size - have)
        report.append(f"{model} inputs {have} -> {size}")
    torch.save(checkpoint, args.out)
    print(f"{args.out}: {', '.join(report)}.")


if __name__ == "__main__":
    main()
