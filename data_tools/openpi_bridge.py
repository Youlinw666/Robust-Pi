"""CPU-only use of the pinned official data transforms; no model parameters.

Model configs are instantiated as metadata, never model.create/load. π0's
native LIBERO H=50 is recorded explicitly when building M2 H=10 samples.
"""
from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import numpy as np

from data_tools.dataset_adapter import ActionChunkDataset, DatasetAdapter, DataError, sha256_file


def official_data_config(model_name: str, horizon: int = 10):
    from openpi.training import config
    name = {"pi0": "pi0_libero_low_mem_finetune", "pi05": "pi05_libero"}[model_name]
    native = config.get_config(name)
    # This creates only config/transform objects, not a neural network.
    model = dataclasses.replace(native.model, action_horizon=horizon)
    data = native.data.create(native.assets_dirs, model)
    return model, data, dict(official_config=name, native_action_horizon=native.model.action_horizon,
        m2_sample_action_horizon=horizon, action_dim=model.action_dim, token_length=model.max_token_len,
        use_quantile_norm=data.use_quantile_norm, extra_delta_transform=native.data.extra_delta_transform,
        discrete_state_input=model.discrete_state_input, weights_loaded=False, training_started=False)


def before_normalization(sample: dict, data_config) -> dict:
    from openpi import transforms
    # DeltaActions modifies its action array. Work on a private copy.
    data = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in sample.items()}
    return transforms.compose([*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs])(data)


def transform_for_model(sample: dict, model_name: str, norm_stats: dict, horizon: int = 10) -> tuple[dict, dict]:
    from openpi import transforms
    _, data, description = official_data_config(model_name, horizon)
    value = before_normalization(sample, data)
    value = transforms.Normalize(norm_stats, use_quantiles=data.use_quantile_norm)(value)
    return transforms.compose(data.model_transforms.inputs)(value), description


def compute_pool_stats(adapter: DatasetAdapter, model_name: str, *, horizon=10) -> tuple[dict, dict]:
    """Only a finalized training pool; no validation/holdout read is possible.

    Accumulate the actual H-step, repeat-last-padded action targets AFTER native
    data transforms and BEFORE normalization/model padding. This matches the
    sampling weights of official compute_norm_stats, including tail repeats.
    Unlike its batch floor, every selected start frame is included here.
    """
    import json
    from openpi.shared.normalize import RunningStats
    if adapter.split != "train" or adapter.shots is None or adapter.manifest_path is None:
        raise DataError("Stats require a manifest-filtered TRAIN few-shot adapter")
    manifest = json.loads(adapter.manifest_path.read_text())
    if manifest["status"] != "AUDITED_FINAL":
        raise DataError("Stats require a fully audited final manifest")
    _, data, description = official_data_config(model_name, horizon)
    running = {key: RunningStats() for key in ("state", "actions")}
    loaded = []
    source_files = []
    starts, padded = 0, 0
    # Images are not needed by LiberoInputs/DeltaActions for numeric statistics.
    # Construct only the numeric projection, explicitly using native delta code.
    from openpi import transforms
    for episode in adapter.episodes():
        loaded.append(episode.episode_id)
        source_files.append(dict(episode_id=episode.episode_id, sha256=sha256_file(episode.path)))
        values = []
        for t in range(len(episode)):
            from data_tools.dataset_adapter import future_actions
            actions, valid = future_actions(episode.actions, t, horizon)
            numeric = dict(state=episode.state[t].copy(), actions=actions)
            if description["extra_delta_transform"]:
                numeric = transforms.DeltaActions(transforms.make_bool_mask(6, -1))(numeric)
            values.append(numeric["actions"])
            padded += int((~valid).sum())
        running["state"].update(episode.state.astype(np.float64))
        running["actions"].update(np.asarray(values, dtype=np.float64))
        starts += len(episode)
    if set(loaded) != adapter.allowed_ids:
        raise DataError("Statistic source IDs differ from requested training pool")
    stats = {key: value.get_statistics() for key, value in running.items()}
    for value in stats.values():
        for field in ("mean", "std", "q01", "q99"):
            if not np.isfinite(getattr(value, field)).all():
                raise DataError("Nonfinite statistics")
    audit = dict(**description, shots_per_task=adapter.shots, manifest_sha256=sha256_file(adapter.manifest_path),
        source_inventory_sha256=sha256_file(adapter.inventory_path), source_files=source_files,
        source_episode_ids=loaded, only_training_pool=True, start_frames=starts,
        action_vectors_including_tail_repeats=starts * horizon, padded_action_vectors=padded,
        histogram_quantile_bins=5000, all_start_frames_included=True,
        zero_std_dimensions={key: np.flatnonzero(value.std == 0).tolist() for key, value in stats.items()},
        quantile_zero_width_dimensions={key: np.flatnonzero(value.q99 == value.q01).tolist() for key, value in stats.items()})
    from openpi.shared import normalize
    numeric_config = dict(**description, transforms_source_sha256=sha256_file(Path(transforms.__file__)),
        normalize_source_sha256=sha256_file(Path(normalize.__file__)),
        bridge_source_sha256=sha256_file(Path(__file__)), accumulation_dtype="float64",
        sample_weighting="all starting frames, H actions each, repeat-last tails included")
    audit["numeric_config"] = numeric_config
    audit["numeric_config_sha256"] = hashlib.sha256(json.dumps(numeric_config, sort_keys=True).encode()).hexdigest()
    return stats, audit
