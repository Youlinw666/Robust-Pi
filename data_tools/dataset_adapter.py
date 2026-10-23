"""Read pinned PI LIBERO expert Parquet episodes, without a model or network.

Image orientation depends on the source. The historical raw-HDF5 formal
Parquet path is unrotated, unlike the rollout input; see Known Limitations.
This reader performs no flip, resampling, scaling, normalization or delta transform.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq


class DataError(ValueError):
    """An explicit unusable-data reason; callers must retain it in audit output."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def future_actions(actions: np.ndarray, time_index: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """LeRobot-compatible repeat-last padding; True mask means REAL action.

    This accepts exactly one episode. It can never obtain a future action from
    another episode. The mask is diagnostic; official openpi currently drops it.
    """
    if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions):
        raise DataError("Expected one nonempty episode actions[T,7]")
    if not 0 <= time_index < len(actions) or horizon < 1:
        raise IndexError("time_index outside episode or nonpositive horizon")
    indices = time_index + np.arange(horizon)
    valid = indices < len(actions)
    return actions[np.minimum(indices, len(actions) - 1)].copy(), valid


@dataclass
class Episode:
    task_id: int
    dataset_task_index: int
    episode_id: int
    prompt: str
    fps: int
    table: pa.Table
    state: np.ndarray
    actions: np.ndarray
    timestamp: np.ndarray
    path: Path

    def __len__(self):
        return len(self.actions)

    def image(self, time_index: int, camera: str) -> np.ndarray:
        if camera not in ("image", "wrist_image") or not 0 <= time_index < len(self):
            raise IndexError("Invalid camera or time_index")
        record = self.table[camera][time_index].as_py()
        if not isinstance(record, dict) or not record.get("bytes"):
            raise DataError("{} frame {} has no embedded image bytes".format(camera, time_index))
        try:
            with Image.open(io.BytesIO(record["bytes"])) as image:
                array = np.array(image)
        except Exception as exc:
            raise DataError("Cannot decode {} frame {}: {}".format(camera, time_index, exc)) from exc
        if (array.ndim != 3 or array.shape[-1] != 3 or array.shape[0] != array.shape[1]
                or array.shape[0] not in (128, 256) or array.dtype != np.uint8):
            raise DataError("Bad image shape/dtype: {} {}".format(array.shape, array.dtype))
        return array

    def sample(self, time_index: int, horizon: int = 10) -> dict:
        actions, valid = future_actions(self.actions, time_index, horizon)
        return dict(image=self.image(time_index, "image"), wrist_image=self.image(time_index, "wrist_image"),
            state=self.state[time_index].copy(), actions=actions, prompt=self.prompt,
            actions_is_pad=~valid, action_valid_mask=valid,
            episode_index=np.int64(self.episode_id), task_id=np.int64(self.task_id),
            task_index=np.int64(self.dataset_task_index), frame_index=np.int64(time_index),
            timestamp=self.timestamp[time_index])


class DatasetAdapter:
    """Task/episode reader plus an optional immutable split/few-shot filter.

    Membership comes from source inventory, never filenames or episode order.
    Supplying a manifest makes load_episode enforce the requested split/pool.
    """
    def __init__(self, data_root: str | Path, inventory: str | Path, *,
                 manifest: str | Path | None = None, split: str | None = None, shots: int | None = None):
        self.root = Path(data_root)
        self.inventory_path = Path(inventory)
        self.inventory = json.loads(self.inventory_path.read_text())
        self.tasks = {task["task_id"]: task for task in self.inventory["spatial_tasks"]}
        self.files = {item["episode_id"]: item for item in self.inventory["selected_files"]}
        if len(self.files) != len(self.inventory["selected_files"]):
            raise DataError("Duplicate source episode IDs")
        self.allowed_ids = set(self.files)
        self.split, self.shots = split, shots
        self.manifest_path = Path(manifest) if manifest else None
        if manifest:
            from data_tools.splits import validate_manifest
            document = json.loads(Path(manifest).read_text())
            validate_manifest(document)
            if document.get("status") != "AUDITED_FINAL":
                raise DataError("A plan is not an audited final dataset split")
            if document["revision"] != self.inventory["revision"] or document["repo_id"] != self.inventory["repo_id"]:
                raise DataError("Manifest/source revision mismatch")
            if document["source_inventory_sha256"] != sha256_file(self.inventory_path):
                raise DataError("Manifest source inventory hash mismatch")
            if split not in ("train", "val", "holdout"):
                raise DataError("A manifest requires an explicit train/val/holdout split")
            if shots is not None and split != "train":
                raise DataError("Few-shot selection is allowed only in train")
            key = "fewshot_episode_ids" if shots is not None else split + "_episode_ids"
            selected = document[key][str(shots)] if shots is not None else document[key]
            self.allowed_ids = {eid for ids in selected.values() for eid in ids}
            if not self.allowed_ids <= self.files.keys():
                raise DataError("Manifest includes episodes outside the source inventory")
            if any(self.files[eid]["task_id"] != int(task) for task, ids in selected.items() for eid in ids):
                raise DataError("Manifest episode/task mapping differs from the source")
        elif split is not None or shots is not None:
            raise DataError("Filtering requires a persisted manifest")

    def episode_ids(self, task_id: int | None = None) -> list[int]:
        if task_id is not None and task_id not in self.tasks:
            raise KeyError("Unknown LIBERO Spatial task_id")
        return sorted(eid for eid in self.allowed_ids if task_id is None or self.files[eid]["task_id"] == task_id)

    def load_episode(self, task_id: int, episode_id: int, *, verify_hash: bool = True) -> Episode:
        if episode_id not in self.allowed_ids:
            raise DataError("Episode {} is outside selected manifest pool".format(episode_id))
        item = self.files[episode_id]
        if item["task_id"] != task_id:
            raise DataError("Episode does not belong to requested task_id")
        path = self.root / item["path"]
        if not path.is_file():
            raise DataError("Missing episode file: " + str(path))
        if path.stat().st_size != item["size"]:
            raise DataError("Episode byte size mismatch")
        if verify_hash and item.get("sha256") and sha256_file(path) != item["sha256"]:
            raise DataError("Episode SHA256 mismatch")
        try:
            table = pq.read_table(path)
        except Exception as exc:
            raise DataError("Unreadable Parquet: " + str(exc)) from exc
        required = {"image", "wrist_image", "state", "actions", "timestamp", "frame_index", "episode_index", "index", "task_index"}
        if not required <= set(table.column_names):
            raise DataError("Missing fields: " + str(sorted(required - set(table.column_names))))
        n = table.num_rows
        if not n or n != item["length"]:
            raise DataError("Empty episode or length differs from source metadata")
        for name, dim in (("state", 8), ("actions", 7)):
            typ = table.schema.field(name).type
            if not pa.types.is_fixed_size_list(typ) or typ.list_size != dim or typ.value_type != pa.float32():
                raise DataError("Unexpected source {} Arrow type {}".format(name, typ))
        for name, expected_type in (("timestamp", pa.float32()), ("frame_index", pa.int64()),
                                   ("episode_index", pa.int64()), ("index", pa.int64()), ("task_index", pa.int64())):
            if table.schema.field(name).type != expected_type or table[name].null_count:
                raise DataError("Unexpected scalar type or nulls in " + name)
        numeric = {}
        for name in ("state", "actions", "timestamp", "frame_index", "episode_index", "index", "task_index"):
            try:
                array = np.asarray(table[name].to_pylist(), dtype=np.float32 if name in ("state", "actions", "timestamp") else np.int64)
            except (ValueError, TypeError) as exc:
                raise DataError("Invalid numeric values in " + name) from exc
            if not np.isfinite(array).all() or table[name].null_count:
                raise DataError("NaN/Inf/null in " + name)
            numeric[name] = array
        if numeric["state"].shape != (n, 8) or numeric["actions"].shape != (n, 7):
            raise DataError("Invalid state/action array dimensions")
        task = self.tasks[task_id]
        if not np.all(numeric["episode_index"] == episode_id) or not np.all(numeric["task_index"] == task["dataset_task_index"]):
            raise DataError("Episode/task identity mismatch inside file")
        if not np.array_equal(numeric["frame_index"], np.arange(n)) or not np.all(np.diff(numeric["index"]) == 1):
            raise DataError("Noncontiguous frame/global indices")
        if not np.allclose(numeric["timestamp"], np.arange(n) / self.inventory["source_fps"], rtol=0, atol=1e-4):
            raise DataError("Timestamps do not match nominal dataset fps")
        return Episode(task_id, task["dataset_task_index"], episode_id, task["prompt"], self.inventory["source_fps"],
            table, numeric["state"], numeric["actions"], numeric["timestamp"], path)

    def episodes(self) -> Iterator[Episode]:
        for eid in self.episode_ids():
            yield self.load_episode(self.files[eid]["task_id"], eid)


class ActionChunkDataset:
    """Map-style NumPy dataset accepted by openpi TransformedDataset/TorchDataLoader.

    It caches one episode only. Returned arrays are copied before official
    transforms (notably legacy DeltaActions) can mutate them.
    """
    def __init__(self, adapter: DatasetAdapter, horizon: int = 10):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.adapter, self.horizon = adapter, horizon
        self.ids = adapter.episode_ids()
        self.ends = np.cumsum([adapter.files[eid]["length"] for eid in self.ids]).tolist()
        self._episode = None

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index: int) -> dict:
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError("Sample index outside selected episodes")
        position = bisect_right(self.ends, index)
        eid = self.ids[position]
        if self._episode is None or self._episode.episode_id != eid:
            self._episode = self.adapter.load_episode(self.adapter.files[eid]["task_id"], eid)
        start = self.ends[position - 1] if position else 0
        return self._episode.sample(index - start, self.horizon)
