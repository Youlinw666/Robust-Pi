#!/usr/bin/env python3
"""Serve one pinned openpi checkpoint on loopback after GPU verification.

The default configuration remains the historical official E0 server.  A
project-local config may set ``checkpoint_role`` to ``self_trained`` to serve
an already validated inference checkpoint without treating it as official E0.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OPENPI_DATA_HOME", str(ROOT / "artifacts/openpi_cache"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/m1_e0.json")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    checkpoint_role = cfg.get("checkpoint_role", "official_e0")
    if checkpoint_role == "official_e0":
        receipt = json.loads((ROOT / "environment/e0_download_receipt.json").read_text())
        if receipt["status"] != "VERIFIED":
            raise RuntimeError("Checkpoint download must finish and verify first")
    elif checkpoint_role == "self_trained":
        checkpoint_dir = Path(cfg["checkpoint_dir"])
        if not (checkpoint_dir / "params").exists() or not (checkpoint_dir / "assets").exists():
            raise RuntimeError("Self-trained inference checkpoint must contain params and assets")
    else:
        raise ValueError(f"Unknown checkpoint_role: {checkpoint_role}")
    import jax
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import config

    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError("GPU required")
    model_config = config.get_config(cfg["config_name"])
    if model_config.model.action_horizon != cfg["action_horizon"]:
        raise ValueError("Config horizon differs from fixed checkpoint config")
    logging.info("Loading %s checkpoint on %s", checkpoint_role, devices)
    policy = policy_config.create_trained_policy(model_config, cfg["checkpoint_dir"])
    # Official Policy defaults to flow RNG seed 0; do not consume it before the episode.
    if cfg["flow_seed"] != 0:
        raise ValueError("This minimal E0 runner only supports official default flow seed 0")
    checkpoint_label = cfg.get("checkpoint_label", cfg.get("checkpoint_dir"))
    server = WebsocketPolicyServer(policy=policy, host="127.0.0.1", port=args.port,
        metadata={**policy.metadata, "robust_pi_run_id": cfg["run_id"],
                  "checkpoint": checkpoint_label, "checkpoint_role": checkpoint_role})
    logging.info("Checkpoint loaded; serving only 127.0.0.1:%d", args.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
