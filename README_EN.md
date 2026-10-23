# Robust-Pi: low-resource π₀ adaptation and auditable failure analysis

A VLA internship engineering project built on pinned **openpi π₀ and LIBERO**, under an **RTX 5070 12 GB VRAM / WSL2 20 GB RAM** budget. It covers data contracts, a LoRA pilot, low-memory checkpoint loading, bounded closed-loop evaluation, and static failure diagnosis.

**Task results: Clean 0/2 and Visual OOD 0/2. All four episodes timed out.** The pipeline ran successfully; task capability and robustness gains have not been established. This is not a complete LIBERO benchmark or a full paper reproduction.

[中文](README.md) · [MIT](LICENSE) · [Architecture](docs/PROJECT_ARCHITECTURE.md) · [Code guide](docs/CODE_GUIDE.md)

## Verified engineering results

- Five tasks, 40 demonstrations **per task**, 200 total; episode-level 40/5/5 splits, nested pools and train-only statistics. Prior replay passed 236/250 structurally valid demonstrations, with limitations retained.
- E1: batch size 1, 100/100 optimizer updates and 100 finite loss records. The pilot remained inside the default 1,000-step warmup; it does not establish convergence.
- Orbax params + normalization assets, structural checks and fresh reload. No optimizer-resume checkpoint. The configuration also updates trainable vision/projection parameters; it is not adapter-only training.
- GPU action chunks of shape `(50,7)`, H=50, K=5, default 10 flow integration steps. G2 used two reused Clean episodes and two new OOD episodes, each 220 actions / 44 inferences / 221 video frames, without system errors.
- G2 maximum sampled total GPU use: 9,004 MiB. Minimum available RAM: 6.44 GiB; minimum GPU free memory: 2,940 MiB. Sampling may miss transients.

Official E0 results are separate from self-trained E1. E2 initialization failed before any optimizer update; no E2 checkpoint exists. E3/E4/E5 training was not run.

## Known Limitations

G3 confirmed a **180-degree orientation mismatch between formal training images and rollout inputs**, using source inspection and real HDF5/Parquet pixel checks. A CPU-only repair design is documented but has not been applied to the historical pipeline. There is **no repaired-model task-success result**, and the mismatch has not been isolated as the sole cause of failure.

Clean success is zero, so OOD retention is undefined, not zero. Two pairs are descriptive only. Twenty nonzero LoRA leaves do not establish an initialization-to-checkpoint update because both factors are randomly initialized in the pinned implementation; the numerical evidence separately verifies 17 frozen base leaves and 33 changed trainable non-LoRA base leaves.

## Review and CPU quick check

See [architecture and workflows](docs/PROJECT_ARCHITECTURE.md), [experiment summary](reports/EXPERIMENTS.md), [failure analysis](reports/FAILURE_ANALYSIS.md), and [reproduction guide](docs/REPRODUCTION_GUIDE.md).

With Python 3.11 and the existing [CPU dependencies](requirements-cpu.txt):

```bash
python3 scripts/cpu_demo.py
python3 -m pytest -q -p no:cacheprovider tests/test_public_cpu.py
python3 tools/audit_release.py
```

The synthetic demo is not a robot evaluation. Historical training/rollout entry points are included for source review and require private assets plus pinned upstream environments; they were not executed or validated by the public quick check. No assets were downloaded during G4.

## Distribution boundary

This candidate excludes the original Git history, weights, datasets, caches, private logs, videos, screenshots and existing charts. Redistribution permission for those media has not been established; only [media descriptions and authorized local reproduction paths](docs/MEDIA_POLICY.md) are provided. Mermaid diagrams contain implementation descriptions rather than dataset images.

Original project material: **MIT, Copyright (c) 2026 youlinw666**, as explicitly confirmed by the owner. Upstream material retains its applicable terms; see [third-party notices](docs/THIRD_PARTY_NOTICES.md). Review [release audit](reports/G4_PUBLIC_RELEASE_AUDIT.md) and [checklist](docs/GITHUB_RELEASE_CHECKLIST.md) before any manual publication. No commit, push, or public upload was performed.
