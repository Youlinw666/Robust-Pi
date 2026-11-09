# Robust-Pi 正式实验 · 2026-11-06

单张 RTX 4090 24 GB 完成 π₀ 的 30,000 步训练，耗时约 2.622 小时，包含初始化、首次编译和保存。设备显存采样峰值为 18,547 MiB，包含 JAX 预分配。第 30,000 步的 Orbax 推理 checkpoint 独立加载通过，含参数和归一化资产，共 22 个文件、6,113,681,670 字节。恢复优化器状态需要另行保存相应资产。

实验沿用锁定的 OpenPI 和 LIBERO 版本。训练集包含任务 0–4 各 40 条示范，另有 25 条验证示范和 25 条留出示范。划分种子为 20261020，训练种子为 20261021。图像按 canonical 合同处理，归一化统计只从训练集计算。训练采用 batch 1、H50、32 维模型动作空间、两分支 LoRA rank 16/32 和原生 freeze filter。保留可训练的视觉和投影参数。具体设置见[公开结果 JSON](canonical_40shot_results.json)。

## 闭环结果

| LIBERO Spatial 任务 | Base | Fine-tuned |
|---|---:|---:|
| 0 | 0/50 | 42/50（84%） |
| 1 | 0/50 | 47/50（94%） |
| 2 | 0/50 | 18/50（36%） |
| 3 | 0/50 | 23/50（46%） |
| 4 | 0/50 | 29/50（58%） |
| 总计 | 0/250（0%） | **159/250（63.6%）** |

两种策略使用相同的官方初态、任务、控制条件和训练集归一化统计。环境种子 7，每回合 flow 种子重置为 0。动作块 H50，每次执行前 K5 个动作，flow 积分 10 步，环境稳定 10 步，执行上限 220 步。微调单独成功 159 对，Base 单独成功 0 对，两者均失败 91 对。结果覆盖五个任务，是对该实验协议的描述性比较。历史 500 段正式视频通过解码和帧数检查，收尾时重新核验了文件哈希。

G2 在最终 checkpoint 上补跑任务 0 的初态 0 和 1。Clean 与 Visual OOD 各成功 2/2。两对回合通过动作、视频和光度变换审计，结果仅描述这两个初态。早期 Pilot 结果见[历史实验记录](EXPERIMENTS.md)。

验证覆盖 25 条示范的 3,055 个起点，平均 native flow-matching loss 为 0.040626。该口径使用 BF16 推理 checkpoint、固定噪声及时间采样、32 维 padding 和 repeat-last 尾部目标。闭环成功率在上表单独计算。

## 代码与复现范围

公开更新包含 [canonical 图像合同](../data_tools/image_preprocessing.py)、[训练入口](../scripts/m3_formal_train.py)、[计划和资产校验](../training/data_binding.py)、[CPU 预检](../training/cpu_entry_preflight.py)与[评测客户端](../scripts/robustpi_eval_client.py)、[模型服务](../scripts/robustpi_eval_server.py)。评测入口改为从当前代码目录查找模块，客户端通过 `--inventory` 接收已审核的数据清单。

公开 CPU 检查可以按[复现指南](../docs/REPRODUCTION_GUIDE.md)运行。GPU 复现需要有权限的 LIBERO 数据、转换后的 Parquet、训练集统计、π₀ Base 参数及锁定上游环境。公开 JSON 是实验配置摘要，实际训练计划还需提供数据、权重、计划和来源文件的路径与哈希。

原预检对项目基础提交和两处上游提交进行校验。复核原流程时，在基础提交 `3d635222bca63459cb631c66524371f9e406adf8` 的独立检出中放入本次训练源码，保留原预检要求。OpenPI / LIBERO 版本见[上游锁](../upstream.lock.json)。准备真实资产后，使用唯一 run ID 和新输出目录，先生成与计划匹配的 CPU 预检，再执行获授权的训练。

```bash
JAX_PLATFORMS=cpu "$MODEL_PY" scripts/m3_formal_train.py --training-plan "$PLAN" --preflight --output "$PREFLIGHT"
JAX_PLATFORMS=cuda "$MODEL_PY" scripts/m3_formal_train.py --training-plan "$PLAN" --preflight-receipt "$PREFLIGHT" --allow-compute --approval-ref "$APPROVAL_REF"
```

闭环评测由模型环境启动服务，由 LIBERO 环境执行客户端。配置包含 `run_id`、`model_role`、`checkpoint`、`norm_stats` 和 `protocol_sources`。Base 指向 π₀ Base checkpoint 并传入同一训练集统计，微调角色从 checkpoint assets 恢复统计。

```bash
"$MODEL_PY" scripts/robustpi_eval_server.py --config "$EVAL_CONFIG" --port 8066
"$LIBERO_PY" scripts/robustpi_eval_client.py --config "$EVAL_CONFIG" --inventory "$INVENTORY" --output "$NEW_OUTPUT" --port 8066
```

GPU 记录来自原运行环境。公开发布的代码检查以 CPU 合成用例为范围，跨机器 GPU 复现、十任务评测、完整扰动评测和实体机器人实验仍待验证。
