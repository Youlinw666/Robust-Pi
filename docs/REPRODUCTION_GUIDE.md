# 公开版复现范围

## 公开CPU检查

Python3.11及已具备的`requirements-cpu.txt`依赖下，在本候选根目录运行：

```bash
python3 scripts/cpu_demo.py
python3 -m pytest -q -p no:cacheprovider tests/test_public_cpu.py
python3 tools/audit_release.py
```

本阶段使用已有隔离Python执行，没有安装依赖、下载权重/数据或创建仿真。CPU demo是明确标记的合成数据协议检查，不是训练、推理或机器人新回合；输出不能替代[真实实验摘要](../reports/EXPERIMENTS.md)。依赖文件只覆盖这一CPU路径，不是完整GPU环境锁。

## 授权本地环境的完整证据复核

拥有原私有资产的人，可按固定源码revision和原文件SHA256恢复相对目录结构：`external/openpi`、模型venv、LIBERO venV、数据目录、checkpoint assets、原manifest/NPZ/日志。外部源码与资产均未放进候选包；G4没有执行恢复或重下载。

G3入口`g3_static_failure_audit.py --output-dir <全新审计目录>`只做CPU读取；需要既有formal inventory/splits、HDF5/Parquet及四条rollout的NPZ/events。已有输出会拒绝覆盖。历史GPU入口只用于阅读实现，未经新的执行范围确认，不应运行或绕过输出保护。

私有证据对应的摘要及内容指纹见[public_results.json](../reports/public_results.json)、[public_failure_summary.json](../reports/public_failure_summary.json)、[private_evidence_fingerprints.json](../provenance/private_evidence_fingerprints.json)。指纹不是数字签名，也不代替原始证据；无法从摘要重新计算完整像素审计。

## 不包含或未验证的内容

- 不包含模型、原始/转换数据、权重缓存、原始状态/动作数组、视频、截图、既有图表和私密日志。
- 没有第二台机器干净GPU安装验收，没有修复后的任务成功率，也不承诺纯CPU clone可复现GPU结果。
- 不能从旧E1推理checkpoint恢复optimizer状态。
- 没有改变旧划分、stats、checkpoint或G2结果。已知图像合同问题只形成[修复设计](IMAGE_PREPROCESSING_PROPOSAL.md)。
