# 代码导航与执行边界

公开候选保留真实项目源码的白名单子集，来源文件hash记录在[source_map.json](../provenance/source_map.json)。除Adapter模块说明更正外，已选历史源码逐字节复制；没有修复生产图像变换、改模型或重写历史结果。

| 路径 | 作用 | 公开检查范围 |
|---|---|---|
| [dataset_adapter.py](../data_tools/dataset_adapter.py) | episode内chunk、读Parquet、split/few-shot约束 | 合成chunk测试；读私有数据需要原资产 |
| [splits.py](../data_tools/splits.py) | 划分与manifest验证 | 源码保留，不重划分旧数据 |
| [openpi_bridge.py](../data_tools/openpi_bridge.py) | 官方变换桥接与train-only统计 | 需固定上游与私有资产，不在快速检查中调用 |
| [augmentation.py](../training/augmentation.py) | 可复现光度/语言增强原语 | 合成RGB、state/action保持测试 |
| [runner.py](../evaluation/runner.py) / [case_manifest.py](../evaluation/case_manifest.py) | case身份、逐回合记录与异常分类 | 纯CPU回调验证，不创建LIBERO环境 |
| [metrics.py](../analysis/metrics.py) | 有效分母、Wilson区间、描述性配对 | CPU指标与真实脱敏摘要校验 |
| [m3_formal_train.py](../scripts/m3_formal_train.py) | 受预算守门的E1/E2 pilot入口 | 历史代码审阅；未在G4执行 |
| [m3_gpu_smoke.py](../scripts/m3_gpu_smoke.py) | DeviceCheckpointWeightLoader等低host-memory实现 | 历史代码审阅；不属于CPU演示 |
| [libero_teaching.py](../scripts/libero_teaching.py) / [serve_e0.py](../scripts/serve_e0.py) | 仿真客户端与策略服务 | 需要私有配置/资产/上游；G4不运行 |
| [g2_e1_visual_ood.py](../scripts/g2_e1_visual_ood.py) | 串行资源守门、hash核对、逐输入审计 | 历史重型入口；不可用改名方式重复旧case |
| [g3_static_failure_audit.py](../scripts/g3_static_failure_audit.py) | 只读CPU失败分析 | 完整复核仍需要私有HDF5/Parquet/NPZ |
| [cpu_demo.py](../scripts/cpu_demo.py) / [公开CPU测试](../tests/test_public_cpu.py) | 不加载模型的可运行样例 | 无GPU、无仿真、无资产下载 |

源码中引用的历史资产相对路径只是文件合同，未将缺失私有文件做成Markdown链接。候选版并不包含它们，也不声称所有历史脚本可在纯CPU clone中运行。没有提供自动下载或自动启动GPU的quickstart。
