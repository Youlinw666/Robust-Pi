# E1实际结果：两对小样本诊断

| task/init | 条件 | 来源 | 结果 | 步数 | 推理 | 视频帧（媒体未分发） |
|---|---|---|---|---:|---:|---:|
| 0/0 | Clean | reused | timeout | 220 | 44 | 221 |
| 0/0 | Visual OOD | new | timeout | 220 | 44 | 221 |
| 0/1 | Clean | reused | timeout | 220 | 44 | 221 |
| 0/1 | Visual OOD | new | timeout | 220 | 44 | 221 |

**Clean0/2、Visual OOD0/2；无system error；retention为undefined/null。** 只有两对，不能外推总体成功率或显著性。G2评测链路验收通过不等于策略质量通过。

E1训练为五任务各40-shot、batch1、seed20261021、100/100更新。eval为H50/K5、max_steps220、settle10，环境/NumPy/Python seed7、flow seed0、默认10步采样。两个条件的checkpoint/stats、初始raw观察/state/instruction、控制器保持对齐；视觉OOD触发概率0.5，brightness[0.8,1.2]、contrast[0.9,1.1]、noise std2.0。旧E0概率1.0及seed规则不同，不能合并为同一协议。

GPU总占用峰值含桌面等其他使用，两个OOD分别9004/8966MiB，最低RAM可用约6.44/7.56GiB。旧Clean没有同时资源流，相关字段留null，不借用G1/OOD数据代填。资源采样可能漏瞬时峰值。

数据接纳限制、100步pilot、小样本、图像方向缺陷及无修复后实验均保留。数值及身份摘要见[public_results.json](public_results.json)，失败诊断见[FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md)。不含原始状态/动作数组、任务画面或权重。
