# G3失败诊断的公开摘要

## 已确认的工程缺陷

训练/rollout图像方向相差180°。HDF5 creator保存raw相机数组，formal转换将这些数组原样编码PNG；Adapter只解码，训练变换没有180°旋转。rollout的`policy_input`则执行`[::-1,::-1]`再resize。真实formal demo0双视角与raw HDF5逐像素一致，且不等于180°旋转后的数组；静态审计证明路径差异，不是依据0成功率猜测。

本候选不分发真实画面。直接审阅[转换源码](../scripts/m2_convert_original_offset.py)、[Adapter](../data_tools/dataset_adapter.py)、[rollout客户端](../scripts/libero_teaching.py)与[G3 CPU审计入口](../scripts/g3_static_failure_audit.py)；固定上游变换来源见[锁定版本](../upstream.lock.json)。已执行审计的选定数值与私有证据hash分别在[公开摘要](public_failure_summary.json)和[内容指纹](../provenance/private_evidence_fingerprints.json)中。没有原私有资产时不能从摘要重新完成像素证明。

## 未被证明的动作错误

真实H50动作块经legacy DeltaActions、mean/std、32维padding、反归一化、AbsoluteActions与截取7维，float32边界往返最大误差约1.305e-7；200条训练示范统计重算吻合。检查不支持把失败直接归为维度或逆归一化错误，但不穷尽所有可能bug。

LIBERO七维action前六维是归一化位置/axis-angle增量，最后是夹爪命令。OSC clip[-1,1]后分别缩放至每分量±0.05m/±0.5rad。大幅旋转命令和状态偏离是确认现象，不能单独确诊训练不足或归一化错误。

## 轨迹、训练与证据边界

两条Clean初始EEF距目标约0.379m，最近仍约0.292/0.285m，最终约1.618/1.441m，支持短暂接近后明显偏离。目标在离散快照中几乎未移动；快照距离不是接触判定，也不能证明所有时刻无接触。

100步batch1仍处默认1000步warmup，没有充分收敛或held-out任务能力证据。相同task的示范支持轨迹形态比较，当前初始化hash不匹配，不把它当逐帧GT。

LoRA描述限于实际证据：17冻结base叶一致、33个可训练非LoRA base叶改变、20个LoRA叶非零。因A/B初始化本来非零，不能据此宣称逐叶adapter更新幅度。

**缺陷存在的置信度高；其对0成功率的独立因果贡献尚未隔离。** 没有修复后GPU对照；[CPU修复设计](../docs/IMAGE_PREPROCESSING_PROPOSAL.md)未实施。旧E1、stats、划分和G2结果保持原样，不能把未来结果补写回本次实验。
