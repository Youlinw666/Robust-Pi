# Third-party notices

自有项目材料由用户明确确认采用MIT，Copyright (c) 2026 youlinw666，见[LICENSE](../LICENSE)。此许可不覆盖第三方材料的独立权利；第三方部分继续遵循其适用条款。

| 来源 | 固定版本与用途 | 保留文本 |
|---|---|---|
| [Physical Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | `215abfb217dbac7d5f1273282331b9b1866c0479`，π₀架构、训练及policy API；本候选不vendor整个模型源码 | [Apache-2.0](licenses/openpi-LICENSE.txt) |
| [Lifelong Robot Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`，任务/仿真接口；本候选不分发仿真资产 | [MIT](licenses/LIBERO-LICENSE.txt) |
| NumPy/PyArrow/Pillow/pytest | CPU依赖，只列版本，不vendor运行时 | 遵循各自发行版许可 |
| 模型权重、原始数据、纹理/视频/截图 | 不在候选包；仓库代码许可不自动证明其再分发许可 | [媒体与资产范围](MEDIA_POLICY.md) |

项目脚本参考/调用固定上游接口；保留来源说明，不声称自研基础模型或复现论文全部效果。若未来导入上游代码、权重或媒体，应重新核对具体材料及NOTICE义务，不能只凭根目录MIT文件覆盖所有内容。
