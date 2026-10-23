# 实际模型结构与训练/推理流程

依据固定openpi `215abfb217dbac7d5f1273282331b9b1866c0479` 与LIBERO `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`。对应上游文件为`models/pi0.py`、`pi0_config.py`、`gemma.py`、`training/config.py`和`policies/libero_policy.py`。候选版不复制整个上游checkout，固定来源见[upstream.lock.json](../upstream.lock.json)。

## 模型

```mermaid
flowchart LR
    RGB["base + wrist RGB / 224×224"] --> V["SigLIP So400m/14 / width2048"]
    TXT["instruction / 最多48 token"] --> L["Gemma2B LoRA rank16"]
    V --> P["embed_prefix / image + language"]
    L --> P
    P --> KV["prefix attention KV cache"]
    ST["state 8→32 / state_proj"] --> SUF["embed_suffix"]
    NOISE["噪声 B×50×32 / flow time"] --> M["action_in_proj + time MLP"]
    M --> SUF
    SUF --> AE["Gemma300M / width1024 / LoRA rank32"]
    KV --> AE
    AE --> VF["action_out_proj / 速度场"]
    VF --> ODE["默认10步Euler / t1→0"]
    ODE --> ACT["逆归一化 + legacy逆变换 / 50×7"]
```

PaliGemma前缀与动作expert是同一多分支Gemma模块中的不同权重分支，不是外接自然语言planner。π₀使用连续state token和时间MLP，不套用π₀.₅的adaRMS分支。缺失right wrist补零并mask=false。两个分支LoRA alpha/r均为1；冻结规则不涵盖所有视觉/投影参数。

Flow Matching：`x_t=t*epsilon+(1-t)*a`，训练目标`epsilon-a`，MSE拟合速度场；源码约定`t=1`为噪声、`t=0`为动作。推理负步长Euler，动作连续生成。H50、K5和flow10的含义互不替代。

## 历史训练流程

```mermaid
flowchart TD
    H["固定HDF5 / 五任务"] --> D["post-action观测配下一动作 / Parquet"]
    D --> B["已知缺陷：图像raw方向保留"]
    D --> S["40-shot各任务 / 200条示范"]
    S --> C["H50 / episode内repeat-last"]
    C --> T["legacy DeltaActions / train-only mean-std"]
    T --> P["pad32 / 图像resize / tokenize"]
    P --> F["Flow Matching / batch1 / 100步"]
    F --> K["Orbax params+assets"]
    K --> V["结构与fresh reload / 叶子审计"]
```

E1“clean”指未启用项目E3/E4/E5增强训练，上游train=True仍有crop、±5°旋转和颜色增强。数据时间合同为`raw_obs[t]→action[t+1]`。mean/std在legacy变换后计算，状态/动作再pad到32。尾部repeat-last没有被loss mask排除，不能将全部loss当作真实动作维度或任务成功率。

## 历史推理与闭环流程

```mermaid
flowchart TD
    REG["固定manifest / checkpoint / stats hash"] --> PRE["RAM可用15GiB / GPU free8GiB / 单模型"]
    PRE --> OBS["LIBERO固定task-init / settle10"]
    OBS --> IMG["raw RGB旋转180°并resize224"]
    IMG --> IN["双RGB + state8 + instruction"]
    IN --> CHUNK["H50动作块 / 取前K5"]
    CHUNK --> CTRL["OSC clip与scale / 7维动作"]
    CTRL --> NEXT["新观察 / 重规划"]
    NEXT --> IN
    CTRL --> SAVE["events / 视频 / 身份与资源证据"]
    SAVE --> GATE["第一条系统与资源验收通过才运行第二条"]
```

推理端`[::-1,::-1]`与训练端保留raw方向形成G3确认的差异。本图描述实际旧实现，并非已修复管线。动作输出已反归一化，客户端不做第二次反归一化；控制器对位置/旋转分量裁剪后分别缩放至每分量±0.05m、±0.5rad。20Hz仿真控制频率不等于真实机器人20Hz实时系统。

模型源图、训练源图、评测源图另保留在[diagrams目录](diagrams/model.mmd)。代码入口与资产边界见[代码导航](CODE_GUIDE.md)。
