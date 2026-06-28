# DSpark-OPD 调研与工程笔记

## 1. DSpark / DeepSpec

DeepSpec 是 DeepSeek 开源的 speculative decoding 训练与评测仓库，覆盖 Eagle3、DFlash 和 DSpark。DSpark 的论文把问题拆成两层：

- 草稿模型质量：纯并行 drafter 一次生成多个 token，但块内 token 之间缺少依赖，后缀 acceptance 很快下降。
- 系统吞吐：长草稿块并不总是好，高并发下验证低置信 token 会占掉 target model batch capacity。

DSpark 的对应解法是：

- 半自回归生成：并行 backbone 先给每个位置出 base logits，再用 Markov/RNN 轻量顺序 head 注入块内依赖。
- 置信度调度验证：confidence head 预测条件 acceptance，prefix scheduler 根据累计 survival probability 和 SPS(B) 选择每个请求要验证的 prefix 长度。
- 训练目标：CE + TV/L1 + confidence BCE，默认权重约为 `0.1 / 0.9 / 1.0`。

DeepSpec 源码中，`deepspec/modeling/dspark/loss.py` 已经实现了这三个监督项；`target_cache_dataset.py` 则把 target hidden states 和 last hidden states 缓存在二进制 shard 中。

## 2. OPD

OPD 的核心不是“换一个 teacher”，而是“换训练状态分布”：

- 离线 KD/SFT：学生在 teacher 或数据集轨迹上学习。
- OPD：学生先用当前策略 rollout，teacher 再在学生实际访问的 state 上给 token-level 分布。

Thinking Machines 的实现走轻量路线：学生 rollout 后，让 teacher 计算同一序列 logprob，用 `log p_student - log p_teacher` 形成 reverse-KL 估计，再把负 KL 当 advantage。verl 把这件事工程化成两类：

- PG OPD：单样本 KL estimator，如 k1/k3，作为 policy-gradient advantage。
- GKD OPD：teacher top-k forward KL，直接反传。

## 3. DeepSeek-V4 里的 OPD

DeepSeek-V4 技术报告说明，它们用多教师 OPD 合并数学、代码、agent、指令等专家模型能力。关键点：

- 统一模型作为 student，在自己的 rollout 上学习多个专家 teacher。
- 目标是 reverse KL。
- 它们认为 token-level KL 估计虽然省资源，但方差高、训练不稳，所以采用 full-vocabulary logit distillation。
- 工程上不直接存全词表 logits，而是缓存 teacher last hidden states，训练时再过对应 LM head 还原 logits。

这与 DSpark/DeepSpec 的 target hidden-state cache 很契合：我们可以把 replay 位置的 teacher hidden states 当成 OPD teacher cache，再局部计算 full KL。

## 4. Draft-OPD 的启发

Draft-OPD 专门讨论 speculative draft model 的 OPD：

- draft-only rollout 容易退化，因为 DFlash/EAGLE 类 drafter 本来不是完整自回归模型。
- target-assisted rollout 保证序列质量，但会把被拒 token 丢掉，破坏 on-policy 信号。
- 解决方案是记录 speculative verification 暴露的 error anchors，再从这些 anchors replay draft blocks。
- 对 accepted tokens 用 forward KL，对 rejected tokens 用 reverse KL，并对后缀 rejected tokens 衰减。

DSpark-OPD 采用这个思想，但结合 DSpark 的 confidence/scheduler：

- 训练仍保留 DSpark 原始 CE/TV/confidence。
- Replay block 用 OPD 给额外监督。
- Accepted/rejected mask 由 verification outcome 产生。
- Full-vocab KL 优先；teacher top-k 或单样本 KL 作为资源受限 fallback。

## 5. 本项目实现边界

本仓库实现的是可测试、可迁移的核心层：

- `losses.py`：numpy 参考实现，便于验证数学。
- `torch_losses.py`：DeepSpec trainer 可调用的 PyTorch 适配层。
- `scheduler.py`：论文 Algorithm 1 的 prefix scheduler。
- `calibration.py`：Sequential Temperature Scaling。
- `replay.py`：Draft-OPD 风格 replay mask。

没有在本仓库直接启动大模型训练，因为公开 DSpark 训练需要多卡、目标模型 cache 和几十 TB 级别数据产物。这里的工程重点是把 OPD 与 DSpark 结合的接口切清楚，让它能接到 DeepSpec 或 verl 的训练系统中。

## 6. 推荐落地路径

1. 先用 DeepSpec 原流程训练/加载 DSpark。
2. 用 current DSpark 做 speculative rollout，记录 anchors、accepted_count、draft_token_ids。
3. Replay anchors，收集 student logits 和 teacher logits/top-k logprobs。
4. 调用 `compute_dspark_opd_torch_loss`，把 supervised DSpark loss 与 OPD loss 相加。
5. 对 confidence head 做 STS，部署时用 scheduler 动态裁剪验证长度。

## 7. 资料来源

完整链接见 `docs/sources.md`。本地 `asset` 目录中已保存 DeepSpec、tinker-cookbook、Draft-OPD、verl 源码，以及 DSpark / DeepSeek-V4 / Draft-OPD 三份 PDF。
