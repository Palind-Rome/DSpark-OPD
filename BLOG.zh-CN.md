# 把 OPD 迁移到 DSpark：一次研究型工程实践

## 1. 什么是 OPD？

一言以蔽之：OPD 让学生模型在“自己会走到的地方”向老师学习。

普通蒸馏常在老师生成的轨迹上训练学生。问题是，推理时学生不一定会走老师那条路。一旦学生走偏，训练时没见过这些状态，老师的知识也帮不上忙。OPD 的做法是：先让学生按当前策略生成，再让老师在这些学生状态上给分布监督。

这正好解决 exposure bias：训练状态和推理状态对齐了。

## 2. 什么是 DSpark？

DSpark 是 DeepSeek 的 speculative decoding drafter。它想让大模型生成得更快，但不改变目标模型的输出分布。

它有两个关键设计：

1. 半自回归草稿生成：并行 backbone 一次预测多个位置，再用很轻的 Markov/RNN head 让后续 token 看见前面已经采样的 token。
2. 置信度调度验证：confidence head 估计每个草稿 token 被接受的概率，scheduler 再根据当前系统负载决定验证多长 prefix。

所以 DSpark 不只是“多猜几个 token”，它还会判断“哪些 token 值得拿去验证”。

## 3. 为什么要把 OPD 放进 DSpark？

DeepSpec 公开实现里，DSpark 主要从 target-generated 轨迹训练。这个流程稳定，但仍然偏离真实 speculative decoding：真正部署时，target 验证的是 DSpark 自己提议的 draft block。

这就是 OPD 的入口：

- 先让当前 DSpark 提议 draft block。
- 让 target model 验证，记录 accepted/rejected token。
- 从这些 anchor replay draft-induced states。
- 让 target 在这些 state 上给 logits 或 top-k logprobs。
- 用 OPD loss 继续训练 DSpark。

换句话说，DSpark 原来学“标准答案长什么样”，OPD 让它补学“自己犯错时老师会怎么走”。

## 4. 相关研究给了什么启发？

Thinking Machines 的 OPD 博客强调：OPD 的价值在于学生 rollout 后的 dense token-level teacher signal。

verl 的实现把 OPD 工程化成两条路：

- PG OPD：单样本 reverse-KL 估计，当 advantage 用。
- GKD OPD：teacher top-k forward KL，直接反传。

DeepSeek-V4 报告又往前推了一步：他们认为单样本 KL 方差偏高，因此采用 full-vocabulary logit distillation，并通过缓存 teacher last hidden states 避免直接存全词表 logits。

Draft-OPD 则提醒我们：draft model 不能简单做完整自回归 rollout。更好的办法是 target-assisted rollout 保持序列质量，同时 replay speculative verification 暴露的错误位置。

## 5. 本项目怎么实现？

本仓库把 DSpark-OPD 拆成五个模块：

- `losses.py`：DSpark 的 CE/TV/confidence loss，加上 OPD 的 full-KL、top-k KL 和单样本 KL。
- `replay.py`：把 draft block 按 accepted/rejected 切 mask，支持 Draft-OPD 式加权。
- `scheduler.py`：实现 DSpark 论文里的 hardware-aware prefix scheduler。
- `calibration.py`：实现 Sequential Temperature Scaling，校准 confidence head。
- `torch_losses.py`：给 DeepSpec trainer 调用的 PyTorch 适配层。

推荐的训练闭环是：

1. 保留 DeepSpec 原 DSpark target-cache 监督。
2. 额外启动当前 DSpark 的 speculative rollout。
3. 记录 replay anchors 和 verification outcome。
4. 对 replay states 计算 teacher logits。
5. 用 `DSpark loss + OPD loss` 更新 drafter。

如果资源足够，优先 full-vocabulary reverse KL；如果资源受限，用 teacher top-k forward KL；再退一步，才用单样本 KL estimator。

## 6. 最后一层直觉

DSpark 负责“快”：多猜、少验证、别浪费 batch capacity。

OPD 负责“贴近真实推理”：别只在老师轨迹上学，要在自己生成的状态上学。

二者结合后的目标很自然：让 DSpark 在真实 speculative decoding 会遇到的位置，学会更像 target model，同时继续保留它的半自回归结构和置信度调度优势。

## 参考资料

- DeepSpec / DSpark：<https://github.com/deepseek-ai/DeepSpec>
- DeepSeek-V4 技术报告：<https://arxiv.org/abs/2606.19348>
- Thinking Machines OPD：<https://thinkingmachines.ai/blog/on-policy-distillation/>
- verl OPD：<https://verl.readthedocs.io/en/latest/algo/opd.html>
- Draft-OPD：<https://arxiv.org/abs/2605.29343>
