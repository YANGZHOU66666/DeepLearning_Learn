# DeepSeek-V4 Technical Report

## 宏观梳理（Abstract+Introduction）

V4-Pro：1.6T（1600B）+49B activated；V4-Flash：284B+13B activated

**模型架构：**

注意力层：hybrid，Compressed Sparse Attention（CSA）+Heavily Compressed Attention（HCA）

残差连接：Manifold-Constrained Hyper-Connections（mHC）代替Standard Hyper-Connections

优化器：Muon代替AdamW

MoE层：沿用DeepSeek-V3的DeepSeek MoE

MTP：沿用DeepSeek-V3的MTP

**训练：**





**推理：**

1M上下文

V4-Pro相较于V3.2只需要27%单token推理FLOPs，10%的KV cache



**各种评测指标：**

