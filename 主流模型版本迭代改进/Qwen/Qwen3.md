# Qwen3 技术报告 笔记

## 基本结构 Architecture

6个Dense模型：Qwen3-0.6B，1.7B，4B，8B，14B，32B

和2个MoE模型：Qwen3-30B-A3B，Qwen3-235B-A22B



注意力机制：GQA（与Qwen-2.5一致）

位置编码：RoPE（与Qwen-2.5一致）

正则化层：RMSNorm（与Qwen-2.5一致）

激活函数：SwiGLU（与Qwen-2.5一致）

移除QKV-bias，引入QK-Norm（？）



MoE模型：256个专家，每次激活8个。去除了共享专家

引入了global-batch load balancing loss负载均衡损失

使用BBPE作为分词器，词表大小151669



## Pre-training

119种语言，**36 trillion** tokens

包括coding，STEM，**reasoning tasks**，books，multilingual texts，synthetic data

用Qwen2.5-VL解析PDF-like documents，使用Qwen2.5再进一步精化

Qwen2.5，Qwen2.5-Math，Qwen2.5-Coder用于合成数据

添加多语言数据

（注意到，引入了推理任务的数据，这一操作的依据可能是模型的推理能力上限由预训练决定）

### Pre-training Stage

S1-General Stage: 30 trillion tokens，最大序列长度4096

S2-Reasoning Stage：添加STEM、推理、coding、合成数据的比例，最大序列长度4096

S3-Long Context Stage：75%有16384-32768 tokens，25%有4096-16384 tokens。提高了RoPE的频率，引入YARN和DCA，为长文本做服务



### Pre-training Evaluation

若干个指标，和Qwen2.5及DeepSeek，Llama，Gemma等模型对比



## Post-training

总体概览：先通过四阶段训练出最大的235B(MoE)/32B(Dense)的模型，然后通过蒸馏的方式训练其他模型

![](./assets/Qwen3 post-train概览.png)

两个核心点：Thinking Control和Strong-to-Weak Distillation



### Long CoT Cold Start

先用长思维链冷启动，本质上是SFT

