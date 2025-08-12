# SFT

预训练 - 大模型获得所有的能力

指令微调 - 如何更好的回答问题



指令微调和预训练没有本质上的不同

网络结构完全相同，Loss基本相同，只有数据不同



## Chat Template

微调需要和原厂的对话模板保持一致

有时需要加上指定的标识符：

```
Hugging Face格式的数据：
[{"role": "system", "content": "You are a helpful assistant."},
 {"role": "user", "content": "天空为什么是蓝色的？"},
 {"role": "assistant", "content": "这是由于光的散射引起的。"}]

实际用来训练大模型的数据格式：
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
天空为什么是蓝色的？<|eot_id|><|start_header_id|>assistant<|end_header_id|>
这是由于光的散射引起的。<|eot_id|>
```

每个大模型和自己的Tokenizer唯一对应，和Chat Template唯一对应

示例：Llama3.1版本的tokenizer_config

```
{
    "add_prefix_space": false,
    "added_tokens_decoder": {
        "151643": {
            "content": "<|endoftext|>",
            "lstrip": false,
            "normalized": false,
            "rstrip": false,
            "single_word": false,
            "special": true
        },
        "151644": {
            "content": "<|im_start|>",
            "lstrip": false,
            "normalized": false,
            "rstrip": false,
            "single_word": false,
            "special": true
        },
        "151645": {
            "content": "<|im_end|>",
            "lstrip": false,
            "normalized": false,
            "rstrip": false,
            "single_word": false,
            "special": true
        }
    },
    "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
    "bos_token": null,
    "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}[{ '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }]{% endif %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}",
    "clean_up_tokenization_spaces": false,
    "eos_token": "<|im_end|>",
    "errors": "replace",
    "model_max_length": 32768,
    "pad_token": "<|endoftext|>",
    "split_token": "<|endoftext|>",
    "spilt_special_tokens": false,
    "tokenizer_class": "Qwen2Tokenizer",
    "unk_token": null
}
```

将结构化的语料转化为大模型需要的格式：

```python
dialog = [{"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "天空为什么是蓝色的？"},
          {"role": "assistant", "content": "这是由于光的散射引起的。"}]

msg = tokenizer.apply_chat_template(dialog, tokenize=False)
```

推理时，使用add_generation_prompt=True，来给prompt后面加上额外的标记，让大模型知道要生成回答

```python
dialog = [{"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "天空为什么是蓝色的？"}]

msg = tokenizer.apply_chat_template(dialog, tokenize=False, add_generation_prompt=True)
```

