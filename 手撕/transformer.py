
x = embedding(x)
x += positional_encoding()

def attention_ffn(x):
    x = self_attention(x) + x
    x = layernorm(x)
    x = ffn(x) + x
    x = layernorm(x)

for i in range(k):
    x = attention_ffn(x)


