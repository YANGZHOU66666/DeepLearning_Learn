# Ragas指标

## Context Precision

对N个（有优先级顺序的）召回的文本块，k从1到N遍历，分别计算precision@k，然后求N个precision@k的和

$precision@k$定义：$precision@k=\frac{前k个文本块里需要用于生成回答的数量}{k}$

$Context\ Precision=\frac{1}{N}\sum_{k=1}^Nprecision@k$



## Faithfulness

