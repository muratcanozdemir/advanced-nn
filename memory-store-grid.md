# FFN and Attention as One Lookup Operation

Notes on Alex Wang, *You could have invented transformers from memory* (October 2026), with a design-space grid and a check against two primary sources.

## 1. The rephrasing

FFN layers and self-attention layers compute the same operation: a similarity-weighted readout of stored values.

```
f(x) = Σ_i n_i · v_i      where  w_i = σ(k_i · x)  and  n = normalize(w)
```

Four choices define each layer:

- **Memory source.** Where the keys and values come from: trained weights, the current context, an external index, or a combination.
- **Kernel σ.** How a similarity becomes a weight. The kernel sets how selective a read is.
- **Normalisation.** Whether the memories compete for weight, and over which pool of memories.
- **Write rule.** How a memory enters the store.

An FFN reads from memories that training wrote into the weights. Self-attention reads from memories that the context wrote at inference time. The kernel sets how selective each read is. The normalisation sets whether memories compete, and against which other memories.

### Slide version

> FFN and attention are two settings of one lookup operation. The FFN reads memories that training wrote into the weights. Attention reads memories that the prompt wrote at inference. In 2014, an MLP scored attention's memories. By 2017, the MLP was recognised as a memory store itself. In 2019, putting both kinds of memory in one softmax matched the standard transformer with fewer parameters, on small-scale benchmarks.

## 2. The design-space grid

| Layer | Memory source | Kernel σ | Normalisation | Write rule | Capacity, random keys, d_k = 64 |
|---|---|---|---|---|---|
| Outer-product store / linear attention | Context, fixed d_k × d_v state | Linear | None | Additive Hebbian (v kᵀ) | ≈ d_k = 64 |
| Optimal linear store | Weights | Linear, whitened | None | Least squares | ≈ 2d_k = 128 |
| ReLU FFN | Weights | ReLU | None | Gradient, offline | ≈ 2d_k = 128 |
| GLU / SwiGLU FFN | Weights | ≈ ReLU² | None | Gradient, offline | ≈ 2,700 |
| Self-attention | Context, grows with M | exp(β·s) | Softmax over context | Append, one memory per token | ≈ 1.2M with exact key; ≈ 12k at query cosine 0.71 |
| All-attention (Sukhbaatar 2019) | Weights + context | exp(β·s) | **Joint** softmax over both pools | Gradient + append | Not measured |
| Bahdanau attention (2014) | Context (encoder states) | exp(MLP score) | Softmax over context | Append | Not measured |
| DeltaNet | Context, fixed state | Linear | None | Delta rule (overwrite by error) | Not measured |
| Product-key memory / MoE router | Weights | Top-k | Softmax over top-k | Gradient, offline | Not measured |
| kNN-LM | External datastore | exp(−distance), top-k | Softmax over top-k | Append to index | Not measured |

Notes on the grid:

- The capacity column comes from Wang's post, except the noisy-query figure. That figure comes from the formula `M ≈ exp(2β(c − β/d_k))` with c = 0.71 and β = 8, which gives about 12k. A numpy run with `isr.py` crosses ratio 1 just below M = 16,384.
- The memory source and the write rule set the cost of a layer. The kernel and the normalisation set its recall quality.
- Normalisation has a scope dimension, not only on or off. Section 3 gives the evidence.
- Not verified against a primary source in this session: the DeltaNet, product-key memory, MoE and kNN-LM rows. These rows come from Wang's reference list and from training knowledge.

## 3. Two primary sources

### 3.1 Bahdanau, Cho & Bengio — arXiv 1409.0473

Verified facts:

- v1 was submitted on 1 September 2014. The paper appeared at ICLR 2015. The final arXiv version (v7) is from May 2016.
- The context vector c_i is a weighted sum of encoder annotations h_j. The weights α_ij are a softmax over scores e_ij = a(s_{i−1}, h_j).
- The authors parametrize the alignment model `a` as a feedforward network, trained jointly with the rest of the system.
- The authors interpret the weighted sum as an expected annotation over possible alignments.

Position in the grid: context source, softmax normalisation, append write. The kernel is the exponential of an MLP score, not of a dot product.

Implication: in the first neural attention, the MLP sat inside attention as the similarity function. The dot-product kernel arrived later, with Vaswani et al. Between 2014 and 2017, the relationship reversed. First the MLP scored the memories. Then the MLP became a memory store itself.

### 3.2 Sukhbaatar, Grave, Lample, Jégou & Joulin — arXiv 1907.01470

Verified facts:

- v1 was submitted on 2 July 2019, by Facebook AI Research.
- Section 4.1 converts the FFN into attention. It replaces ReLU with softmax and removes the biases. The rows of the first matrix become keys. The columns of the second matrix become values. This matches the appendix in Vaswani et al., arXiv 1706.03762 v2 and v3.
- The all-attention layer adds N learned, context-independent key-value pairs ("persistent vectors") to the context keys and values. A single softmax runs over both pools. The FFN sublayer is removed.

Results:

| Benchmark | All-attention | Comparison model |
|---|---|---|
| enwik8, small | 1.01 bpc, 39M params | 1.02 bpc, Transformer + adaptive span, 39M |
| enwik8, large | 0.98 bpc, 114M params | 0.98 bpc, Transformer + adaptive span, 209M |
| text8, large | 1.08 test bpc, 114M params | 1.07 test bpc, Transformer + adaptive span, 209M |
| WikiText-103 | 20.6 test ppl, 133M params | 24.0 test ppl, Transformer-XL standard, 151M |

Ablation on text8:

- N = 0 (no persistent vectors, no FFN) performs poorly.
- A joint softmax ("all-attn") beats separate softmaxes for the context pool and the persistent pool ("attn-split").
- Splitting heads into context-only and persistent-only groups ("head-split") performs worse.
- "FF-attn" keeps the FFN as a separate sublayer and only swaps ReLU for softmax. It performs worse than all-attn. The authors conclude that the switch from ReLU to softmax alone is not sufficient.

The paper attributes the first introduction of attention to Jordan & Jacobs, in the context of mixture of experts.

## 4. What the sources change

1. **"Separate histories" is too strong.** Sukhbaatar et al. trace attention back to mixture-of-experts gating. MoE gating is a softmax over expert networks, which is a memory read over parametric components. A documented common ancestor therefore exists at the gating level. Corrected statement: FFN and attention converge on one readout form, and one recorded lineage of attention passes through softmax gating over experts. The Jordan & Jacobs paper was not fetched. This point rests on the attribution in Sukhbaatar et al.

2. **Normalisation scope is a real design axis.** The all-attn versus attn-split result shows a measurable gain when parametric and contextual memories compete in one softmax. For that reason, the grid lists joint normalisation as a separate value.

3. **Counterpoint to Wang's capacity argument.** The post predicts that a softmax kernel gives an FFN far more capacity than ReLU. In the Sukhbaatar ablation, the sharper kernel alone (FF-attn) did not produce the gain. The gain came from joint competition between the two pools. One interpretation: at this scale, FFN capacity was not the binding constraint. This interpretation is a guess. The FF-attn arm also changed depth (24 layers, FFN width 3,072) to keep the parameter count constant, so the comparison has a confound.

4. **Limits of the Sukhbaatar evidence.**
   - The models are 2019-scale, up to about 200M parameters.
   - The benchmarks are character-level text and WikiText-103 only.
   - Both arms use adaptive span.
   - The sections read report no run-to-run variance.
   - The 114M versus 209M comparison is between different configurations, not one controlled change.
   - Current large LLMs keep a separate SwiGLU FFN or MoE in the FFN slot. This statement comes from training knowledge. A plausible reason is that a dense FFN matmul costs less than softmax over N + T keys per head. That reason is not verified.

## 5. Where the equivalence stops

- **Normalisation changes the semantics, not only the sharpness.** Softmax makes memories compete: weights sum to 1, so a higher weight on one memory lowers the others. ReLU memories are independent and can all fire together. An MLP can therefore superpose many partial matches, while attention tends to select.
- **The analogy covers one layer's readout.** It does not cover depth. Composition across layers produces induction heads and multi-step lookups, which no single-store capacity number captures.
- **An MLP also computes, not only retrieves.** Geva et al. found that some hidden units act as clean pattern-matching keys. Many others act as distributed features with no single interpretable key.
- **The capacity numbers do not compare equal costs.** The outer-product store is a fixed d_k × d_v matrix. An MLP that holds M keys has M · (d_k + d_v) parameters. Attention stores every memory in the KV cache and does O(M) work per query.
- **Wang's test uses the exact stored key.** With query noise, the capacity of softmax attention falls exponentially, because the exponent scales with the margin between target similarity and the β/d_k term.

## 6. Replication results (`isr.py`, d_k = 64, median of 64 trials)

Interference-to-signal ratio. Recall fails above 1.0.

| Kernel | Query cosine | M = 64 | M = 128 | M = 2,048 | M = 16,384 |
|---|---|---|---|---|---|
| linear | 1.00 | 0.99 | 1.38 | 5.73 | 16.19 |
| ReLU | 1.00 | 0.65 | 0.99 | 3.92 | 10.96 |
| ReLU² | 1.00 | 0.14 | 0.21 | 0.83 | 2.41 |
| ReLU³ | 1.00 | 0.04 | 0.05 | 0.22 | 0.65 |
| softmax, β = 8 | 1.00 | 0.01 | 0.01 | 0.04 | 0.11 |
| linear | 0.89 | 1.09 | 1.56 | 6.28 | 18.08 |
| ReLU² | 0.89 | 0.18 | 0.25 | 1.06 | 2.93 |
| softmax, β = 8 | 0.89 | 0.01 | 0.02 | 0.09 | 0.27 |
| linear | 0.71 | 1.40 | 1.94 | 8.09 | 22.40 |
| ReLU² | 0.71 | 0.29 | 0.41 | 1.68 | 4.70 |
| softmax, β = 8 | 0.71 | 0.06 | 0.10 | 0.45 | 1.19 |

The exact-key rows match the post's charts to within about 0.05.

## Sources

- Wang, A. — [You could have invented transformers from memory](https://alexwang.ai/posts/associative-memory/)
- Bahdanau, D., Cho, K., Bengio, Y. — [Neural Machine Translation by Jointly Learning to Align and Translate, arXiv 1409.0473](https://arxiv.org/abs/1409.0473)
- Sukhbaatar, S. et al. — [Augmenting Self-attention with Persistent Memory, arXiv 1907.01470](https://arxiv.org/abs/1907.01470)
- Vaswani, A. et al. — [Attention Is All You Need, arXiv 1706.03762v2](https://arxiv.org/pdf/1706.03762v2) (appendix "Attention over Parameters" present in v2 and v3; absent in v4 and v5)
