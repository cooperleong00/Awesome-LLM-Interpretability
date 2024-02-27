# Awesome-LLM-Interpretability

An curated list of LLM Interpretability related material.

## Tutorial

* **Concrete Steps to Get Started in Transformer Mechanistic Interpretability** [[Neel Nanda's blog]](https://www.neelnanda.io/mechanistic-interpretability/getting-started)

* **Mechanistic Interpretability Quickstart Guide** [[Neel Nanda's blog]](https://www.neelnanda.io/mechanistic-interpretability/getting-started)

## Library

* **TransformerLens** [[github]](https://github.com/neelnanda-io/TransformerLens)
* **CircuitsVis** [[github]](https://github.com/alan-cooney/CircuitsVis)
* **baukit** [[github]](https://github.com/davidbau/baukit)
  * Contains some methods for tracing and editing internal activations in a network.

## Survey

* **Toward Transparent AI: A Survey on Interpreting the Inner Structures of Deep Neural Networks** [[SaTML 2023]](https://ieeexplore.ieee.org/abstract/document/10136140) [[arxiv 2207]](https://arxiv.org/abs/2207.13243)

* **Neuron-level Interpretation of Deep NLP Models: A Survey** [[TACL 2022]](https://aclanthology.org/2022.tacl-1.74)

* **Explainability for Large Language Models: A Survey** [[TIST 2024]](https://dl.acm.org/doi/10.1145/3639372) [[arxiv 2309]](https://arxiv.org/abs/2309.01029)

*Note: These Alignment surveys discuss the relation between Interpretability and LLM Alignment.*

* **Large Language Model Alignment: A Survey** [[arxiv 2309]](https://arxiv.org/abs//2309.15025)

* **AI Alignment: A Comprehensive Survey** [[arxiv 2310]](https://arxiv.org/abs/2310.19852)  [[github]](https://github.com/PKU-Alignment/AlignmentSurvey)  [[website]](https://alignmentsurvey.com/)

## Video

* **Neel Nanda's Channel** [[Youtube]](https://www.youtube.com/@neelnanda2469)
* **Chris Olah - Looking Inside Neural Networks with Mechanistic Interpretability** [[Youtube]](https://www.youtube.com/watch?v=2Rdp9GvcYOE)
* **Concrete Open Problems in Mechanistic Interpretability: Neel Nanda at SERI MATS** [[Youtube]](https://www.youtube.com/watch?v=FnNTbqSG8w4)

* **BlackboxNLP's Channel** [[Youtube]](https://www.youtube.com/@blackboxnlp)

## Paper & Blog

### By Source

* **Transformer Circuits Thread** [[blog]](https://transformer-circuits.pub/)
* **AI Alignment Forum** [[forum]](https://www.alignmentforum.org/)
* **Lesswrong** [[forum]](https://www.lesswrong.com/)
* **Neel Nanda** [[blog]](https://www.neelnanda.io/) [[google scholar]](https://scholar.google.com/citations?user=GLnX3MkAAAAJ)
* **Jacob Steinhardt** [[google scholar]](https://scholar.google.com/citations?hl=en&user=LKv32bgAAAAJ)

* **Gor meva** [[google scholar]](https://scholar.google.com/citations?user=GxpQbSkAAAAJ)

* **David Bau** [[google scholar]](https://scholar.google.com/citations?hl=en&user=CYI6cKgAAAAJ)

* **Yonatan Belinkov** [[google scholar]](https://scholar.google.com/citations?user=K-6ujU4AAAAJ)

### By Topic

#### Tools/Techniques/Methods

##### General

* 🌟**A mathematical framework for transformer circuits** [[blog]](https://transformer-circuits.pub/2021/framework/index.html)

##### Embedding Projection

* **interpreting GPT: the logit lens** [[Lesswrong 2020]](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)

* **Analyzing Transformers in Embedding Space** [[ACL 2023]](https://aclanthology.org/2023.acl-long.893)
* **Eliciting Latent Predictions from Transformers with the Tuned Lens** [[arxiv 2303]](https://arxiv.org/abs/2303.08112)

##### Probing

* **Enhancing Neural Network Transparency through Representation Analysis** [[arxiv 2310]](https://arxiv.org/abs/2310.01405) [[openreview]](https://openreview.net/forum?id=aCgybhcZFi)

##### Causal Intervention

* **Analyzing And Editing Inner Mechanisms of Backdoored Language Models** [[arxiv 2303]](http://arxiv.org/abs/2302.12461)
* **Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations** [[arxiv 2303]](https://arxiv.org/abs/2303.02536)
* **Localizing Model Behavior with Path Patching** [[arxiv 2304]](https://arxiv.org/abs/2304.05969)
* **Interpretability at Scale: Identifying Causal Mechanisms in Alpaca** [[NIPS 2023]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f6a8b109d4d4fd64c75e94aaf85d9697-Abstract-Conference.html)
* **Towards Best Practices of Activation Patching in Language Models: Metrics and Methods** [[ICLR 2024]](https://openreview.net/forum?id=Hf17y6u9BC)
* **Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching** [[ICLR 2024]](https://openreview.net/forum?id=Ebt7JgMHv1)


##### Automation

* **Towards Automated Circuit Discovery for Mechanistic Interpretability** [[NIPS 2023]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/34e1dbe95d34d7ebaf99b9bcaeb5b2be-Abstract-Conference.html)
* **Neuron to Graph: Interpreting Language Model Neurons at Scale** [[arxiv 2305]](https://arxiv.org/abs/2305.19911) [[openreview]](https://openreview.net/forum?id=JBLHIR8kBZ)
* **Discovering Knowledge-Critical Subnetworks in Pretrained Language Models** [[openreview]](https://openreview.net/forum?id=Mkdwvl3Y8L)

##### Sparse Coding

* **Towards monosemanticity: Decomposing language models with dictionary learning** [[Transformer Circuits Thread]](https://transformer-circuits.pub/2023/monosemantic-features)
* **Sparse Autoencoders Find Highly Interpretable Features in Language Models** [[ICLR 2024]](https://openreview.net/forum?id=F76bwRSLeK)

##### Visulization

* **Interpreting Transformer's Attention Dynamic Memory and Visualizing the Semantic Information Flow of GPT** [[arxiv 2305]]([10.48550/arXiv.2305.13417](http://arxiv.org/abs/2305.13417)) [[github]](https://github.com/shacharKZ/Visualizing-the-Information-Flow-of-GPT)

##### Translation

* **Tracr: Compiled Transformers as a Laboratory for Interpretability** [[arxiv 2301]](http://arxiv.org/abs/2301.05062)

#### Task Solving/Function/Ability

##### General

* **Circuit Component Reuse Across Tasks in Transformer Language Models** [[ICLR 2024 spotlight]](https://openreview.net/forum?id=fpoAYV6Wsk)

##### Reasoning

* **Towards a Mechanistic Interpretation of Multi-Step Reasoning Capabilities of Language Models** [[EMNLP 2023]](https://aclanthology.org/2023.emnlp-main.299)

* **How Large Language Models Implement Chain-of-Thought?** [[openreview]](https://openreview.net/forum?id=b2XfOm3RJa)

#### Function

* **Interpretability in the wild: a circuit for indirect object identification in GPT-2 small** [[ICLR 2023]](https://openreview.net/forum?id=NpsVSN6o4ul)
* **How do Language Models Bind Entities in Context?** [[ICLR 2024]](https://openreview.net/forum?id=zb3b6oKO77)
* **How Language Models Learn Context-Free Grammars?** [[openreview]](https://openreview.net/forum?id=qnbLGV9oFL)
* **How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model** [[NIPS 2023]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/efbba7719cc5172d175240f24be11280-Abstract-Conference.html)
* **Does Circuit Analysis Interpretability Scale? Evidence from Multiple Choice Capabilities in Chinchilla** [[arxiv 2307]](http://arxiv.org/abs/2307.09458)
* **Forbidden Facts: An Investigation of Competing Objectives in Llama-2** [[arxiv 2312]](http://arxiv.org/abs/2312.08793)

##### Arithmetic Ability

* **Progress measures for grokking via mechanistic interpretability** [[ICLR 2023]](https://openreview.net/forum?id=9XFSbDPmdW)
* **Interpreting the Inner Mechanisms of Large Language Models in Mathematical Addition** [[openreview]](https://openreview.net/forum?id=VpCqrMMGVm)
* **Arithmetic with Language Models: from Memorization to Computation** [[openreview]](https://openreview.net/forum?id=YxzEPTH4Ny)
* **Carrying over Algorithm in Transformers** [[openreview]](https://openreview.net/forum?id=t3gOYtv1xV)
* **A simple and interpretable model of grokking modular arithmetic tasks** [[openreview]](https://openreview.net/forum?id=0ZUKLCxwBo)
* **Understanding Addition in Transformers** [[ICLR 2024]](https://openreview.net/forum?id=rIx1YXVWZb)

##### In-context Learning

* 🌟**In-context learning and induction heads**

* **In-Context Learning Creates Task Vectors** [[EMNLP 2023 Findings]](https://aclanthology.org/2023.findings-emnlp.624)
* **LLMs Represent Contextual Tasks as Compact Function Vectors** [[ICLR 2024]](https://openreview.net/forum?id=AwyxtyMwaG)
* **Where Does In-context Machine Translation Happen in Large Language Models?** [[openreview]](https://openreview.net/forum?id=3i7iNGxw6r)
* **In-Context Learning in Large Language Models: A Neuroscience-inspired Analysis of Representations** [[openreview]](https://openreview.net/forum?id=UEdS2lIgfY)

##### Factual Knowledge

* 🌟**Dissecting Recall of Factual Associations in Auto-Regressive Language Models** [[EMNLP 2023]](https://aclanthology.org/2023.emnlp-main.751)
* **Summing Up the Facts: Additive Mechanisms behind Factual Recall in LLMs** [[openreview]](https://openreview.net/forum?id=P2gnDEHGu3)
* **A Mechanism for Solving Relational Tasks in Transformer Language Models** [[openreview]](https://openreview.net/forum?id=ZmzLrl8nTa)
* **Overthinking the Truth: Understanding how Language Models Process False Demonstrations** [[ICLR 2024 spotlight]](https://openreview.net/forum?id=Tigr1kMDZy)

#### Component

##### General

* **The Hydra Effect: Emergent Self-repair in Language Model Computations** [[arxiv 2307]](https://arxiv.org/abs/2307.15771)
* **Exploring the Residual Stream of Transformers** [[arxiv 2312]](http://arxiv.org/abs/2312.12141)

##### Attention

* 🌟**In-context learning and induction heads** [[Transformer Circuits Thread]](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
* **On the Expressivity Role of LayerNorm in Transformers' Attention** [[ACL 2023 Findings]](https://aclanthology.org/2023.findings-acl.895.pdf)
* **On the Role of Attention in Prompt-tuning** [[ICML 2023]](https://openreview.net/forum?id=qorOnDor89)
* **Copy Suppression: Comprehensively Understanding an Attention Head** [[ICLR 2024]](https://openreview.net/forum?id=g8oaZRhDcf)
* **Successor Heads: Recurring, Interpretable Attention Heads In The Wild** [[ICLR 2024]](https://openreview.net/forum?id=kvcbV8KQsi)

##### MLP/FFN

* 🌟**Transformer Feed-Forward Layers Are Key-Value Memories** [[EMNLP 2021]](https://aclanthology.org/2021.emnlp-main.446)
* **Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space** [[EMNLP 2022]](https://aclanthology.org/2022.emnlp-main.3)
* **What does GPT store in its MLP weights? A case study of long-range dependencies** [[openreview]](https://openreview.net/forum?id=nUGFpDCu3W)

##### Neuron

* 🌟**Finding Neurons in a Haystack: Case Studies with Sparse Probing** [[TMLR 2023]](https://openreview.net/forum?id=JYs1R9IMJr)
* **Neurons in Large Language Models: Dead, N-gram, Positional** [[arxiv 2309]](http://arxiv.org/abs/2309.04827)

#### Learning Dynamics

##### Phase Transition/Grokking

* **Progress measures for grokking via mechanistic interpretability** [[ICLR 2023]](https://openreview.net/forum?id=9XFSbDPmdW)

* **A Toy Model of Universality: Reverse Engineering How Networks Learn Group Operations** [[ICML 2023]](https://openreview.net/forum?id=jCOrkuUpss)

* 🌟**The Mechanistic Basis of Data Dependence and Abrupt Learning in an In-Context Classification Task** [[ICLR 2024 oral]](https://openreview.net/forum?id=aN4Jf6Cx69)
  * Highest scores at ICLR 2024: 10, 10, 8, 8. And by one author only!

* **Sudden Drops in the Loss: Syntax Acquisition, Phase Transitions, and Simplicity Bias in MLMs** [[ICLR 2024 spotlight]](https://openreview.net/forum?id=MO5PiKHELW)
* **A simple and interpretable model of grokking modular arithmetic tasks** [[openreview]](https://openreview.net/forum?id=0ZUKLCxwBo)

##### Fine-tuning

* **Mechanistically analyzing the effects of fine-tuning on procedurally defined tasks** [[ICLR 2024]](https://openreview.net/forum?id=A0HKeKl4Nl)

#### Feature Representation/Probing-based

##### General

* **All Roads Lead to Rome? Exploring the Invariance of Transformers' Representations** [[arxiv 2305]](http://arxiv.org/abs/2305.14555)
* **Observable Propagation: Uncovering Feature Vectors in Transformers** [[openreview]](https://openreview.net/forum?id=sNWQUTkDmA)
* **In-Context Learning in Large Language Models: A Neuroscience-inspired Analysis of Representations** [[openreview]](https://openreview.net/forum?id=UEdS2lIgfY)
* **Challenges with unsupervised LLM knowledge discovery** [[arxiv 2312]](https://arxiv.org/abs/2312.10029)
* **Still No Lie Detector for Language Models: Probing Empirical and Conceptual Roadblocks** [[arxiv 2307]](http://arxiv.org/abs/2307.00175)

##### Linearity

* **Language Models Linearly Represent Sentiment** [[openreview]](https://openreview.net/forum?id=iGDWZFc7Ya)
* **Language Models Represent Space and Time** [[openreview]](https://openreview.net/forum?id=jE8xbmvFin)
* **The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets** [[openreview]](https://openreview.net/forum?id=CeJEfNKstt)
* **Linearity of Relation Decoding in Transformer Language Models** [[ICLR 2024]](https://openreview.net/forum?id=w7LU2s14kE)

#### Application

##### Inference-Time Intervention

* **Inference-Time Intervention: Eliciting Truthful Answers from a Language Model** [[NIPS 2023]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/81b8390039b7302c909cb769f8b6cd93-Abstract-Conference.html) [[github]](https://github.com/likenneth/honest_llama)
* **Steering Llama 2 via Contrastive Activation Addition** [[arxiv 2312]](http://arxiv.org/abs/2312.06681)

##### Knowledge Editing

* **Locating and Editing Factual Associations in GPT** (*ROME*) [[NIPS 2022]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html) [[github]](https://github.com/kmeng01/rome)
* **Inspecting and Editing Knowledge Representations in Language Models** [[arxiv 2304]](http://arxiv.org/abs/2304.00740) [[github]](https://github.com/evandez/REMEDI)

##### Hallucination

* **The Internal State of an LLM Knows When It's Lying** [[EMNLP 2023 Findings]](https://arxiv.org/abs/2304.13734)
* **Do Androids Know They're Only Dreaming of Electric Sheep?** [[arxiv 2312]](https://arxiv.org/abs/2312.17249)


