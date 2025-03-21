# Awesome-LLM-Interpretability

A curated list of LLM Interpretability related material.

## ToC
- [Tutorial](#tutorial)
- [History](#history)
- [Code](#code)
  - [Library](#library)
  - [Codebase](#codebase)
- [Survey](#survey)
- [Video](#video)
- [Paper & Blog](#paper--blog)
  - [By Source](#by-source)
  - [By Topic](#by-topic)
    - [Tools/Techniques/Methods](#toolstechniquesmethods)
      - [General](#general)
      - [Embedding Projection](#embedding-projection)
      - [Probing](#probing)
      - [Causal Intervention](#causal-intervention)
      - [Automation](#automation)
      - [Sparse Coding](#sparse-coding)
      - [Visualization](#visualization)
      - [Translation](#translation)
      - [Evaluation/Dataset/Benchmark](#evaluationdatasetbenchmark)
    - [Task Solving/Function/Ability](#task-solvingfunctionability)
      - [General](#general-1)
      - [Reasoning](#reasoning)
      - [Function](#function)
      - [Arithmetic Ability](#arithmetic-ability)
      - [In-context Learning](#in-context-learning)
      - [Factual Knowledge](#factual-knowledge)
      - [Multilingual/Crosslingual](#multilingualcrosslingual)
      - [Multimodal](#multimodal)
    - [Component](#component)
      - [General](#general-2)
      - [Attention](#attention)
      - [MLP/FFN](#mlpffn)
      - [Neuron](#neuron)
    - [Learning Dynamics](#learning-dynamics)
      - [General](#general-3)
      - [Phase Transition/Grokking](#phase-transitiongrokking)
      - [Fine-tuning](#fine-tuning)
    - [Feature Representation/Probing-based](#feature-representationprobing-based)
      - [General](#general-4)
      - [Linearity](#linearity)
    - [Application](#application)
      - [Inference-Time Intervention/Activation Steering](#inference-time-interventionactivation-steering)
      - [Knowledge/Model Editing](#knowledgemodel-editing)
      - [Hallucination](#hallucination)
      - [Pruning/Redundancy Analysis](#pruningredundancy-analysis)


## Tutorial

* **Concrete Steps to Get Started in Transformer Mechanistic Interpretability** [[Neel Nanda's blog]](https://www.neelnanda.io/mechanistic-interpretability/getting-started)
* **Mechanistic Interpretability Quickstart Guide** [[Neel Nanda's blog]](https://www.neelnanda.io/mechanistic-interpretability/getting-started)
* **ARENA Mechanistic Interpretability Tutorials by Callum McDougall** [[website]](https://arena-ch1-transformers.streamlit.app/)
* **200 Concrete Open Problems in Mechanistic Interpretability: Introduction by Neel Nanda** [[AlignmentForum]](https://www.alignmentforum.org/s/yivyHaCAmMJ3CqSyj)
* **Transformer-specific Interpretability** [[EACL 2023 Tutorial]](https://projects.illc.uva.nl/indeep/tutorial/)

## History

* **Mechanistic?** [[BlackBoxNLP workshop at EMNLP 2024]](https://arxiv.org/abs/2410.09087)
  * This paper explores the multiple definitions and uses of "mechanistic interpretability," tracing its evolution in NLP research and revealing a critical divide within the interpretability community.

## Code

### Library

* **TransformerLens** [[github]](https://github.com/neelnanda-io/TransformerLens)
  * A library for mechanistic interpretability of GPT-style language models
* **CircuitsVis** [[github]](https://github.com/alan-cooney/CircuitsVis)
  * Mechanistic Interpretability visualizations
* **baukit** [[github]](https://github.com/davidbau/baukit)
  * Contains some methods for tracing and editing internal activations in a network.
* **transformer-debugger** [[github]](https://github.com/openai/transformer-debugger)
  * Transformer Debugger (TDB) is a tool developed by OpenAI's Superalignment team with the goal of supporting investigations into specific behaviors of small language models. The tool combines automated interpretability techniques with sparse autoencoders.
* **pyvene** [[github]](https://github.com/stanfordnlp/pyvene)
  * Supports customizable interventions on a range of different PyTorch modules
  * Supports complex intervention schemes with an intuitive configuration format, and its interventions can be static or include trainable parameters.
* **ViT-Prisma** [[github]](https://github.com/soniajoseph/ViT-Prisma)
  * An open-source mechanistic interpretability library for vision and multimodal models.
* **pyreft** [[github]](https://github.com/stanfordnlp/pyreft)
  * A Powerful, Parameter-Efficient, and Interpretable way of fine-tuning
* **SAELens** [[github]](https://github.com/jbloomAus/SAELens)
  * Training and analyzing sparse autoencoders on Language Models

### Codebase

* **mamba interpretability** [[github]](https://github.com/Phylliida/mamba_interp)

## Survey

* **A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of Large Language Models** [[arxiv2503]](https://arxiv.org/abs/2503.05613)
* **Representation Engineering for Large-Language Models: Survey and Research Challenges** [[arxiv2502]](http://arxiv.org/abs/2502.17601)
* **Toward Transparent AI: A Survey on Interpreting the Inner Structures of Deep Neural Networks** [[SaTML 2023]](https://ieeexplore.ieee.org/abstract/document/10136140) [[arxiv 2207]](https://arxiv.org/abs/2207.13243)
* **Neuron-level Interpretation of Deep NLP Models: A Survey** [[TACL 2022]](https://aclanthology.org/2022.tacl-1.74)
* **Explainability for Large Language Models: A Survey** [[TIST 2024]](https://dl.acm.org/doi/10.1145/3639372) [[arxiv 2309]](https://arxiv.org/abs/2309.01029)
* **Opening the Black Box of Large Language Models: Two Views on Holistic Interpretability** [[arxiv 2402]](http://arxiv.org/abs/2402.10688)
* **Usable XAI: 10 Strategies Towards Exploiting Explainability in the LLM Era** [[arxiv 2403]](http://arxiv.org/abs/2403.08946)
* **Mechanistic Interpretability for AI Safety -- A Review** [[arxiv 2404]](http://arxiv.org/abs/2404.14082)
* **A Primer on the Inner Workings of Transformer-based Language Models** [[arxiv  2405]](https://arxiv.org/abs/2405.00208)
* **🌟A Practical Review of Mechanistic Interpretability for Transformer-Based Language Models** [[arxiv 2407]](http://arxiv.org/abs/2407.02646)
* **Internal Consistency and Self-Feedback in Large Language Models: A Survey** [[arxiv 2407]](https://arxiv.org/abs/2407.14507)
* **The Quest for the Right Mediator: A History, Survey, and Theoretical Grounding of Causal Interpretability** [[arxiv 2408]](https://arxiv.org/abs/2408.01416)
* **Attention Heads of Large Language Models: A Survey** [[arxiv 2409]](https://arxiv.org/abs/2409.03752) [[github]](https://github.com/IAAR-Shanghai/Awesome-Attention-Heads)

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

* **🌟ICML 2024 Workshop on Mechanistic Interpretability** [[openreview]](https://openreview.net/group?id=ICML.cc/2024/Workshop/MI#tab-accept-oral)

* **🌟Transformer Circuits Thread** [[blog]](https://transformer-circuits.pub/)
* **BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP** [[workshop]](https://aclanthology.org/venues/blackboxnlp/)
* **AI Alignment Forum** [[forum]](https://www.alignmentforum.org/)
* **Lesswrong** [[forum]](https://www.lesswrong.com/)
* **Neel Nanda** [[blog]](https://www.neelnanda.io/) [[google scholar]](https://scholar.google.com/citations?user=GLnX3MkAAAAJ)
* **Mor Geva** [[google scholar]](https://scholar.google.com/citations?user=GxpQbSkAAAAJ)
* **David Bau** [[google scholar]](https://scholar.google.com/citations?hl=en&user=CYI6cKgAAAAJ)
* **Jacob Steinhardt** [[google scholar]](https://scholar.google.com/citations?hl=en&user=LKv32bgAAAAJ)
* **Yonatan Belinkov** [[google scholar]](https://scholar.google.com/citations?user=K-6ujU4AAAAJ)

### By Topic

[[Interactive UI]](https://cooperleong00.github.io/llminterp/)

![https://cooperleong00.github.io/llminterp/](screenshot.png)


#### Tools/Techniques/Methods

##### General

* 🌟**A mathematical framework for transformer circuits** [[blog]](https://transformer-circuits.pub/2021/framework/index.html)
* **Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models** [[arxiv]](http://arxiv.org/abs/2401.06102)

##### Embedding Projection

* **🌟interpreting GPT: the logit lens** [[Lesswrong 2020]](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)

* **🌟Analyzing Transformers in Embedding Space** [[ACL 2023]](https://aclanthology.org/2023.acl-long.893)
* **Eliciting Latent Predictions from Transformers with the Tuned Lens** [[arxiv 2303]](https://arxiv.org/abs/2303.08112)
* **An Adversarial Example for Direct Logit Attribution: Memory Management in gelu-4l** [arxiv 2310](http://arxiv.org/abs/2310.07325)
* **Future Lens: Anticipating Subsequent Tokens from a Single Hidden State** [[CoNLL 2023]](https://aclanthology.org/2023.conll-1.37/)
* **SelfIE: Self-Interpretation of Large Language Model Embeddings** [[arxiv 2403]](https://arxiv.org/abs/2403.10949)
* **InversionView: A General-Purpose Method for Reading Information from Neural Activations** [[ICML 2024 MI Workshop]](https://openreview.net/forum?id=P7MW0FahEq)

##### Probing

* **Enhancing Neural Network Transparency through Representation Analysis** [[arxiv 2310]](https://arxiv.org/abs/2310.01405) [[openreview]](https://openreview.net/forum?id=aCgybhcZFi)

##### Causal Intervention

* **Analyzing And Editing Inner Mechanisms of Backdoored Language Models** [[arxiv 2303]](http://arxiv.org/abs/2302.12461)
* **Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations** [[arxiv 2303]](https://arxiv.org/abs/2303.02536)
* **Localizing Model Behavior with Path Patching** [[arxiv 2304]](https://arxiv.org/abs/2304.05969)
* **Interpretability at Scale: Identifying Causal Mechanisms in Alpaca** [[NIPS 2023]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f6a8b109d4d4fd64c75e94aaf85d9697-Abstract-Conference.html)
* **Towards Best Practices of Activation Patching in Language Models: Metrics and Methods** [[ICLR 2024]](https://openreview.net/forum?id=Hf17y6u9BC)
* **Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching** [[ICLR 2024]](https://openreview.net/forum?id=Ebt7JgMHv1)
  * **A Reply to Makelov et al. (2023)'s "Interpretability Illusion" Arguments** [[arxiv 2401]](https://arxiv.org/abs/2401.12631)
* **CausalGym: Benchmarking causal interpretability methods on linguistic tasks** [[arxiv 2402]](http://arxiv.org/abs/2402.12560)
* **🌟How to use and interpret activation patching** [[arxiv 2404]](http://arxiv.org/abs/2404.15255)

##### Automation

* **Towards Automated Circuit Discovery for Mechanistic Interpretability** [[NIPS 2023]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/34e1dbe95d34d7ebaf99b9bcaeb5b2be-Abstract-Conference.html)
* **Neuron to Graph: Interpreting Language Model Neurons at Scale** [[arxiv 2305]](https://arxiv.org/abs/2305.19911) [[openreview]](https://openreview.net/forum?id=JBLHIR8kBZ)
* **Discovering Variable Binding Circuitry with Desiderata** [[arxiv 2307]](http://arxiv.org/abs/2307.03637)
* **Discovering Knowledge-Critical Subnetworks in Pretrained Language Models** [[openreview]](https://openreview.net/forum?id=Mkdwvl3Y8L)
* **Attribution Patching Outperforms Automated Circuit Discovery** [[arxiv 2310]](https://arxiv.org/abs/2310.10348)
* **AtP\*: An efficient and scalable method for localizing LLM behaviour to components** [[arxiv 2403]](https://arxiv.org/abs/2403.00745)
* **Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms** [[arxiv 2403]](http://arxiv.org/abs/2403.17806)
* **Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models** [[arxiv 2403]](https://arxiv.org/abs/2403.19647)
* **Automatically Identifying Local and Global Circuits with Linear Computation Graphs** [[arxiv 2405]](https://arxiv.org/abs/2405.13868)
* **Sparse Autoencoders Enable Scalable and Reliable Circuit Identification in Language Models** [[arxiv 2405]](http://arxiv.org/abs/2405.12522)
* **Hypothesis Testing the Circuit Hypothesis in LLMs** [[ICML 2024 MI Workshop]](https://openreview.net/forum?id=ibSNv9cldu)

##### Sparse Coding

* 🌟**Towards monosemanticity: Decomposing language models with dictionary learning** [[Transformer Circuits Thread]](https://transformer-circuits.pub/2023/monosemantic-features)
* **Sparse Autoencoders Find Highly Interpretable Features in Language Models** [[ICLR 2024]](https://openreview.net/forum?id=F76bwRSLeK)
* **Open Source Sparse Autoencoders for all Residual Stream Layers of GPT2-Small** [[Alignment Forum]](https://www.alignmentforum.org/posts/f9EgfLSurAiqRJySD/open-source-sparse-autoencoders-for-all-residual-stream)
* **Attention SAEs Scale to GPT-2 Small** [[Alignment Forum]](https://www.alignmentforum.org/posts/FSTRedtjuHa4Gfdbr/attention-saes-scale-to-gpt-2-small)
* **We Inspected Every Head In GPT-2 Small using SAEs So You Don’t Have To** [[Alignment Forum]](https://www.alignmentforum.org/posts/xmegeW5mqiBsvoaim/we-inspected-every-head-in-gpt-2-small-using-saes-so-you-don)
* **Understanding SAE Features with the Logit Lens** [[Alignment Forum]](https://www.alignmentforum.org/posts/qykrYY6rXXM7EEs8Q/understanding-sae-features-with-the-logit-lens)
* **Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet** [[Transformer Circuits Thread]](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
* **Sparse Autoencoders Enable Scalable and Reliable Circuit Identification in Language Models** [[arxiv 2405]](http://arxiv.org/abs/2405.12522)
* **Scaling and evaluating sparse autoencoders** [[arxiv 2406]](https://arxiv.org/abs/2406.04093) [[code]](https://github.com/openai/sparse_autoencoder/)
* **Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models** [[ICML 2024 MI Workshop]](https://openreview.net/forum?id=qzsDKwGJyB)
* **Sparse Autoencoders Match Supervised Features for Model Steering on the IOI Task** [[ICML 2024 MI Workshop]](https://openreview.net/forum?id=JdrVuEQih5)
* **Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning** [[ICML 2024 MI Workshop]](https://openreview.net/forum?id=bcV7rhBEcM)
* **Transcoders find interpretable LLM feature circuits** [[ICML 2024 MI Workshop]](https://openreview.net/forum?id=GWqzUR2dOX)
* **Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders** [[arxiv 2407]](http://arxiv.org/abs/2407.14435)
* **Sparse Autoencoders Reveal Temporal Difference Learning in Large Language Models** [[arxiv 2410]](https://arxiv.org/abs/2410.01280)
* **Mechanistic Permutability: Match Features Across Layers** [[arxiv 2410]](https://arxiv.org/abs/2410.07656)
* **Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language Models** [[arxiv 2410]](https://arxiv.org/abs/2410.06981)
* **Investigating Sensitive Directions in GPT-2: An Improved Baseline and Comparative Analysis of SAEs** [[arxiv 2410]](https://arxiv.org/abs/2410.12555)


##### Visualization

* **Interpreting Transformer's Attention Dynamic Memory and Visualizing the Semantic Information Flow of GPT** [[arxiv 2305]]([10.48550/arXiv.2305.13417](http://arxiv.org/abs/2305.13417)) [[github]](https://github.com/shacharKZ/Visualizing-the-Information-Flow-of-GPT)
* **Sparse AutoEncoder Visulization** [[github]](https://github.com/callummcdougall/sae_vis)
  * **SAE-VIS: Announcement Post** [[lesswrong]](https://www.lesswrong.com/posts/nAhy6ZquNY7AD3RkD/sae-vis-announcement-post-1)
* **LM Transparency Tool: Interactive Tool for Analyzing Transformer Language Models** [[arxiv 2404]](http://arxiv.org/abs/2404.07004) [[github]](https://github.com/facebookresearch/ llm-transparency-tool)

##### Translation

* **Tracr: Compiled Transformers as a Laboratory for Interpretability** [[arxiv 2301]](http://arxiv.org/abs/2301.05062)
* **Opening the AI black box: program synthesis via mechanistic interpretability** [[arxiv 2402]](http://arxiv.org/abs/2402.05110)
* **An introduction to graphical tensor notation for mechanistic interpretability** [[arxiv 2402]](http://arxiv.org/abs/2402.01790)

##### Evaluation/Dataset/Benchmark

* **Look Before You Leap: A Universal Emergent Decomposition of Retrieval Tasks in Language Models** [[arxiv 2312]](http://arxiv.org/abs/2312.10091)
* **RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations** [[arxiv 2402]](https://arxiv.org/abs/2402.17700)
* **Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control** [[arxiv 2405]](http://arxiv.org/abs/2405.08366)
* **InterpBench: Semi-Synthetic Transformers for Evaluating Mechanistic Interpretability Techniques** [[arxiv 2407]](http://arxiv.org/abs/2407.14494)

#### Task Solving/Function/Ability

##### General

* **Circuit Component Reuse Across Tasks in Transformer Language Models** [[ICLR 2024 spotlight]](https://openreview.net/forum?id=fpoAYV6Wsk)
* **Towards Universality: Studying Mechanistic Similarity Across Language Model Architectures** [[arxvi 2410]](https://arxiv.org/abs/2410.06672)
* **From Tokens to Words: On the Inner Lexicon of LLMs** [[arxiv 2410]](https://arxiv.org/abs/2410.05864)

##### Reasoning

* **Towards a Mechanistic Interpretation of Multi-Step Reasoning Capabilities of Language Models** [[EMNLP 2023]](https://aclanthology.org/2023.emnlp-main.299)
* **How Large Language Models Implement Chain-of-Thought?** [[openreview]](https://openreview.net/forum?id=b2XfOm3RJa)
* **Do Large Language Models Latently Perform Multi-Hop Reasoning?** [[arxiv 2402]](http://arxiv.org/abs/2402.16837)
* **How to think step-by-step: A mechanistic understanding of chain-of-thought reasoning** [[arxiv 2402]](https://arxiv.org/abs/2402.18312)
* **Focus on Your Question! Interpreting and Mitigating Toxic CoT Problems in Commonsense Reasoning** [[arxiv 2402]](https://arxiv.org/abs/2402.18344)
* **Iteration Head: A Mechanistic Study of Chain-of-Thought** [[arxiv 2406]](https://arxiv.org/abs/2406.02128)
* **From Sparse Dependence to Sparse Attention: Unveiling How Chain-of-Thought Enhances Transformer Sample Efficiency** [[arxiv 2410]](https://arxiv.org/abs/2410.05459)

##### Function

* 🌟**Interpretability in the wild: a circuit for indirect object identification in GPT-2 small** [[ICLR 2023]](https://openreview.net/forum?id=NpsVSN6o4ul)
* **Entity Tracking in Language Models** [[ACL 2023]](https://aclanthology.org/2023.acl-long.213)
* **How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model** [[NIPS 2023]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/efbba7719cc5172d175240f24be11280-Abstract-Conference.html)
* **Can Transformers Learn to Solve Problems Recursively?** [[arxiv 2305]](http://arxiv.org/abs/2305.14699)
* **Analyzing And Editing Inner Mechanisms of Backdoored Language Models** [[NeurIPS 2023 Workshop]](https://openreview.net/forum?id=e9F4fB23o0)
* **Does Circuit Analysis Interpretability Scale? Evidence from Multiple Choice Capabilities in Chinchilla** [[arxiv 2307]](http://arxiv.org/abs/2307.09458)
* **Refusal mechanisms: initial experiments with Llama-2-7b-chat** [[AlignmentForum 2312]](https://www.alignmentforum.org/posts/pYcEhoAoPfHhgJ8YC/refusal-mechanisms-initial-experiments-with-llama-2-7b-chat)
* **Forbidden Facts: An Investigation of Competing Objectives in Llama-2** [[arxiv 2312]](http://arxiv.org/abs/2312.08793)
* **How do Language Models Bind Entities in Context?** [[ICLR 2024]](https://openreview.net/forum?id=zb3b6oKO77)
* **How Language Models Learn Context-Free Grammars?** [[openreview]](https://openreview.net/forum?id=qnbLGV9oFL)
* 🌟**A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity** [[arxiv 2401]](http://arxiv.org/abs/2401.01967)
* **Do Llamas Work in English? On the Latent Language of Multilingual Transformers** [[arxiv 2402]](http://arxiv.org/abs/2402.10588)
* **Evidence of Learned Look-Ahead in a Chess-Playing Neural Network** [[arxiv2406]](https://arxiv.org/abs/2406.00877)
* **How much do contextualized representations encode long-range context?** [[arxiv 2410]](https://arxiv.org/abs/2410.12292)

##### Arithmetic Ability

* 🌟**Progress measures for grokking via mechanistic interpretability** [[ICLR 2023]](https://openreview.net/forum?id=9XFSbDPmdW)
* 🌟**The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks** [[NIPS 2023]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/56cbfbf49937a0873d451343ddc8c57d-Abstract-Conference.html)
* **Interpreting the Inner Mechanisms of Large Language Models in Mathematical Addition** [[openreview]](https://openreview.net/forum?id=VpCqrMMGVm)
* **Arithmetic with Language Models: from Memorization to Computation** [[openreview]](https://openreview.net/forum?id=YxzEPTH4Ny)
* **Carrying over Algorithm in Transformers** [[openreview]](https://openreview.net/forum?id=t3gOYtv1xV)
* **A simple and interpretable model of grokking modular arithmetic tasks** [[openreview]](https://openreview.net/forum?id=0ZUKLCxwBo)
* **Understanding Addition in Transformers** [[ICLR 2024]](https://openreview.net/forum?id=rIx1YXVWZb)
* **Increasing Trust in Language Models through the Reuse of Verified Circuits** [[arxiv 2402]](http://arxiv.org/abs/2402.02619)
* **Pre-trained Large Language Models Use Fourier Features to Compute Addition** [[arxiv 2406]](https://arxiv.org/abs/2406.03445)

##### In-context Learning

* 🌟**In-context learning and induction heads** [[Transformer Circuits Thread]](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
* **In-Context Learning Creates Task Vectors** [[EMNLP 2023 Findings]](https://aclanthology.org/2023.findings-emnlp.624)
* **Label Words are Anchors: An Information Flow Perspective for Understanding In-Context Learning** [[EMNLP 2023]](https://aclanthology.org/2023.emnlp-main.609)
  * EMNLP 2023 best paper
* **LLMs Represent Contextual Tasks as Compact Function Vectors** [[ICLR 2024]](https://openreview.net/forum?id=AwyxtyMwaG)
* **Understanding In-Context Learning in Transformers and LLMs by Learning to Learn Discrete Functions** [[ICLR 2024]](https://openreview.net/forum?id=ekeyCgeRfC)
* **Where Does In-context Machine Translation Happen in Large Language Models?** [[openreview]](https://openreview.net/forum?id=3i7iNGxw6r)
* **In-Context Learning in Large Language Models: A Neuroscience-inspired Analysis of Representations** [[openreview]](https://openreview.net/forum?id=UEdS2lIgfY)
* **Analyzing Task-Encoding Tokens in Large Language Models** [[arxiv 2401]](http://arxiv.org/abs/2401.11323)
* **How do Large Language Models Learn In-Context? Query and Key Matrices of In-Context Heads are Two Towers for Metric Learning** [[arxiv 2402]](http://arxiv.org/abs/2402.02872)
* **Parallel Structures in Pre-training Data Yield In-Context Learning** [[arxiv 2402]](http://arxiv.org/abs/2402.12530)
* **What needs to go right for an induction head? A mechanistic study of in-context learning circuits and their formation** [[arxiv 2404]](http://arxiv.org/abs/2404.07129)
* **Task Diversity Shortens the ICL Plateau** [[arxiv 2410]](https://arxiv.org/abs/2410.05448)
* **Inference and Verbalization Functions During In-Context Learning** [[arxiv 2410]](https://arxiv.org/abs/2410.09349)

##### Factual Knowledge

* 🌟**Dissecting Recall of Factual Associations in Auto-Regressive Language Models** [[EMNLP 2023]](https://aclanthology.org/2023.emnlp-main.751)
* **Characterizing Mechanisms for Factual Recall in Language Models** [[EMNLP 2023]](https://aclanthology.org/2023.emnlp-main.615/)
* **Summing Up the Facts: Additive Mechanisms behind Factual Recall in LLMs** [[openreview]](https://openreview.net/forum?id=P2gnDEHGu3)
* **A Mechanism for Solving Relational Tasks in Transformer Language Models** [[openreview]](https://openreview.net/forum?id=ZmzLrl8nTa)
* **Overthinking the Truth: Understanding how Language Models Process False Demonstrations** [[ICLR 2024 spotlight]](https://openreview.net/forum?id=Tigr1kMDZy)
* 🌟**Fact Finding: Attempting to Reverse-Engineer Factual Recall on the Neuron Level** [[AlignmentForum 2312]](https://www.alignmentforum.org/s/hpWHhjvjn67LJ4xXX/p/iGuwZTHWb6DFY3sKB)
* **Cutting Off the Head Ends the Conflict: A Mechanism for Interpreting and Mitigating Knowledge Conflicts in Language Models** [[arxiv 2402]](https://arxiv.org/abs/2402.18154)
* **Competition of Mechanisms: Tracing How Language Models Handle Facts and Counterfactuals** [[arxiv 2402]](http://arxiv.org/abs/2402.11655)
* **A Glitch in the Matrix? Locating and Detecting Language Model Grounding with Fakepedia** [[arxiv 2403]](http://arxiv.org/abs/2312.02073)
* **Mechanisms of non-factual hallucinations in language models** [[arxiv 2403]](https://arxiv.org/abs/2403.18167)
* **Interpreting Key Mechanisms of Factual Recall in Transformer-Based Language Models** [[arxiv 2403]](arXiv:2403.19521)
* **Locating and Editing Factual Associations in Mamba** [[arxiv 2404]](arXiv:2404.03646)
* **Probing Language Models on Their Knowledge Source** [[arxiv 2410]](https://arxiv.org/abs/2410.05817}

##### Multilingual/Crosslingual

* **Do Llamas Work in English? On the Latent Language of Multilingual Transformers** [[arxiv 2402]](http://arxiv.org/abs/2402.10588)
* **Language-Specific Neurons: The Key to Multilingual Capabilities in Large Language Models** [[arxiv 2402]](http://arxiv.org/abs/2402.16438)
* **How do Large Language Models Handle Multilingualism?** [[arxiv 2402]](https://arxiv.org/abs/2402.18815)
* **Large Language Models are Parallel Multilingual Learners** [[arxiv 2403]](https://arxiv.org/abs/2403.09073)
* **Understanding the role of FFNs in driving multilingual behaviour in LLMs** [[arxiv 2404]](http://arxiv.org/abs/2404.13855)
* **How do Llamas process multilingual text? A latent exploration through activation patching** [[ICML 2024 MI Workshop]](https://openreview.net/forum?id=0ku2hIm4BS)
* **Concept Space Alignment in Multilingual LLMs** [[EMNLP 2024]](https://arxiv.org/abs/2410.01079)
* **On the Similarity of Circuits across Languages: a Case Study on the Subject-verb Agreement Task** [[EMNLP 2024 Findings]](https://arxiv.org/abs/2410.06496)

##### Multimodal

* **Interpreting CLIP's Image Representation via Text-Based Decomposition** [[ICLR 2024 oral]](https://openreview.net/forum?id=5Ca9sSzuDp)
* **Interpreting CLIP with Sparse Linear Concept Embeddings (SpLiCE)** [[NIPS 2024]](https://arxiv.org/abs/2402.10376)
* **Diffusion Lens: Interpreting Text Encoders in Text-to-Image Pipelines** [[arxiv 2403]](https://arxiv.org/abs/2403.05846)
* **The First to Know: How Token Distributions Reveal Hidden Knowledge in Large Vision-Language Models?** [[arxiv 2403]](https://arxiv.org/abs/2403.09037)
* **Understanding Information Storage and Transfer in Multi-modal Large Language Models** [[arxiv 2406]](https://arxiv.org/abs/2406.04236)
* **Towards Interpreting Visual Information Processing in Vision-Language Models** [[arxiv 2410]](https://arxiv.org/abs/2410.07149)
* **Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models** [[arxiv 2410]](https://arxiv.org/abs/2410.12662)
* **Pixology: Probing the Linguistic and Visual Capabilities of Pixel-based Language Models** [[arxiv 2410]](https://arxiv.org/abs/2410.12011)

#### Component

##### General

* **The Hydra Effect: Emergent Self-repair in Language Model Computations** [[arxiv 2307]](https://arxiv.org/abs/2307.15771)
* **Unveiling A Core Linguistic Region in Large Language Models** [[arxiv 2310]](http://arxiv.org/abs/2310.14928)
* **Exploring the Residual Stream of Transformers** [[arxiv 2312]](http://arxiv.org/abs/2312.12141)
* **Characterizing Large Language Model Geometry Solves Toxicity Detection and Generation** [[arxiv 2312]](https://arxiv.org/abs/2312.01648)
* **Explorations of Self-Repair in Language Models** [[arxiv 2402]](http://arxiv.org/abs/2402.15390)
* **Massive Activations in Large Language Models** [[arxiv 2402]](https://arxiv.org/abs/2402.17762)
* **Interpreting Context Look-ups in Transformers: Investigating Attention-MLP Interactions** [[arxiv 2402]](https://arxiv.org/abs/2402.15055)
* **Fantastic Semantics and Where to Find Them: Investigating Which Layers of Generative LLMs Reflect Lexical Semantics** [[arxiv 2403]](https://arxiv.org/abs/2403.01509)
* **The Heuristic Core: Understanding Subnetwork Generalization in Pretrained Language Models** [[arxiv 2403]](http://arxiv.org/abs/2403.03942)
* **Localizing Paragraph Memorization in Language Models** [[github 2403]](http://arxiv.org/abs/2403.19851)

##### Attention

* 🌟**Awesome-Attention-Heads** [[github]](https://github.com/IAAR-Shanghai/Awesome-Attention-Heads)
  * A carefully compiled list that summarizes the diverse functions of the attention heads.

* 🌟**In-context learning and induction heads** [[Transformer Circuits Thread]](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
* **On the Expressivity Role of LayerNorm in Transformers' Attention** [[ACL 2023 Findings]](https://aclanthology.org/2023.findings-acl.895.pdf)
* **On the Role of Attention in Prompt-tuning** [[ICML 2023]](https://openreview.net/forum?id=qorOnDor89)
* **Copy Suppression: Comprehensively Understanding an Attention Head** [[ICLR 2024]](https://openreview.net/forum?id=g8oaZRhDcf)
* **Successor Heads: Recurring, Interpretable Attention Heads In The Wild** [[ICLR 2024]](https://openreview.net/forum?id=kvcbV8KQsi)
* **A phase transition between positional and semantic learning in a solvable model of dot-product attention** [[arxiv 2024]](http://arxiv.org/abs/2402.03902)
* **Retrieval Head Mechanistically Explains Long-Context Factuality** [[arxiv 2404]](http://arxiv.org/abs/2404.15574)
* **Iteration Head: A Mechanistic Study of Chain-of-Thought** [[arxiv 2406]](https://arxiv.org/abs/2406.02128)
* **When Attention Sink Emerges in Language Models: An Empirical View** [[arxiv 2410]](https://arxiv.org/abs/2410.10781)

##### MLP/FFN

* 🌟**Transformer Feed-Forward Layers Are Key-Value Memories** [[EMNLP 2021]](https://aclanthology.org/2021.emnlp-main.446)
* **Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space** [[EMNLP 2022]](https://aclanthology.org/2022.emnlp-main.3)
* **What does GPT store in its MLP weights? A case study of long-range dependencies** [[openreview]](https://openreview.net/forum?id=nUGFpDCu3W)
* **Understanding the role of FFNs in driving multilingual behaviour in LLMs** [[arxiv 2404]](http://arxiv.org/abs/2404.13855)

##### Neuron

* 🌟**Toy Models of Superposition** [[Transformer Circuits Thread]](https://transformer-circuits.pub/2022/toy_model/index.html)
* **Knowledge Neurons in Pretrained Transformers** [[ACL 2022]](https://aclanthology.org/2022.acl-long.581)
* **Polysemanticity and Capacity in Neural Networks** [[arxiv 2210]](http://arxiv.org/abs/2210.01892)
* 🌟**Finding Neurons in a Haystack: Case Studies with Sparse Probing** [[TMLR 2023]](https://openreview.net/forum?id=JYs1R9IMJr)
* **DEPN: Detecting and Editing Privacy Neurons in Pretrained Language Models** [[EMNLP 2023]](https://aclanthology.org/2023.emnlp-main.174)
* **Neurons in Large Language Models: Dead, N-gram, Positional** [[arxiv 2309]](http://arxiv.org/abs/2309.04827)
* **Universal Neurons in GPT2 Language Models** [[arxiv 2401]](http://arxiv.org/abs/2401.12181)
* **Language-Specific Neurons: The Key to Multilingual Capabilities in Large Language Models** [[arxiv 2402]](http://arxiv.org/abs/2402.16438)
* **How do Large Language Models Handle Multilingualism?** [[arxiv 2402]](https://arxiv.org/abs/2402.18815)
* **PURE: Turning Polysemantic Neurons Into Pure Features by Identifying Relevant Circuits** [[arxiv 2404]](http://arxiv.org/abs/2404.06453)

#### Learning Dynamics

##### General

* **JoMA: Demystifying Multilayer Transformers via JOint Dynamics of MLP and Attention** [[ICLR 2024]](https://openreview.net/forum?id=LbJqRGNYCf)
* **Learning Associative Memories with Gradient Descent** [[arxiv 2402]](https://arxiv.org/abs/2402.18724)
* **Mechanics of Next Token Prediction with Self-Attention** [[arxiv 2402]](http://arxiv.org/abs/2403.08081)
* **The Garden of Forking Paths: Observing Dynamic Parameters Distribution in Large Language Models** [[arxiv 2403]](http://arxiv.org/abs/2403.08739)
* **LLM Circuit Analyses Are Consistent Across Training and Scale** [[ICML 2024 MI Workshop]](https://openreview.net/forum?id=1WeLXvaNJP)
* **Geometric Signatures of Compositionality Across a Language Model's Lifetime** [[arxiv 2410]](https://arxiv.org/abs/2410.01444)

##### Phase Transition/Grokking

* 🌟**Progress measures for grokking via mechanistic interpretability** [[ICLR 2023]](https://openreview.net/forum?id=9XFSbDPmdW)
* **A Toy Model of Universality: Reverse Engineering How Networks Learn Group Operations** [[ICML 2023]](https://openreview.net/forum?id=jCOrkuUpss)
* 🌟**The Mechanistic Basis of Data Dependence and Abrupt Learning in an In-Context Classification Task** [[ICLR 2024 oral]](https://openreview.net/forum?id=aN4Jf6Cx69)
  * Highest scores at ICLR 2024: 10, 10, 8, 8. And by one author only!
* **Sudden Drops in the Loss: Syntax Acquisition, Phase Transitions, and Simplicity Bias in MLMs** [[ICLR 2024 spotlight]](https://openreview.net/forum?id=MO5PiKHELW)
* **A simple and interpretable model of grokking modular arithmetic tasks** [[openreview]](https://openreview.net/forum?id=0ZUKLCxwBo)
* **Unified View of Grokking, Double Descent and Emergent Abilities: A Perspective from Circuits Competition** [[arxiv 2402]](http://arxiv.org/abs/2402.15175)
* **Interpreting Grokked Transformers in Complex Modular Arithmetic** [[arxiv 2402]](https://arxiv.org/abs/2402.16726)
* **Towards Tracing Trustworthiness Dynamics: Revisiting Pre-training Period of Large Language Models** [[arxiv 2402]](https://arxiv.org/abs/2402.19465)
* **Learning to grok: Emergence of in-context learning and skill composition in modular arithmetic tasks** [[arxiv 2406]](https://arxiv.org/abs/2406.02550)
* **Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization** [[ICML 2024 MI Workshop]](https://openreview.net/forum?id=ns8IH5Sn5y)

##### Fine-tuning

* **Studying Large Language Model Generalization with Influence Functions** [[arxiv 2308]](http://arxiv.org/abs/2308.03296)
* **Mechanistically analyzing the effects of fine-tuning on procedurally defined tasks** [[ICLR 2024]](https://openreview.net/forum?id=A0HKeKl4Nl)
* **Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking** [[ICLR 2024]](https://openreview.net/forum?id=8sKcAWOf2D)
* **The Hidden Space of Transformer Language Adapters** [[arxiv 2402]](http://arxiv.org/abs/2402.13137)
* **Dissecting Fine-Tuning Unlearning in Large Language Models** [[EMNLP 2024]](https://arxiv.org/abs/2410.06606)

#### Feature Representation/Probing-based

##### General

* **Implicit Representations of Meaning in Neural Language Models** [[ACL 2021]](https://aclanthology.org/2021.acl-long.143)
* **All Roads Lead to Rome? Exploring the Invariance of Transformers' Representations** [[arxiv 2305]](http://arxiv.org/abs/2305.14555)
* **Observable Propagation: Uncovering Feature Vectors in Transformers** [[openreview]](https://openreview.net/forum?id=sNWQUTkDmA)
* **In-Context Learning in Large Language Models: A Neuroscience-inspired Analysis of Representations** [[openreview]](https://openreview.net/forum?id=UEdS2lIgfY)
* **Challenges with unsupervised LLM knowledge discovery** [[arxiv 2312]](https://arxiv.org/abs/2312.10029)
* **Still No Lie Detector for Language Models: Probing Empirical and Conceptual Roadblocks** [[arxiv 2307]](http://arxiv.org/abs/2307.00175)
* **Position Paper: Toward New Frameworks for Studying Model Representations** [[arxiv 2402]](http://arxiv.org/abs/2402.03855)
* **How Large Language Models Encode Context Knowledge? A Layer-Wise Probing Study** [[arxiv 2402]](http://arxiv.org/abs/2402.16061)
* **More than Correlation: Do Large Language Models Learn Causal Representations of Space** [[arxiv 2312]](https://arxiv.org/abs/2312.16257)
* **Do Large Language Models Mirror Cognitive Language Processing?** [[arxiv 2402]](https://arxiv.org/abs/2402.18023)
* **On the Scaling Laws of Geographical Representation in Language Models** [[arxiv 2402]](https://arxiv.org/abs/2402.19406)
* **Monotonic Representation of Numeric Properties in Language Models** [[arxiv 2403]](http://arxiv.org/abs/2403.10381)
* **Exploring Concept Depth: How Large Language Models Acquire Knowledge at Different Layers?** [[arxiv 2404]](http://arxiv.org/abs/2404.07066)
* **Simple probes can catch sleeper agents** [[Anthropic Blog]](https://www.anthropic.com/research/probes-catch-sleeper-agents)
* **PaCE: Parsimonious Concept Engineering for Large Language Models** [[arxiv 2406]](https://arxiv.org/abs/2406.04331)
* **The Geometry of Categorical and Hierarchical Concepts in Large Language Models** [[ICML 2024 MI Workshop]](https://openreview.net/forum?id=KXuYjuBzKo)
* **Concept Space Alignment in Multilingual LLMs** [[EMNLP 2024]](https://arxiv.org/abs/2410.01079)
* **Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language Models** [[arxiv 2410]](https://arxiv.org/abs/2410.06981)

##### Linearity

* 🌟**Actually, Othello-GPT Has A Linear Emergent World Representation** [[Neel Nanda's blog]](https://www.neelnanda.io/mechanistic-interpretability/othello)
* **Language Models Linearly Represent Sentiment** [[openreview]](https://openreview.net/forum?id=iGDWZFc7Ya)
* **Language Models Represent Space and Time** [[openreview]](https://openreview.net/forum?id=jE8xbmvFin)
* **The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets** [[openreview]](https://openreview.net/forum?id=CeJEfNKstt)
* **Linearity of Relation Decoding in Transformer Language Models** [[ICLR 2024]](https://openreview.net/forum?id=w7LU2s14kE)
* **The Linear Representation Hypothesis and the Geometry of Large Language Models** [[arxiv 2311]](https://arxiv.org/abs/2311.03658)
* **Language Models Represent Beliefs of Self and Others** [[arxiv 2402]](https://arxiv.org/abs/2402.18496)
* **On the Origins of Linear Representations in Large Language Models** [[arxiv 2403]](http://arxiv.org/abs/2403.03867)
* **Refusal in LLMs is mediated by a single direction** [[Lesswrong 2024]](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction)

#### Application

##### Training
* **Aligning Large Language Models with Human Preferences through Representation Engineering** [[arxiv2312]](http://arxiv.org/abs/2312.15997)
* **ReFT: Representation Finetuning for Language Models** [[arxiv 2404]](https://arxiv.org/abs/2404.03592) [[github]](https://github.com/stanfordnlp/pyreft)
* **Direct Preference Optimization Using Sparse Feature-Level Constraints** [[arxiv2411]](https://arxiv.org/abs/2411.07618)
* **LLM Pretraining with Continuous Concepts** [[arxiv2502]](https://arxiv.org/abs/2502.08524)


##### Inference-Time Intervention/Activation Steering

* 🌟**Inference-Time Intervention: Eliciting Truthful Answers from a Language Model** [[NIPS 2023]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/81b8390039b7302c909cb769f8b6cd93-Abstract-Conference.html) [[github]](https://github.com/likenneth/honest_llama)
* **Activation Addition: Steering Language Models Without Optimization** [[arxiv 2308]](http://arxiv.org/abs/2308.10248)
* **Self-Detoxifying Language Models via Toxification Reversal** [[EMNLP 2023]](https://aclanthology.org/2023.emnlp-main.269)
* **DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models** [[arxiv 2309]](https://arxiv.org/abs/2309.03883)
* **In-context Vectors: Making In Context Learning More Effective and Controllable Through Latent Space Steering** [[arxiv 2311]](http://arxiv.org/abs/2311.06668)
* **Steering Llama 2 via Contrastive Activation Addition** [[arxiv 2312]](http://arxiv.org/abs/2312.06681)
* **A Language Model's Guide Through Latent Space** [[arxiv 2402]](http://arxiv.org/abs/2402.14433)
* **Backdoor Activation Attack: Attack Large Language Models using Activation Steering for Safety-Alignment** [[arxiv 2311]](https://arxiv.org/abs/2311.09433)
* **Extending Activation Steering to Broad Skills and Multiple Behaviours** [[arxiv 2403]](https://arxiv.org/abs/2403.05767)
* **Spectral Editing of Activations for Large Language Model Alignment** [[arxiv 2405]](http://arxiv.org/abs/2405.09719)
* **Controlling Large Language Model Agents with Entropic Activation Steering** [[arxiv 2406]](https://arxiv.org/abs/2406.00244)
* **Analyzing the Generalization and Reliability of Steering Vectors** [[ICML 2024 MI Workshop]](https://openreview.net/forum?id=akCsMk4dDL)
* **Towards Inference-time Category-wise Safety Steering for Large Language Models** [[arxiv 2410]](https://arxiv.org/abs/2410.01174)
* **A Timeline and Analysis for Representation Plasticity in Large Language Models** [[arxiv 2410]](https://arxiv.org/abs/2410.06225)
* **Semantics-Adaptive Activation Intervention for LLMs via Dynamic Steering Vectors** [[arxiv 2410]](https://arxiv.org/abs/2410.12299)

##### Knowledge/Model Editing

* **Locating and Editing Factual Associations in GPT** (*ROME*) [[NIPS 2022]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html) [[github]](https://github.com/kmeng01/rome)
* **Memory-Based Model Editing at Scale** [[ICML 2022]](https://proceedings.mlr.press/v162/mitchell22a.html)
* **Editing models with task arithmetic** [[ICLR 2023]](https://openreview.net/forum?id=6t0Kwf8-jrj)
* **Mass-Editing Memory in a Transformer** [[ICLR 2023]](https://openreview.net/forum?id=MkbcAHIYgyS)
* **Detecting Edit Failures In Large Language Models: An Improved Specificity Benchmark** [[ACL 2023 Findings]](https://aclanthology.org/2023.findings-acl.733)
* **Can LMs Learn New Entities from Descriptions? Challenges in Propagating Injected Knowledge** [[ACL 2023]](https://aclanthology.org/2023.acl-long.300)
* **Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models** [[NIPS 2023]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3927bbdcf0e8d1fa8aa23c26f358a281-Abstract-Conference.html)
* **Inspecting and Editing Knowledge Representations in Language Models** [[arxiv 2304]](http://arxiv.org/abs/2304.00740) [[github]](https://github.com/evandez/REMEDI)
* **Methods for Measuring, Updating, and Visualizing Factual Beliefs in Language Models** [[EACL 2023]](https://aclanthology.org/2023.eacl-main.199)
* **Editing Common Sense in Transformers** [[EMNLP 2023]](https://aclanthology.org/2023.emnlp-main.511)
* **DEPN: Detecting and Editing Privacy Neurons in Pretrained Language Models** [[EMNLP 2023]](https://aclanthology.org/2023.emnlp-main.174)
* **MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions** [[EMNLP 2023]](https://aclanthology.org/2023.emnlp-main.971)
* **PMET: Precise Model Editing in a Transformer** [[arxiv 2308]](http://arxiv.org/abs/2308.08742)
* **Untying the Reversal Curse via Bidirectional Language Model Editing** [[arxiv 2310]](http://arxiv.org/abs/2310.10322)
* **Unveiling the Pitfalls of Knowledge Editing for Large Language Models** [[ICLR 2024]](https://openreview.net/forum?id=fNktD3ib16)
* **A Comprehensive Study of Knowledge Editing for Large Language Models** [[arxiv 2401]](http://arxiv.org/abs/2401.01286)
* **Trace and Edit Relation Associations in GPT** [[arxiv 2401]](http://arxiv.org/abs/2401.02976)
* **Model Editing with Canonical Examples** [[arxiv 2402]](https://arxiv.org/abs/2402.06155)
* **Updating Language Models with Unstructured Facts: Towards Practical Knowledge Editing** [[arxiv 2402]](http://arxiv.org/abs/2402.18909)
* **Editing Conceptual Knowledge for Large Language Models** [[arxiv 2403]](https://arxiv.org/abs/2403.06259)
* **Editing the Mind of Giants: An In-Depth Exploration of Pitfalls of Knowledge Editing in Large Language Models** [[arxiv 2406]](https://arxiv.org/abs/2406.01436)
* **Locate-then-edit for Multi-hop Factual Recall under Knowledge Editing** [[arxiv 2410]](https://arxiv.org/abs/2410.06331)
* **Keys to Robust Edits: from Theoretical Insights to Practical Advances** [[arxiv 2410]](https://arxiv.org/abs/2410.09338)

##### Hallucination

* **The Internal State of an LLM Knows When It's Lying** [[EMNLP 2023 Findings]](https://arxiv.org/abs/2304.13734)
* **Do Androids Know They're Only Dreaming of Electric Sheep?** [[arxiv 2312]](https://arxiv.org/abs/2312.17249)
* **INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection** [[ICLR 2024]](https://openreview.net/forum?id=Zj12nzlQbz)
* **TruthX: Alleviating Hallucinations by Editing Large Language Models in Truthful Space** [[arxiv 2402]](https://arxiv.org/abs/2402.17811)
* **Characterizing Truthfulness in Large Language Model Generations with Local Intrinsic Dimension** [[arxiv 2402]](https://arxiv.org/abs/2402.18048)
* **Whispers that Shake Foundations: Analyzing and Mitigating False Premise Hallucinations in Large Language Models** [[arxiv 2402]](https://arxiv.org/abs/2402.19103)
* **In-Context Sharpness as Alerts: An Inner Representation Perspective for Hallucination Mitigation** [[arxiv 2403]](http://arxiv.org/abs/2403.01548)
* **Unsupervised Real-Time Hallucination Detection based on the Internal States of Large Language Models** [[arxiv 2403]](https://arxiv.org/abs/2403.06448)
* **Adaptive Activation Steering: A Tuning-Free LLM Truthfulness Improvement Method for Diverse Hallucinations Categories** [[arxiv 2406]](https://arxiv.org/abs/2406.00034)

##### Pruning/Redundancy Analysis

* **Not all Layers of LLMs are Necessary during Inference** [[arxiv 2403]](http://arxiv.org/abs/2403.02181)
* **ShortGPT: Layers in Large Language Models are More Redundant Than You Expect** [[arxiv 2403]](http://arxiv.org/abs/2403.03853)
* **The Unreasonable Ineffectiveness of the Deeper Layers** [[arxiv 2403]](http://arxiv.org/abs/2403.17887)
* **The Remarkable Robustness of LLMs: Stages of Inference?** [[ICML 2024 MI Workshop]](https://openreview.net/forum?id=R5unwb9KPc)
