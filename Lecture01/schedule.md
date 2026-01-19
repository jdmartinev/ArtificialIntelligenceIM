| Wk | Session (3 h) | Highlights & Hands-on (↯ = lab/demo) | Reading / Prep |
|----|---------------|--------------------------------------|----------------|
| **Module 1 – Foundations of AI & Deep-Learning Math** ||||
| 1 | **Linear Models & Matrix Calculus Refresher** | Linear regression ⇒ ridge/LASSO<br>Vector-Jacobian products; automatic differentiation primer (single scalar y) ↯ PyTorch autograd warm-up | Boyd & Vandenberghe §2; Goodfellow 1.2 |
| 2 | **Loss Functions & Probabilistic Interpretation** | MSE ↔ Gaussian likelihood, cross-entropy ↔ Bernoulli/softmax<br>Bias–variance, maximum-likelihood ↯ Fit logistic regression on credit risk, derive gradient by hand | Murphy §8.1, 8.3 |
| 3 | **Back-propagation in Tensor Form** | Computational graph theory; chain rule in tensor notation<br>Vanishing/exploding gradients ↯ Derive backprop for 2-layer MLP on whiteboard, verify with PyTorch hooks | Goodfellow §6.5 |
| 4 | **Optimisation Algorithms** | GD, momentum, NAG, RMSProp, Adam, decoupled weight decay; learning-rate scheduling ↯ Compare optimisers on CIFAR-10 tiny subset, plot convergence | Ruder “Optimisation for DL” |
| **Module 2 – Computer Vision** ||||
| 5 | **Convolutional Neural Nets: From Correlation to Hierarchies** | Convolution math, receptive fields, padding, pooling; weight sharing efficiency ↯ Implement CNN forward pass from scratch (no libs) | Dumoulin & Visin “Conv Arithmetic” |
| 6 | **Modern Conv Architectures & Transfer Learning** | VGG→ResNet→EffNet; batch-norm, residuals; fine-tuning strategies ↯ Fine-tune ResNet-18 on EuroSAT; freeze vs. full-train comparison | He et al. (2016) |
| 7 | **Generative Models I – Autoencoders & VAEs** | Undercomplete vs. denoising; ELBO derivation; reparameterisation trick ↯ Train a β-VAE on MNIST, visualise latent traversal | Kingma & Welling (2014) |
| 8 | **Generative Models II – GANs & Diffusion** | Min–max game, JS/WD divergence; U-Net + DDPM math; score matching ↯ Use Diffusers to fine-tune class-conditional model; FID evaluation | Ho et al. (2020) |
| **Module 3 – NLP, Transformers & LLMs** ||||
| 9 | **Word Embeddings & Self-Attention** | Distributional hypothesis recap; word2vec/fastText → attention mechanism derivation ↯ Build scaled-dot-attention layer from scratch | Vaswani §2–3 |
| 10 | **Transformer Encoder & Decoder Math** | Positional encodings (sinusoidal + RoPE); multi-head; causal masking ↯ Train tiny transformer on character-level language modelling | Liu et al. RoPE |
| 11 | **LLM Scaling & Fine-Tuning** | Scaling laws; LoRA/PEFT, adapters, bits-and-bytes quantisation ↯ LoRA-adapt Llama-3-8 B (HF) on domain Q&A; measure perplexity & cost | Kaplan et al.; Hu et al. LoRA |
| 12 | **Evaluation, Safety & Alignment** | Zero-shot vs. RLHF/DPO; toxicity metrics; jailbreak taxonomy ↯ Prompt-attack a chat model, then add safety-layer rule-based guard | Anthropic “Constitutional AI” |
| **Module 4 – Modern AI Apps & Engineering** ||||
| 13 | **Generative AI APIs & Prompt Engineering** | Few/zero-shot patterns, chain-of-thought, function calling ↯ Build RAG FAQ bot with OpenAI API + FAISS | OpenAI Cookbook |
| 14 | **LLM Agents & Tool Use** | ReAct, function‐calling, reflexion; planning vs. chaining ↯ LangChain tools agent (calculator + web search) | Yao et al. ReAct |
| 15 | **MLOps & Deployment at Scale** | Model registry, MLflow, DVC; Docker/Kubernetes; ONNX, Torch-TensorRT ↯ Containerise ResNet service; CI test & Helm chart | Chip Huyen Ch. 11 |
| 16 | **Capstone Showcase & Futures** | Student demos (10 min each) + peer critique; panel on AI research frontiers; final oral quiz | Capstone repo & slide deck due |
