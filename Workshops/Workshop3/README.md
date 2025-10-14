# Workshop: Teaching Gemma 3 (1B) to Reason with GRPO

In this hands-on workshop, participants will explore how **reinforcement learning from verifiable feedback** can improve reasoning in large language models.  
We will fine-tune **Gemma 3 (1B)** using **Group Relative Policy Optimization (GRPO)** on a multiple-choice **reasoning question-answering dataset**.  
The goal is to teach the model not just to guess answers, but to **reason step-by-step** before selecting the correct one.

---

## 🎯 Learning Objectives
- Understand the **GRPO** algorithm as an alternative to PPO in RLHF workflows.  
- Learn how to design **verifiable rewards** for reasoning tasks (e.g., auto-checking the final answer).  
- Generate, evaluate, and score model outputs using a **reward function**.  
- Fine-tune a Gemma model on a **QA dataset** (e.g., RACE or QuALITY) to improve reasoning accuracy.  
- Measure and visualize improvements in the model’s performance.

---

## 🧩 Workshop Outline
1. **Introduction to GRPO and reasoning tasks**  
   - Motivation for reward-based fine-tuning  
   - Key differences between PPO and GRPO  
2. **Dataset preparation**  
   - Using multiple-choice QA datasets with verifiable answers  
   - Building prompts that encourage reasoning (“Think step-by-step…”)  
3. **Reward modeling**  
   - Implementing and testing a simple reward function (`mc_reward`)  
   - (Optional) Training a small neural reward model from labeled data  
4. **Fine-tuning with GRPO**  
   - Loading `unsloth/gemma-3-1b-it`  
   - Running a short GRPO training loop  
5. **Evaluation and discussion**  
   - Comparing pre- and post-training accuracy  
   - Observing how reasoning chains improve answer quality  

---

## ⚙️ Tools and Frameworks
- 🧩 **Unsloth** for efficient Gemma loading and fine-tuning  
- 🤗 **Transformers** and **TRL** for GRPO training  
- 📚 **Datasets**: RACE / QuALITY for reasoning QA  
- 🧮 **Torch / Accelerate** for GPU execution  

---

## 🚀 Expected Outcomes
By the end of the workshop, you will:
- Know how to **implement a GRPO fine-tuning loop**.  
- Understand how to **quantify reasoning ability** with automatic rewards.  
- Gain hands-on experience improving a **Gemma 3 (1B)** model’s reasoning on QA tasks.

## 🧭 Workshop Instructions and Deliverables

This workshop is divided into three main stages.  
You will progressively move from **understanding and evaluating** the base Gemma model, to **fine-tuning it with GRPO**, and finally **comparing and analyzing** your results.

---

### 🧩 1. Dataset Exploration and Base Model Evaluation 

Follow the notebook [01_dataset_evaluation](notebooks/01_dataset_evaluation.ipynb)

**Objective:**  
Understand the dataset and evaluate the reasoning ability of the **base Gemma 3 (1B)** model.

**Instructions:**
1. Load the QA dataset (e.g., **RACE**) and inspect a few examples.  
   - Identify the structure: passage, question, multiple-choice options, correct answer.  
2. Load the base model (`unsloth/gemma-3-1b-it`) and evaluate its accuracy on a subset of the test set.  
3. Experiment with **prompt engineering**:
   - Try simple Q&A prompts.
   - Try *chain-of-thought* style prompts (“Think step-by-step before answering”).  
4. Record how the model’s reasoning or accuracy changes with different prompts.

**Deliverables:**
- ✅ A brief dataset description (what type of reasoning it tests).  
- ✅ At least **two different prompts** evaluated on the dataset.  
- ✅ Computed **accuracy or reward-based score** for the base model.  
- ✅ Short analysis: how prompt design affects reasoning quality.  
- 📊 *(Optional)* Simple comparison table or plot (accuracy per prompt).

---

### 🧠 2. Fine-Tuning with GRPO 

Adjust the notebook [02_fine_tuning_grpo](02_fine_tuning_grpo.ipynb) for this dataset.

**Objective:**  
Fine-tune the base model using **Group Relative Policy Optimization (GRPO)** and a custom **reward model** to enhance reasoning.

**Instructions:**
1. Define your **reward function** — for example:
   - A rule-based reward (`mc_reward`) that checks if the final answer letter matches the gold label.
   - *(Optional)* a **learned reward model** trained from labeled pairs.  
2. Configure the GRPO trainer and train the model for a few epochs.  
3. Save and **push your fine-tuned model** to the Hugging Face Hub under your namespace.  
4. Document the main hyperparameters and training setup.

**Deliverables:**
- ✅ Working **reward function** and explanation of its design.  
- ✅ Functional **GRPO training loop** (logs showing learning progress).  
- ✅ **Model uploaded** to the Hugging Face Hub with a clear name and short model card.  
- 📈 Short reflection or plot showing how the model’s rewards or loss evolved during training.  
- 🧾 Link or screenshot of your model on the Hub.

---

### ⚖️ 3. Evaluation of the Fine-Tuned Model 

Create a notebook `03_comparison_evaluation.ipynb` to evaluate the fine-tuned model-

**Objective:**  
Compare the reasoning performance of the **fine-tuned model** versus the **base model**.

**Instructions:**
1. Load your fine-tuned model from the Hugging Face Hub.  
2. Re-run the evaluation on the same test subset using the same prompts.  
3. Compare results between:
   - Base model  
   - Fine-tuned (GRPO) model  
4. Analyze whether reasoning quality or accuracy improved.

**Deliverables:**
- ✅ Comparison table with **base vs fine-tuned accuracy**.  
- ✅ At least **three example questions** showing reasoning improvements (or failures).  
- 📊 Visualization of results (bar chart or summary table).  
- 🧩 Short reflection: what did the model learn through GRPO?

---

### ⚙️ Optional but Highly Recommended

Once you have completed the workshop and fine-tuned your model, try **deploying it using LightningAI Studio**.  
This allows you to interact with your model through a user-friendly chat interface and share it easily.

🔗 **Follow this LightningAI Studio tutorial:**  
[Chat with DeepSeek-R1: An Advanced AI Reasoning Model (LightningAI)](https://lightning.ai/collectiveai/studios/chat-with-deepseek-r1-an-advanced-ai-reasoning-model?view=public&section=featured&query=deepseek&tab=files&path=cloudspaces%2F01jjgzsemxx1cs3fwdw0y6p2dd%2Fdeepseek-r1&y=2&x=1)

You can adapt this workflow to **deploy your fine-tuned Gemma 3 reasoning model**, experiment with prompts, and visualize reasoning outputs interactively.

---

### 🧾 4. Final Report (Markdown or PDF)

**Objective:**  
Summarize your experiment and communicate findings clearly.

**Deliverables:**
- Title, author(s), and link to your Hugging Face model.  
- Description of:
  - Dataset and prompt format  
  - Reward model design  
  - Training configuration  
  - Quantitative and qualitative results  
  - Key insights and limitations  
- *(Optional)* Ideas for improving reasoning further (e.g., curriculum training, hybrid rewards).

---

### ✅ Expected Outcome

By the end of this workshop, you will:
- Understand how **GRPO** can align language models through verifiable feedback.  
- Know how to define and implement a **reward function** for reasoning tasks.  
- Gain hands-on experience **fine-tuning and evaluating** a Gemma 3 model.  
- Be able to publish and share your own **fine-tuned reasoning model** on the Hugging Face Hub.

