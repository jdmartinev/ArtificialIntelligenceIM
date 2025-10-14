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
