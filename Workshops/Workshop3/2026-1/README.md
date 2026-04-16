# Workshop: Internals of a LLaMA-like LLM

## Overview

This workshop focuses on understanding the internal components of a **LLaMA-style Large Language Model (LLM)** through hands-on implementation.

Students will progressively build and analyze key components of a transformer-based architecture, with emphasis on:

- Masked attention
- Transformer blocks
- Forward pass mechanics
- Model structure and data flow

The goal is not just to use models, but to **understand how they work internally**.

---

## Workshop Structure

The workshop is divided into **4 notebooks**, each corresponding to a specific stage:

### `exc1.ipynb` — Core Components
- Introduction to the architecture
- Implementation of basic building blocks
- Understanding tensor shapes and flow

### `exc2.ipynb` — Attention Mechanism
- Implementation of **masked self-attention**
- Exploration of causal masking
- Debugging attention outputs

### `exc3.ipynb` — Transformer Block
- Integration of:
  - Attention
  - Feedforward layers
  - Residual connections
- Understanding normalization and stability

### `exc4.ipynb` — Full Model Assembly
- Putting everything together into a **mini LLaMA-like model**
- Forward pass through the complete architecture
- Visualization and testing

---
## Instructions

Each notebook includes sections marked with `TODO`. Your task is to complete these sections so that the model components work as expected.

For each exercise:

- Implement the missing code  
- Run the notebook end-to-end  
- Verify outputs (shapes, values, behavior)  
- Fix any inconsistencies  

Avoid copying solutions — the goal is to understand the mechanics of the model.

---
## What You Should Achieve

By completing this workshop, you should be able to:

- Implement masked self-attention from scratch  
- Explain how causal masking affects token dependencies  
- Build a full transformer block (attention + MLP + residuals)  
- Understand how data flows through a LLaMA-like architecture  
- Debug tensor shape issues in transformer models  

---
## Key Things to Watch

These are the most common failure points:

- Incorrect tensor shapes (especially in attention)  
- Wrong masking (future tokens must not be visible)  
- Missing residual connections  
- Misuse of normalization layers  

When something breaks, inspect tensor shapes using print statements such as: print(tensor.shape)

---
## Recommended Workflow

Work sequentially:

1. exc1.ipynb  
2. exc2.ipynb  
3. exc3.ipynb  
4. exc4.ipynb  

At each step:

- Complete all TODOs before moving on  
- Make sure you understand the output before continuing  
- Do small tests instead of running everything blindly  

---
## Evaluation Criteria

Your work can be assessed based on:

- Correct implementation of components  
- Code readability and structure  
- Proper handling of tensor operations  
- Understanding of attention and transformer logic  

---
## Extensions (Optional)

If you finish early, try:

- Visualizing attention weights  
- Modifying the masking strategy  
- Changing the number of heads or embedding size  
- Introducing dropout or regularization  

---
## Notes

This workshop assumes:

- Basic PyTorch knowledge  
- Familiarity with neural networks  
- Introductory understanding of transformers  

If something feels unclear, revisit the theory before proceeding
