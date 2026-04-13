# GRPO — Group Relative Policy Optimization
### Math & Tensor Shapes — Script 4

---

## The Problem GRPO Solves

PPO requires four models in memory simultaneously:

| Model | Role |
|-------|------|
| Policy $\pi_\theta$ | the LM being trained |
| Reference $\pi_\text{ref}$ | frozen SFT model for KL penalty |
| Critic $V_\phi$ | estimates $\mathbb{E}[G_t \mid s_t]$ per token |
| Reward model $R_\phi$ | scores completed responses |

The critic is the expensive one — it is another full LM-sized network. It exists solely to provide the per-token baseline $V_\phi(s_t)$ so that GAE can compute per-token advantages.

GRPO's question: **can we get a good baseline without training a critic at all?**

---

## The Core Idea — Group Sampling

Instead of generating **one** response per prompt and estimating the baseline with $V_\phi$, generate **G responses** per prompt and use their statistics directly.

For prompt $x$, sample:

$$y_1, y_2, \ldots, y_G \sim \pi_\theta(\cdot \mid x)$$

Compute rewards:

$$R_1, R_2, \ldots, R_G$$

The advantage of response $i$ is:

$$\boxed{A_i = \frac{R_i - \text{mean}(R_1, \ldots, R_G)}{\text{std}(R_1, \ldots, R_G)}}$$

No critic. No $G_t$. No $\delta_t$. No backwards loop. Just statistics across the group.

---

## Why This Works as a Baseline

Recall from script 1: any baseline $b$ with $\mathbb{E}\left[b \cdot \nabla_{\theta} \log \pi_{\theta}\right] = 0$ can be subtracted without introducing bias.

$\text{mean}(R_1, \ldots, R_G)$ satisfies this — it is an estimate of $\mathbb{E}_{\pi}[R \mid x]$, the expected reward for **this specific prompt**.

This is strictly better than the flat batch mean $\bar{R}$ used in script 1:

| Baseline | What it estimates | Prompt-specific? |
|----------|-------------------|-----------------|
| $\bar{R}$ (script 1) | average reward across all prompts | No |
| $V_\phi(s_t)$ (PPO) | expected return from state $s_t$ | Yes, per token |
| $\text{mean}(R_1..R_G)$ (GRPO) | expected reward for this prompt | Yes, per prompt |

If a prompt is easy (model already rewards it well), the group mean is high and $A_i$ is small — the gradient correctly signals "you're already good here." A global mean would wrongly signal "above average" for every response.

---

## What Standardization Does

Dividing by $\text{std}$ (the denominator in $A_i$) does two things:

**1. Makes advantages comparable across prompts.**
Some prompts have high-variance rewards (model is uncertain), others low-variance. Without std normalization, high-variance prompts would dominate the gradient. After normalization, every prompt contributes equally.

**2. Scales gradient steps.**
$A_i \in [-1, +1]$ roughly (before clipping), so gradient magnitudes are stable regardless of the absolute scale of rewards.

Edge case: if all G responses get the same reward, $\text{std} = 0$. We add $\epsilon = 10^{-8}$ to prevent division by zero. In this case all $A_i = 0$ — no gradient, which is correct (the group gave no information about which response is better).

---

## The GRPO Loss

### Clipped surrogate — same as PPO

For each response $i$ and each token $t$ within it:

$$r_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_\text{old}(a_t \mid s_t)} = \exp(\log\pi_\theta(a_t) - \log\pi_\text{old}(a_t))$$

The clipped actor loss for response $i$:

$$L_\text{clip}^{(i)} = -\frac{1}{T_i} \sum_{t=1}^{T_i} \min\left( r_t \cdot A_i,\ clip(r_t, 1-\varepsilon, 1+\varepsilon) \cdot A_i \right)$$

Note $A_i$ is **constant across all tokens in response $i$** — the whole response gets one score, unlike PPO where $A_t$ differs per token.

### KL penalty

$$L_\text{KL}^{(i)} = \frac{\beta}{T_i} \sum_{t=1}^{T_i} \log \frac{\pi_\theta(a_t \mid s_t)}{\pi_\text{ref}(a_t \mid s_t)}$$

This penalizes divergence from the reference (SFT) model. Clipping prevents large single steps; KL penalty prevents long-term drift.

### Full GRPO loss

$$\boxed{L_\text{GRPO} = \frac{1}{G} \sum_{i=1}^{G} \left( L_\text{clip}^{(i)} + L_\text{KL}^{(i)} \right)}$$

Expanding fully:

$$L_\text{GRPO} = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{T_i} \sum_{t=1}^{T_i} \left[
-\min\!\left( r_t A_i,\ \operatorname{clip}(r_t, 1-\varepsilon, 1+\varepsilon) A_i \right)
+ \beta \log \frac{\pi_\theta(a_t)}{\pi_\text{ref}(a_t)}
\right]$$

---

## PPO vs GRPO — The Key Differences

### Advantage: per-token vs per-response

**PPO** (script 2): $A_t$ is different for each token, computed via GAE:

$$A_t^\text{GAE} = \sum_k (\gamma\lambda)^k \delta_{t+k}$$

Token $t=0$ gets credit for all future rewards; token $t=T_r-1$ gets credit only for the last step.

**GRPO**: $A_i$ is one scalar per response, broadcast to all tokens:

$$A_i = \frac{R_i - \text{mean}(R_1..R_G)}{\text{std}(R_1..R_G)}$$

Every token in response $i$ receives the same gradient signal. There is no credit assignment within the response.

This is GRPO's main weakness vs PPO — a token that caused a bad outcome at $t=2$ and a token that had nothing to do with it at $t=4$ are treated identically. But in practice it works well because:
- The reward model implicitly captures which parts of the response mattered
- Averaging over G responses and many training steps provides enough signal

### Memory

| Setup | Models in memory |
|-------|-----------------|
| PPO | policy $\pi_\theta$, reference $\pi_\text{ref}$, critic $V_\phi$, reward model $R_\phi$ |
| GRPO | policy $\pi_\theta$, reference $\pi_\text{ref}$ |

GRPO reduces memory by ~33% by eliminating the critic. For a 70B model, this is the difference between needing 8 GPUs vs 6 GPUs.

---

## Tensor Shapes — Script 4

```
Setup
  G                       : int                   number of responses per prompt
  PROMPT                  : list of int            token ids — same for all G rollouts

Phase 1 — G rollouts (under π_old, no grad)
  all_tokens[i]           : list of int, len T_i   tokens sampled for response i
  all_log_probs[i]        : (T_i,)                 log π_old(a_t | s_t) for response i
  all_rewards[i]          : scalar float           R_i = rule-based reward for response i

Phase 2 — Group advantage
  rewards_t               : (G,)                   all R_i stacked
  mean_R                  : scalar                 mean of rewards_t
  std_R                   : scalar                 std  of rewards_t + 1e-8
  advantages              : (G,)                   A_i = (R_i - mean_R) / std_R

Phase 3 — Loss computation (per response i, with grad)
  lp_new_i                : (T_i,)                 log π_θ(a_t | s_t)   has grad
  lp_old_i                : (T_i,)                 log π_old(a_t | s_t) detached
  lp_ref_i                : (T_i,)                 log π_ref(a_t | s_t) detached

  ratio_i                 : (T_i,)                 exp(lp_new_i - lp_old_i), starts at 1.0
  clipped_i               : (T_i,)                 clamp(ratio_i, 1-ε, 1+ε)

  A_i                     : scalar                 broadcast to all T_i tokens

  unclip_i = ratio_i  * A_i   : (T_i,)
  clip_i   = clipped_i * A_i  : (T_i,)

  actor_loss_i  = -min(unclip_i, clip_i).mean()  : scalar
  kl_i          = (lp_new_i - lp_ref_i).mean()   : scalar
  kl_loss_i     = β * kl_i                        : scalar
  loss_i        = actor_loss_i + kl_loss_i        : scalar

Total loss
  total_loss = sum(loss_i for i in G) / G         : scalar
  total_loss.backward()                           : fills grads on π_θ only
```

---

## Bias–Variance: Where GRPO Fits

| Method | Advantage $A$ | Bias | Variance | Credit assignment |
|--------|--------------|------|----------|-------------------|
| REINFORCE | $R(\tau)$ | zero | very high | none |
| + flat baseline | $R(\tau) - \bar{R}$ | zero | lower | none |
| GRPO | $(R_i - \mu_G) / \sigma_G$ | low | low | per response |
| PPO + GAE | $\sum_k (\gamma\lambda)^k \delta_{t+k}$ | low | low | per token |

GRPO sits between script 1 and PPO in terms of credit assignment granularity, but matches PPO in bias-variance because the group mean is a strong, prompt-specific baseline.

---

## Hyperparameters

| Parameter | Typical value | Effect |
|-----------|--------------|--------|
| $G$ | 8–16 | Group size — larger $G$ → better baseline estimate, more compute |
| $\varepsilon$ | 0.2 | Clip range — same role as in PPO |
| $\beta$ | 0.01–0.1 | KL penalty — larger → stays closer to $\pi_\text{ref}$ |
| lr | 1e-6 – 1e-5 | Typically smaller than PPO (no critic warmup needed) |

---

## Full Training Loop

```
for each training step:

  1. Sample a batch of prompts  x_1, ..., x_B  from dataset

  2. For each prompt x_b:
       sample G responses  y_1, ..., y_G  ~  π_old(· | x_b)
       compute rewards      R_1, ..., R_G
       compute advantages   A_i = (R_i - mean) / std

  3. For each PPO epoch (K iterations on fixed data):
       for each response i:
         recompute lp_new under π_θ
         ratio_t = exp(lp_new - lp_old)
         L_clip  = clipped surrogate with A_i broadcast to all tokens
         L_kl    = β · mean(log π_θ - log π_ref)
         loss_i  = L_clip + L_kl
       total_loss = mean(loss_i) over all responses
       total_loss.backward()
       optimizer.step()

  4. Update π_old ← π_θ
```

The only structural difference from PPO's loop is step 2: instead of one rollout per prompt, you generate G and compute the advantage from their statistics rather than from a critic.
