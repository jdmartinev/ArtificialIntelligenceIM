# PPO from Scratch — Math & Tensor Shapes
### Three scripts, one coherent derivation

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $V$ | vocabulary size (8 in our scripts) |
| $D$ | embedding / model dimension (16) |
| $T_p$ | prompt length |
| $T_r$ | response length (varies per rollout) |
| $T = T_p + T_r$ | full sequence length |
| $B$ | batch size (1 in our scripts, noted where it generalizes) |
| $\theta$ | actor (LM) parameters |
| $\phi$ | critic parameters (script 2+) |
| $\pi_\theta(a \mid s)$ | probability the policy assigns to token $a$ given context $s$ |
| $G_t$ | discounted return from step $t$: $\sum_{k \geq 0} \gamma^k r_{t+k}$ |
| $V_\phi(s_t)$ | critic estimate of $\mathbb{E}[G_t \mid s_t]$ |
| $\delta_t$ | TD error at step $t$: $r_t + \gamma V(s_{t+1}) - V(s_t)$ |
| $A_t$ | advantage at step $t$: how much better action $a_t$ was than expected |

---

## Script 1 — REINFORCE with Baseline

### The objective

We want to find $\theta$ that maximizes expected reward over all possible responses:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

where $\tau = (a_1, \ldots, a_{T_r})$ is a full response sampled autoregressively and $R(\tau) \in \mathbb{R}$ is a scalar reward for the whole sequence.

### The log-derivative trick

Because sampling is non-differentiable, we use:

$$\nabla_\theta \mathbb{E}_\tau [R(\tau)] = \mathbb{E}_\tau [ R(\tau) \cdot \nabla_\theta \log \pi_\theta(\tau) ]$$

**Derivation sketch:**

$$\nabla_\theta \mathbb{E}_\tau [R] = \nabla_\theta \sum_\tau R(\tau)\,\pi_\theta(\tau) = \sum_\tau R(\tau)\,\pi_\theta(\tau) \cdot \frac{\nabla_\theta \pi_\theta(\tau)}{\pi_\theta(\tau)} = \mathbb{E}_\tau [ R(\tau)\,\nabla_\theta \log \pi_\theta(\tau) ]$$

Since the policy factorises autoregressively:

$$\log \pi_\theta(\tau) = \sum_{t=1}^{T_r} \log \pi_\theta(a_t \mid s_t)$$

so:

$$\nabla_\theta J(\theta) = \mathbb{E}_\tau \left[ R(\tau) \cdot \sum_{t=1}^{T_r} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]$$

This is the **REINFORCE** estimator. In code:

```python
loss = -R * log_probs.sum()   # scalar
loss.backward()               # fills .grad on all parameters
```

### Why variance is high

$R(\tau) \geq 0$ always (in our rule-based reward). Every gradient step pushes **all** tokens up, even tokens that contributed nothing. The signal is dominated by random fluctuations in $R$.

### Baseline subtraction and the Advantage

For any baseline $b$ that does not depend on $a_t$:

$$\mathbb{E}_\tau [ b \cdot \nabla_\theta \log \pi_\theta(\tau) ] = 0$$

**Proof:**

$$\mathbb{E}_\tau [ b \cdot \nabla_\theta \log \pi_\theta(\tau) ] = b \cdot \nabla_\theta \mathbb{E}_\tau[1] = b \cdot \nabla_\theta \sum_\tau \pi_\theta(\tau) = b \cdot \nabla_\theta\, 1 = 0$$

Therefore subtracting $b$ from $R$ leaves the gradient **unbiased** but reduces variance.
The quantity $R(\tau) - b$ is called the **advantage** $A$:

$$A = R(\tau) - b$$

It measures how much better (or worse) this particular rollout was **relative to the baseline expectation**. The gradient update becomes:

$$\nabla_\theta J(\theta) = \mathbb{E}_\tau \left[ A \cdot \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]$$

- $A > 0$ → this rollout was **better than expected** → increase the probability of all its tokens
- $A < 0$ → this rollout was **worse than expected** → decrease the probability of all its tokens
- $A = 0$ → exactly average → zero gradient, no update

In script 1, $b = \text{mean}(R_{\text{batch}})$, so $A$ is the same scalar for every token in the sequence.

### Tensor shapes — Script 1

```
Forward pass
  input  token_ids        : (1, T_p + t)          grows by 1 each generation step
  output logits           : (1, T_p + t, V)        one distribution per position
  slice  next_logits      : (V,)                   logits for the next token only

Sampling
  probs                   : (V,)                   softmax of next_logits
  action (int)            : scalar                 sampled token id

Log-prob collection
  log_prob (scalar)       : ()                     log pi_theta(a_t | s_t)
  log_probs_tensor        : (T_r,)                 stacked over all response steps

Reward
  R                       : scalar float           rule-based, whole sequence

Advantage (with flat baseline)
  b                       : scalar float           mean reward of the batch
  A = R - b               : scalar float           same value for ALL tokens in tau

Loss
  loss = -A * log_probs_tensor.sum()  : scalar     one number, .backward() fills grads
```

---

## Script 2 — Critic $V(s_t)$ and GAE

### The credit assignment problem

In script 1, the advantage $A = R(\tau) - b$ is a **single scalar shared by all tokens**. This is the fundamental weakness: token $a_0$ (the first word) gets the same gradient signal as token $a_{T_r-1}$ (the last word), even though their causal contributions to the final reward are completely different.

We need a **per-token advantage** $A_t$ that answers:
> *At step $t$, given everything that happened so far, how much better was action $a_t$ than what I would have expected on average?*

### The Critic: a per-token baseline

The critic $V_\phi(s_t)$ estimates the expected total future reward from state $s_t$:

$$V_\phi(s_t) \approx \mathbb{E}_{\pi_\theta}[G_t \mid s_t], \qquad G_t = \sum_{k=0}^{T_r - t - 1} \gamma^k\, r_{t+k}$$

This is the **state-dependent baseline** we needed. The per-token advantage is:

$$A_t = G_t - V_\phi(s_t)$$

- $G_t$ — what actually happened (real discounted return from $t$ onward)
- $V_\phi(s_t)$ — what the critic expected
- $A_t > 0$: actual return was better than predicted → this action was good
- $A_t < 0$: actual return was worse than predicted → this action was bad

The critic shares the transformer backbone with the actor, reading hidden state $h_t \in \mathbb{R}^D$:

$$V_\phi(s_t) = \text{MLP}_\phi(h_t) \in \mathbb{R}$$

### TD error — the one-step surprise

Computing $G_t$ exactly requires waiting until the episode ends. The **TD error** $\delta_t$ is an incremental per-step estimate of the surprise:

$$\delta_t = r_t + \gamma\, V_\phi(s_{t+1}) - V_\phi(s_t)$$

| Term | Meaning |
|------|---------|
| $V_\phi(s_t)$ | what the critic predicted we'd get from state $t$ |
| $r_t + \gamma V_\phi(s_{t+1})$ | one real reward step, then let the critic predict the rest |
| $\delta_t$ | the **surprise**: actual one-step outcome vs. prediction |

If the critic is perfect, $\mathbb{E}[\delta_t] = 0$ — no surprise, no gradient signal. Real critics are imperfect, so $\delta_t \neq 0$ and training continues. At EOS, bootstrap is $V_\phi(s_{T_r+1}) = 0$.

---

### GAE — Generalized Advantage Estimation

#### The core tension

We have two extreme options to estimate $A_t$:

**Monte Carlo:**

$$A_t^{\text{MC}} = G_t - V_\phi(s_t) = \sum_{k=0}^{T_r-t-1} \gamma^k r_{t+k} - V_\phi(s_t)$$

✅ Unbiased — no assumptions about $V_\phi$ &nbsp;&nbsp; ❌ High variance — sum of many random future rewards

**Pure TD(0):**

$$A_t^{\text{TD}} = \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

✅ Low variance — only one random term $r_t$ &nbsp;&nbsp; ❌ Biased — errors in $V_\phi$ propagate directly

GAE interpolates between both.

#### Building GAE from $k$-step returns

A $k$-step return advantage uses $k$ real reward steps, then bootstraps with $V_\phi$:

$$A_t^{(k)} = \sum_{i=0}^{k-1} \gamma^i r_{t+i} + \gamma^k V_\phi(s_{t+k}) - V_\phi(s_t)$$

Notice this can be written as a sum of TD errors:

$$A_t^{(1)} = \delta_t$$

$$A_t^{(2)} = \delta_t + \gamma \delta_{t+1}$$

$$A_t^{(k)} = \sum_{i=0}^{k-1} \gamma^i \delta_{t+i}$$

So **a $k$-step advantage is exactly the sum of the first $k$ TD errors from step $t$**. The full MC advantage ($k = T_r - t$) is all TD errors to the end.

GAE takes a **geometrically weighted average** of all $k$-step estimates:

$$A_t^{\text{GAE}}(\lambda) = (1-\lambda)\sum_{k=1}^{\infty} \lambda^{k-1} A_t^{(k)}$$

Substituting $A_t^{(k)} = \sum_{i=0}^{k-1} \gamma^i \delta_{t+i}$ and collecting terms by $\delta_{t+j}$:

$$A_t^{\text{GAE}} = \sum_{k=0}^{T_r-t-1} (\gamma \lambda)^k\, \delta_{t+k}$$

The weight on $\delta_{t+k}$ decays as $(\gamma\lambda)^k$ — near TD errors get higher weight, far-future ones are discounted.

#### What $\lambda$ controls

| $\lambda$ | Effective formula | Trusts | Bias | Variance |
|-----------|-------------------|--------|------|----------|
| $0$ | $A_t = \delta_t$ | Critic $V_\phi$ heavily | High | Low |
| $0.95$ | Near-even weights, slight decay | Mix of $V_\phi$ + real rewards | Low | Low |
| $1$ | $A_t = G_t - V_\phi(s_t)$ | Only real returns | Zero | High |

**Key insight:** $\lambda$ is not about how many steps ahead you look — it controls **how much you trust the critic vs. real sampled returns**. A perfect critic → small $\lambda$ is safe. An inaccurate critic (early training) → larger $\lambda$ is safer.

#### Recursive computation — why backwards

$$A_t^{\text{GAE}} = \delta_t + \gamma\lambda\, A_{t+1}^{\text{GAE}}, \qquad A_{T_r}^{\text{GAE}} = 0$$

This is why the code runs **backwards** from $t = T_r - 1$ to $t = 0$:

```python
gae = 0.0
for t in reversed(range(T_r)):
    gae = deltas[t] + gamma * lam * gae
    advantages[t] = gae
```

Each iteration reuses the previous `gae`, which equals $A_{t+1}^{\text{GAE}}$.

#### Numeric example

$T_r = 4$, $\gamma = 0.99$, $\lambda = 0.95$, TD errors: $\delta_0 = +0.10,\ \delta_1 = -0.05,\ \delta_2 = +0.20,\ \delta_3 = -0.08$

| Step | Computation | $A_t^{\text{GAE}}$ |
|------|-------------|---------------------|
| $t=3$ | $-0.08 + 0$ | $-0.0800$ |
| $t=2$ | $+0.20 + (0.9405)(-0.08)$ | $+0.1248$ |
| $t=1$ | $-0.05 + (0.9405)(+0.1248)$ | $+0.0674$ |
| $t=0$ | $+0.10 + (0.9405)(+0.0674)$ | $+0.1634$ |

$t=0$ accumulates more positive future signal than $t=1$; $t=3$ is negative because the final TD error was negative with nothing ahead to compensate. With a flat baseline instead ($R = 0.17$, $b = 0.10$), every token would get $A = 0.07$ — far less informative.

---

### Two losses, one backward pass

**Actor loss** — maximize per-token advantage-weighted log-probs:

$$\mathcal{L}_{\text{actor}} = -\sum_{t=1}^{T_r} A_t^{\text{GAE}} \cdot \log \pi_\theta(a_t \mid s_t)$$

**Critic loss** — regress $V_\phi$ onto actual returns:

$$\mathcal{L}_{\text{critic}} = \sum_{t=1}^{T_r} ( V_\phi(s_t) - G_t )^2$$

**Joint loss:**

$$\mathcal{L} = \mathcal{L}_{\text{actor}} + c_v \cdot \mathcal{L}_{\text{critic}}, \qquad c_v = 0.5$$

The feedback loop: better $V_\phi$ → smaller $\delta_t$ → lower variance $A_t^{\text{GAE}}$ → more reliable actor gradient → better policy → better data for critic.

### Tensor shapes — Script 2

```
Forward pass (single call, full context including prompt)
  input  token_ids        : (1, T_p + t)
  output logits           : (1, T_p + t, V)        actor head
  output values           : (1, T_p + t)            critic head — one V(s) per position
  slice  next_logits      : (V,)
  slice  next_val         : ()                      V(s_t) at current step

Per-step collections (stacked after rollout)
  log_probs_old           : (T_r,)                  log pi_theta(a_t | s_t)
  values                  : (T_r,)                  V(s_t) for t = 0..T_r-1
  rewards                 : (T_r,)                  r_t per step (sparse or dense)

TD errors
  values_ext              : (T_r + 1,)              values + [bootstrap = 0 at EOS]
  deltas                  : (T_r,)                  delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)

GAE advantages (computed backwards, Python loop)
  advantages              : (T_r,)                  A_t^GAE — different value per token

Returns (computed backwards)
  returns                 : (T_r,)                  G_t = r_t + gamma*G_{t+1}

Loss computation (recomputed with grad for backprop)
  adv_t                   : (T_r,)   tensor         detached — used as fixed targets
  ret_t                   : (T_r,)   tensor         detached — used as fixed targets
  lp_t                    : (T_r,)   tensor         has grad — actor parameters flow here
  val_t                   : (T_r,)   tensor         has grad — critic parameters flow here

  actor_loss  = -(adv_t * lp_t).sum()          : scalar
  critic_loss = MSE(val_t, ret_t)               : scalar
  joint_loss  = actor_loss + 0.5 * critic_loss  : scalar
```

---

## Script 3 — PPO Clipping

### The problem with multiple gradient steps

After collecting data under $\pi_{\text{old}}$, we want multiple gradient steps from the same rollout (data efficiency). Each step changes $\theta$, making $\pi_\theta \neq \pi_{\text{old}}$. The advantage $A_t^{\text{GAE}}$ was estimated under $\pi_{\text{old}}$ — if the policy drifts too far, $A_t$ becomes invalid and updates can be catastrophically large.

### The probability ratio

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)} = \exp(\log \pi_\theta(a_t \mid s_t) - \log \pi_{\text{old}}(a_t \mid s_t))$$

- $r_t = 1$ at the start of each PPO epoch
- $r_t > 1$: $\pi_\theta$ assigns more probability to $a_t$ than $\pi_{\text{old}}$
- $r_t < 1$: less probability

The **unclipped** surrogate: $L^{\text{PG}}(\theta) = \mathbb{E}_t [ r_t(\theta) \cdot A_t ]$

### The clipped surrogate

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \cdot A_t,\ \text{clip}(r_t(\theta),\, 1-\varepsilon,\, 1+\varepsilon) \cdot A_t \right) \right]$$

**Case $A_t > 0$:** gradient is zero when $r_t > 1+\varepsilon$ — already pushed enough.

**Case $A_t < 0$:** gradient is zero when $r_t < 1-\varepsilon$ — already decreased enough.

The clip only fires when going too far in the beneficial direction — never blocks penalising a bad action.

### Full PPO loss

$$\mathcal{L}^{\text{PPO}}(\theta) = -\mathbb{E}_t \left[ \min\left( r_t A_t,\ \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) A_t \right) \right] + c_v \cdot \mathbb{E}_t \left[ (V_\phi(s_t) - G_t)^2 \right] - c_e \cdot H[\pi_\theta(\cdot \mid s_t)]$$

Where the three terms are: actor (clipped surrogate) + critic (value regression) - entropy bonus (optional).

### Tensor shapes — Script 3

```
Frozen snapshot
  pi_old                  : copy.deepcopy(model)   weights do NOT change

Rollout (under pi_old, no grad)
  log_probs_old           : (T_r,)                 log pi_old(a_t | s_t)   FIXED
  values_old              : (T_r,)                 V(s_t)                  FIXED
  rewards                 : (T_r,)                 per-step                FIXED
  adv_t                   : (T_r,)                 GAE advantages          FIXED
  ret_t                   : (T_r,)                 discounted returns      FIXED

Per PPO epoch (pi_old frozen, model = pi_new updates)
  lp_new_t                : (T_r,)                 log pi_new(a_t | s_t)  changes each epoch
  val_t                   : (T_r,)                 V_new(s_t)             changes each epoch

  ratio_t = exp(lp_new_t - lp_old_t)   : (T_r,)   r_t in (0, inf), starts at 1.0
  ratio_clipped           : (T_r,)                 clamp(r_t, 1-eps, 1+eps)
  unclipped = ratio_t * adv_t           : (T_r,)
  clipped   = ratio_clipped * adv_t     : (T_r,)
  was_clipped             : (T_r,) bool            where ratio hit the boundary

  actor_loss  = -min(unclipped, clipped).sum()   : scalar
  critic_loss = MSE(val_t, ret_t)                : scalar
  joint_loss  = actor_loss + 0.5 * critic_loss   : scalar
```

---

## Full RLHF-PPO Loop

```
┌─────────────────────────────────────────────────────────────┐
│  1. ROLLOUT  (pi_old frozen)                                │
│     for each prompt in batch:                               │
│       generate response autoregressively                    │
│       collect: tokens, log_probs_old, values, rewards       │
│     compute: GAE advantages, returns          <- Script 2   │
│                                                             │
│  2. PPO EPOCHS  (K iterations on fixed data)                │
│     for epoch in range(K):                                  │
│       recompute log_probs_new, values under pi_theta        │
│       ratio_t = exp(log_p_new - log_p_old)   <- Script 3   │
│       L_clip  = clipped surrogate             <- Script 3   │
│       L_crit  = MSE(V, G)                     <- Script 2   │
│       L_total = L_clip + c_v * L_crit                      │
│       L_total.backward(); optimizer.step()    <- Script 1   │
│                                                             │
│  3. UPDATE pi_old <- pi_theta  (after K epochs)             │
└─────────────────────────────────────────────────────────────┘
```

In RLHF, the reward model score is augmented with a per-token KL penalty against the frozen SFT reference model $\pi_{\text{ref}}$:

$$\tilde{r}_t = r_t - \beta \cdot \log \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{ref}}(a_t \mid s_t)}$$

This replaces raw $r_t$ in the GAE computation, keeping the LM anchored to its SFT initialisation.

---

## Bias–Variance Summary

| Method | Advantage $A_t$ | Bias | Variance | Notes |
|--------|----------------|------|----------|-------|
| REINFORCE | $R(\tau)$ | zero | very high | No centering; $R \geq 0$ always |
| + flat baseline | $R(\tau) - \bar{R}$ | zero | lower | Same scalar $A$ for all tokens |
| + critic TD(0) | $\delta_t$ | high (bad $V_\phi$) | low | $\lambda=0$; critic errors become bias |
| + GAE $\lambda=0.95$ | $\sum_k (\gamma\lambda)^k \delta_{t+k}$ | low | low | Per-token; practical sweet spot |
| + PPO clip | same as GAE | — | — | Controls update size, not variance |

---

## Hyperparameters and Their Role

| Parameter | Typical value | Effect |
|-----------|--------------|--------|
| $\gamma$ | 0.99 | Discount — how much future rewards matter |
| $\lambda$ | 0.95 | GAE — how much to trust $V_\phi$ vs. real returns |
| $\varepsilon$ | 0.2 | PPO clip range — max policy change per epoch |
| $c_v$ | 0.5 | Critic loss weight relative to actor |
| $K$ | 4–8 | PPO epochs per rollout — efficiency vs. staleness |
| $\beta$ | 0.01–0.1 | KL penalty strength (RLHF only) |
| lr | 1e-4 – 3e-4 | Learning rate (sensitive in PPO) |
