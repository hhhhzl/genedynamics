# MRMFMBD Theory: S1 + S3

## Posterior Bridge

Target: sample from posterior π(θ|R) ∝ p₀(θ) p(R|θ) over design+controller θ=(x,φ).

Annealed bridge: πₖ(θ) ∝ p₀(θ) p(R|θ)^βₖ with 0 ≤ β₀ < … < βₖ = 1.

## S1: Mode Marginalization

Discrete latent modes c (contact/friction regimes). Marginal likelihood:

```
p(R|θ) = Σ_c p(c) p(R|θ,c)
```

Boltzmann reward-to-likelihood: p(R|θ,c) ∝ exp(R_c/T).

```
log p(R|θ) = log Σ_c p(c) exp(R_c/T) = logsumexp(log p(c) + R_c/T)
```

Responsibilities (posterior over modes):

```
w_c = p(c|θ,R) = softmax(log p(c) + R_c/T)
```

**Critical:** Annealing βₖ is applied externally:
```
log πₖ(θ) = log p₀(θ) + βₖ · log p(R|θ)
```
β must NOT appear inside the mixture terms.

## S3: Multi-Fidelity Ladder

Fidelity levels ℓ ∈ {0,1,2} (coarse → fine). Cost(ℓ₀) < Cost(ℓ₁) < Cost(ℓ₂).

Schedule: ℓ(k) maps bridge step k to level. Geometric allocation:
```
n_ℓ = n₀ · r^ℓ,  Σ_ℓ n_ℓ = K
```
More steps at coarse → cost-optimal.

Ladder types: fixed, geometric, linear, cosine.

## MCSA Score Ascent

Zeroth-order gradient: proposals z_m = θ + σ·ε_m, weights w_m ∝ π(z_m).

```
score = (1/σ) Σ_m w_m · ε_m
θ_next = θ + η·score + τ·noise
```
