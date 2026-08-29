# Transient Refusal Dynamics in an Instruction-Tuned Transformer

This repository contains the code and analysis for a short mechanistic
interpretability study conducted for the Winter 2027 MATS application.

## Research question

Do jailbreak-related outcomes correspond to a localized departure from an
internal refusal-associated trajectory in an instruction-tuned transformer?

Rather than treating hidden states as static representations, I analyze the
trajectory of the residual stream across transformer layers.

## Model and data

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Benchmark: JailbreakBench
- Hardware: Apple Silicon using PyTorch MPS
- Representation: final-prompt-token residual stream
- Hidden-state levels: 37
- Hidden dimension: 2048

## Method

For each prompt \(x\), let \(h_\ell(x)\) denote its residual-stream
representation at layer \(\ell\). After normalizing the states, I construct an
independent refusal trajectory \(r_\ell\) using a separate collection of
consistently refused prompts.

The layerwise distance from the refusal trajectory is

$$
d_\ell(x)=\|\hat h_\ell(x)-r_\ell\|_2.
$$

I then define the local escape

$$
e_\ell(x)=d_{\ell+1}(x)-d_\ell(x).
$$

The exploratory late-layer escape score is

$$
E(x)=\max_{\ell=22,\ldots,30}e_\ell(x).
$$

## Main result

Using 57 non-ambiguous provisionally labeled examples:

- provisional failures: 22
- provisional successes: 35
- mean escape score, failures: 0.0715
- mean escape score, successes: 0.1066
- AUROC: 0.8325
- permutation p-value: 0.00020
- bootstrap 95% CI for AUROC: [0.693, 0.943]

The result suggests that jailbreak-related behavior may be associated with a
localized late-layer departure from a refusal-associated trajectory rather
than globally larger residual-stream movement.

## Important limitation

The final outcome labels are provisional and heuristic rather than produced by
an independent established jailbreak judge. The layer window was also chosen
during exploratory analysis rather than preregistered. The result should
therefore be interpreted as a candidate mechanistic signature rather than a
causal explanation of jailbreak behavior.

## Repository structure

- `code/` — main end-to-end experiment
- `figures/` — main figures

## Reproduction

Create a Python environment and install:

```bash
pip install -r requirements.txt
