## Paper Identity First

**Working title:** *Adaptive Developing Autoencoders: Data-Driven Growth Scheduling for Ordered Latent Representations*

**Core claim:** The Dev-AE's fixed growth schedule is a critical but unexamined design choice. We show that principled, data-driven growth triggers produce superior latent ordering, sparsity, and downstream performance — and explain *why* the coarse-to-fine ordering varies across datasets.

**Contributions (as you'd write them in the paper):**
1. Three principled adaptive growth triggers replacing the heuristic schedule
2. Systematic ablation of growth rate and timing (Gap 7 — the paper's ignored dimension)
3. A mutual information analysis explaining the MNIST/CIFAR-10 divergence (Gap 6)
4. Evaluation beyond linear probe: k-NN, transfer learning, interpolation quality

---

## Full Roadmap: 8 Weeks

```
Week 1  (Jun 9–15)   Setup + Baseline Reproduction
Week 2  (Jun 16–22)  Ablation Experiments (growth rate + timing)
Week 3  (Jun 23–29)  Adaptive Trigger Implementation
Week 4  (Jun 30–Jul 6)  Adaptive Trigger Experiments
Week 5  (Jul 7–13)   MI Analysis + Extended Evaluation
Week 6  (Jul 14–20)  Results Analysis + Figures
Week 7  (Jul 21–27)  Paper Writing
Week 8  (Jul 28–Aug 7) Revision + Submission
```

---

## Week 1 — Setup & Baseline Reproduction (Jun 9–15)

**Goal:** Working codebase, reproduced baselines, confirmed experimental setup.

### Tasks

**Day 1-2: Codebase audit**
- Clone the original repo: `https://github.com/davidvgr/developing-autoencoders`
- Read every file. Understand: training loop, growth trigger logic, evaluation code
- Identify what's missing: MI estimation, k-NN eval, adaptive trigger hooks
- Document all hardcoded hyperparameters (growth rate, trigger threshold, lr, etc.)

**Day 3-4: Reproduce original results**
- Run AE, PCA-AE, Dev-AE on CIFAR-10 with their exact settings
- Reproduce Fig 2A (loss curves), Fig 2B (intrinsic dimensionality), Fig 5B (classification accuracy)
- Target: your numbers should be within ~1-2% of theirs across 5 seeds
- If there are discrepancies, dig into them — they may reveal implementation details the paper glosses over

**Day 5: Extend to CIFAR-100**
- CIFAR-100 is the natural harder benchmark (same image space, 100 classes vs 10)
- Run the same AE/PCA-AE/Dev-AE baselines
- This becomes your second dataset for all subsequent experiments
- Note: CIFAR-100 has finer class distinctions → hypothesis is that late units matter even more than CIFAR-10

**Day 6-7: Experimental infrastructure**
- Build a config-driven experiment runner (YAML configs or argparse)
- Logging: W&B or MLflow — you need clean tracking across many runs
- Evaluation suite: wrap linear probe, k-NN (k=1,5,10,20), and TwoNN ID into a single `evaluate(encoder, dataset)` function
- Seed management: fix seeds, log them, run everything over N=10 seeds minimum (original uses 40 but that's expensive — 10 is defensible for ablations)

### Deliverable
A results table: AE / PCA-AE / Dev-AE × CIFAR-10 / CIFAR-100, with linear probe accuracy, k-NN accuracy, and intrinsic dimensionality. This is your Table 1 baseline.

---

## Week 2 — Growth Schedule Ablation (Jun 16–22)

**Goal:** Answer Gap 7 — what happens when the schedule is wrong?

### The Ablation Design

Two independent axes:

**Axis 1: Growth rate** (how many units added per step)
- 10% (very slow, conservative)
- 40% (moderate)
- 70% (original paper's choice)
- 150% (aggressive)
- Schedule: 6 → end at 128 in each case, just at different step sizes

**Axis 2: Growth trigger** (when to grow)
- Every N epochs fixed (N = 5, 10, 20 — map to "too fast", "medium", "too slow")
- Loss plateau trigger (original paper's approach — your baseline)

This gives you a 4×4 grid = 16 configurations × 2 datasets × 10 seeds = ~320 runs. This sounds like a lot but each run is short (60 epochs on CIFAR-10). Parallelize on your HPC.

### What to Measure per Configuration
- Final linear probe accuracy
- Final k-NN accuracy (k=5)
- Final intrinsic dimensionality
- **Ordering score**: quantify the frequency ordering of receptive fields. Concretely: for each model, compute the mean peak frequency of each neuron group's receptive field. A perfectly ordered model has monotonically increasing peak frequencies. Measure the Spearman correlation between neuron group index and mean peak frequency. This gives you a scalar "ordering score" between -1 and 1.
- Sparsity (mean inactive neurons per image)

### Key Hypotheses to Test
- Too-fast growth (150%, every 5 epochs) destroys ordering — new units don't have time to specialize before the next growth event
- Too-slow growth (10%, every 20 epochs) recovers ordering but wastes compute and may under-utilize capacity
- The original 70%/plateau combination is decent but not optimal — your adaptive triggers (Week 3) should beat it

### Deliverable
A heatmap figure: ordering score and accuracy as a function of growth rate × growth timing. This is a clean, visually compelling result that tells the whole story in one figure.

---

## Week 3 — Adaptive Trigger Implementation (Jun 23–29)

**Goal:** Implement and validate three principled growth triggers.

### Trigger 1: Intrinsic Dimensionality Saturation (ID-Trigger)

**Logic:** Grow when the model has "filled" its current capacity.

**Implementation:**
- Every epoch, compute TwoNN ID on a fixed subset of the test set (1000 samples is sufficient, full set is slow)
- Define saturation: ID hasn't increased by more than δ (e.g., δ=0.1) over the last W epochs (e.g., W=5)
- When saturated: grow by a fixed fraction (try 70% to match original, and also try ID-guided: add (target_ID - current_ID) × scale units)
- Key hyperparameter to tune: δ and W. Run a small grid: δ ∈ {0.05, 0.1, 0.2} × W ∈ {3, 5, 10}

**Why this is principled:** The TwoNN ID directly measures whether the model still has unexploited representational dimensions. Growing before saturation wastes the ordering benefit of the constraint; growing after is exactly the right time.

### Trigger 2: Gradient Variance Trigger (GV-Trigger)

**Logic:** Grow when the bottleneck weights are no longer receiving informative gradient signal.

**Implementation:**
- Every epoch, compute the variance of gradients w.r.t. bottleneck layer weights: `Var(∇L / ∇W_bottleneck)`
- Smooth over a window of W epochs
- Trigger growth when this variance drops below threshold τ (tune: τ as a percentile of the early-training variance)
- Growth amount: fixed 70% (decouple trigger from amount for clean ablation)

**Why this is principled:** Low gradient variance means the optimizer has found a flat region for the current capacity. This is a cleaner signal than loss plateau (which can be noisy and is computed on the full loss, not specifically the bottleneck).

### Trigger 3: Reconstruction-Classification Gap Trigger (RCG-Trigger)

**Logic:** The paper shows early units matter for reconstruction, late units for classification. Grow when the gap between reconstruction quality and classification quality stops narrowing — meaning the current capacity is insufficient to further improve discriminative structure.

**Implementation:**
- Every epoch, compute: reconstruction loss (already computed) + a lightweight proxy for classification quality (e.g., silhouette score of bottleneck activations, which doesn't require labels and is fast)
- Define the gap: G(t) = normalized_reconstruction_loss(t) - normalized_silhouette(t)
- When dG/dt < ε over W epochs, trigger growth
- This is the most novel trigger — it explicitly ties growth to the model's functional state rather than just optimization dynamics

**Note:** This trigger requires no labels, maintaining the unsupervised nature of the Dev-AE. That's an important point to make in the paper.

### Deliverable
Three cleanly implemented, configurable trigger classes with unit tests. Each trigger should log its internal signal (ID curve, gradient variance curve, gap curve) — these become supporting figures in the paper.

---

## Week 4 — Adaptive Trigger Experiments (Jun 30–Jul 6)

**Goal:** Full experimental comparison of all triggers.

### Experiment Design

Compare: Original (loss plateau, 70%) vs. ID-Trigger vs. GV-Trigger vs. RCG-Trigger

For each: CIFAR-10 and CIFAR-100, 10 seeds, evaluate on full suite (linear probe, k-NN, ordering score, sparsity, ID, number of growth events triggered).

**Additional analysis — growth event logging:**
- Log when each trigger fires and how many units are added
- Plot growth event timing vs. training curve
- Do adaptive triggers fire at "better" moments? Characterize this qualitatively

**Stress test:**
- Run all triggers on a harder dataset: STL-10 (96×96 images, only 5000 labeled, 100k unlabeled — tests whether adaptive growth helps more when data is scarce)
- This becomes your "scaling" result: fixed schedules degrade more than adaptive ones as data complexity grows

### Key Result to Aim For
ID-Trigger should be your best-performing adaptive method (it's the most directly motivated). GV-Trigger should be more compute-efficient (doesn't require TwoNN estimation). RCG-Trigger is the most novel but may be noisier.

### Deliverable
Main results table: all methods × all datasets × all metrics. This is your Table 2.

---

## Week 5 — MI Analysis + Extended Evaluation (Jul 7–13)

**Goal:** Explain the MNIST/CIFAR-10 divergence, extend evaluation.

### MI Analysis (Gap 6 Explanation)

**The question:** Why do early units dominate classification in MNIST but late units in CIFAR-10?

**The hypothesis:** In MNIST, class-discriminative information is low-frequency (digit shape). In CIFAR-10, it's higher-frequency (texture, fine structure). Dev-AE's ordering aligns units with frequency components, so whichever frequency range is most class-relevant determines which unit group is most classification-important.

**Implementation:**
- For each neuron group g and each class c, compute I(bottleneck_group_g; class_label_c) using MINE (Mutual Information Neural Estimation) or a simpler binning estimator (faster, sufficient for this analysis)
- Plot I(group; class) vs. group index for MNIST and CIFAR-10
- Expected result: MNIST curve peaks early, CIFAR-10 curve peaks late
- Cross-validate: run the same analysis on CIFAR-100 (should peak even later than CIFAR-10)

**Extend to frequency analysis:**
- Compute the mean spatial frequency of class-discriminative features for each dataset (this can be done by measuring the frequency content of class activation maps or simply of within-class vs. between-class variance in Fourier space)
- Show correlation: datasets where class information is higher-frequency → later Dev-AE units matter more
- This is a clean, testable prediction that generalizes beyond your three datasets

### Extended Downstream Evaluation

**Transfer learning:**
- Pretrain encoder (AE / PCA-AE / Dev-AE / best adaptive trigger) on CIFAR-10
- Freeze encoder, train linear head on STL-10 (different distribution, same visual domain)
- This tests whether Dev-AE's structured representations are genuinely more transferable

**Latent interpolation quality:**
- Sample 100 pairs of test images from different classes
- Interpolate 10 steps in latent space, decode each step
- Measure: FID of interpolated images (do they look realistic?), and semantic consistency (does the interpolation pass through plausible intermediate concepts?)
- Dev-AE's ordered structure should produce smoother, more semantically coherent interpolations

### Deliverable
- MI analysis figure (this becomes Fig. 4 or 5 in your paper — it's the analytical highlight)
- Transfer learning results (add column to Table 2)
- Interpolation figure (qualitative, goes in appendix or main paper as a visual)

---

## Week 6 — Results Analysis & Figures (Jul 14–20)

**Goal:** Polish all results, build all paper figures, identify the story.

### Figure Plan

| Figure | Content | Key message |
|--------|---------|-------------|
| Fig 1 | Schematic of adaptive triggers | Intuitive overview of all three trigger types |
| Fig 2 | Ablation heatmap (growth rate × timing) | Fixed schedules are sensitive; wrong settings hurt badly |
| Fig 3 | Trigger signal curves over training | Show when each trigger fires and why |
| Fig 4 | Main results: ordering score + accuracy | Adaptive > fixed across datasets |
| Fig 5 | MI analysis: group vs. class MI per dataset | Explains the MNIST/CIFAR-10 divergence |
| Fig 6 | Transfer learning + interpolation | Structured representations generalize better |

### Analysis Checklist
- Statistical testing: all comparisons need t-tests with Bonferroni correction (match the original paper's approach)
- Effect sizes: report Cohen's d, not just p-values
- Ablation sensitivity: which trigger hyperparameter (δ, W, τ) matters most? Run a sensitivity analysis
- Failure modes: document cases where adaptive triggers fail or give unexpected behavior — honest papers are stronger papers

### Story Check
At end of Week 6, you should be able to state the paper's narrative in 3 sentences:

*"The Developing Autoencoder's growth schedule is a critical but unexamined design choice. We show that data-driven triggers — particularly one based on intrinsic dimensionality saturation — consistently outperform fixed schedules across datasets of varying complexity, producing more ordered, sparser, and more transferable representations. Furthermore, our mutual information analysis reveals that the optimal trigger timing is governed by the frequency content of class-discriminative features, providing a principled basis for schedule design in future work."*

---

## Week 7 — Paper Writing (Jul 21–27)

### Writing Order (counterintuitive but efficient)

1. **Methods** first (you know exactly what you did)
2. **Results** second (describe figures and tables you already have)
3. **Introduction** third (frame the problem now that you know the answer)
4. **Related Work** fourth (place yourself in the literature)
5. **Discussion + Conclusion** fifth
6. **Abstract** last (compress everything into 150 words)

### Section Outline

**Abstract** (~150 words): Problem → gap → method → key results

**Introduction** (~0.75 pages):
- Hook: Dev-AE is promising but treats the growth schedule as a black box
- Gap: No principled criterion for when/how much to grow
- Contribution bullets (3)
- Brief results preview

**Related Work** (~0.5 pages):
- Ordered AE representations (nested dropout, PCA-AE, Dev-AE)
- Adaptive network growth (GradMax, Self-Expanding NNs)
- Information-theoretic representation analysis (MINE, IB theory)
- Keep tight — 5-8 pages total, related work shouldn't dominate

**Methods** (~1.5 pages):
- Brief Dev-AE recap (assume reader has read original paper but summarize key mechanics)
- Three adaptive triggers: one subsection each, with equations
- Evaluation protocol: datasets, metrics, seeds, compute

**Experiments** (~2.5 pages):
- Baseline reproduction (confirm your numbers match original)
- Growth schedule ablation
- Adaptive trigger comparison
- MI analysis
- Transfer learning (can be compact)

**Discussion** (~0.5 pages):
- When to use which trigger (practical guidance)
- Limitations: compute overhead of TwoNN, sensitivity to hyperparameters
- Future work: adaptive growth across all layers, not just bottleneck

**Conclusion** (~0.25 pages)

---

## Week 8 — Revision & Submission (Jul 28–Aug 7)

**Days 1-3:** Complete full draft, internal consistency check
- Every claim in the introduction has a corresponding result
- Every figure is referenced in the text
- All numbers in text match tables/figures

**Days 4-5:** Anonymization check (NLDL is double-blind)
- Remove all author names, affiliations
- Check acknowledgements section (remove or anonymize)
- Check references: no "our previous work" without blinding
- Check code links: if you share code, use anonymous GitHub

**Days 6-7:** LaTeX template compliance
- Download NLDL template from their GitHub
- Check page count (5-8 pages, references can overflow)
- DOIs on all references (NLDL requires this)
- Check all arXiv citations — if published version exists, cite that instead

**Days 8-10:** Final proofread, OpenReview account setup, submission

---

## Resource Requirements

| Resource | Estimate |
|----------|---------|
| GPU compute | ~500 GPU-hours total (320 ablation runs × ~1h + adaptive trigger runs) |
| Datasets | CIFAR-10/100 (auto-download), STL-10 (auto-download) — all free |
| New code | ~800 lines on top of original repo |
| Human time | ~5-6h/day for 8 weeks |

Your HPC should handle this fine given your existing nnU-Net workflows.

---

## Immediate Next Steps (This Week)

1. Clone the repo today, read it fully
2. Run the original Dev-AE on CIFAR-10 — confirm you can reproduce their Fig 2A loss curves
3. Set up W&B logging
4. Extend to CIFAR-100 baseline by end of week
