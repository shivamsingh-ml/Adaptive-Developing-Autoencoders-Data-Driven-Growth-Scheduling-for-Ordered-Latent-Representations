This is exactly the right calibration. Let me take all of this — the original roadmap, the audit fixes, and this scoping advice — and produce a single revised master plan that replaces everything before it.

---

## Revised Master Plan (August 7 Target)

### Paper Thesis (Sharpened)

**"Intrinsic dimensionality saturation predicts when representational growth should occur in developing autoencoders."**

This is cleaner and more defensible than "adaptive triggers improve performance." It makes a scientific claim, not just an engineering one, and it directly connects your ID trigger to the paper's own diagnostic tool (TwoNN).

---

### Revised Scope

| Component | Status | Priority |
|-----------|--------|----------|
| Baseline reproduction (AE, PCA-AE, Dev-AE) | Must-have | Week 1 |
| Growth schedule ablation | Must-have | Week 2 |
| ID trigger | Must-have | Week 3 |
| GV trigger | Strongly recommended | Week 3 |
| RCG trigger | Drop | — |
| k-NN + linear probe + ordering score | Must-have | Throughout |
| MI analysis (lightweight, kNN-based) | Stretch | Week 5 |
| Transfer eval (STL-10) | Stretch | Week 5 |
| Interpolation quality | Drop | — |

Dropping RCG and interpolation is the right call. The paper is stronger with two well-validated triggers than three rushed ones.

---

### Revised Weekly Roadmap

```
Week 1  (Jun 9–15)   Reproduction + Infrastructure
Week 2  (Jun 16–22)  Schedule Ablation (5 seeds)
Week 3  (Jun 23–29)  ID + GV Trigger Implementation + Smoke Tests
Week 4  (Jun 30–Jul 6)  Full Trigger Experiments (10 seeds)
Week 5  (Jul 7–13)   Gap 6 MI Analysis + Optional Stretch
Week 6  (Jul 14–20)  Results Consolidation + All Figures
Week 7  (Jul 21–27)  Paper Writing
Week 8  (Jul 28–Aug 7)  Revision + Submission
```

---

## Week 1 — Reproduction + Infrastructure

**Success criterion:** Your CIFAR-10 numbers for AE/PCA-AE/Dev-AE match the paper within ±2%, and CIFAR-100 runs end-to-end without errors.

### Codebase freeze checklist

Before running any experiment, confirm every critical fix is applied:

| Fix | File | Verified? |
|-----|------|-----------|
| `next_size()` uses `int()` not `ceil()` | `models/growth.py` | ☐ |
| `FixedTrigger.step_epoch()` increments `self.epoch` | `triggers/fixed_trigger.py` | ☐ |
| New unit init uses `N(μ,σ)` of existing weights | `models/growth.py` | ☐ |
| `IDTrigger` triggers on delta stall, not absolute value | `triggers/id_trigger.py` | ☐ |
| TwoNN NaN guard on degenerate input | `evaluation/intrinsic_dim.py` | ☐ |
| `linear_probe_eval` removes `multi_class` and `n_jobs` | `evaluation/linear_probe.py` | ☐ |
| `ordering_score` NaN guard + grouped variant | `evaluation/ordering_score.py` | ☐ |
| `GVTrigger` reference resets to current value | `triggers/gv_trigger.py` | ☐ |
| `compute_bottleneck_gradient_variance()` in solver | `solver.py` | ☐ |

### Results directory structure

Create this now, before any runs:

```
results/
├── raw/          ← one JSON per run (see format below)
├── processed/    ← aggregated CSVs per experiment group
├── figures/      ← final paper figures
└── tables/       ← LaTeX table source files
```

Every run saves exactly this JSON to `results/raw/`:

```json
{
  "run_id": "cifar10_devae_fixed_gr17_seed3",
  "seed": 3,
  "dataset": "cifar10",
  "model": "devae",
  "trigger": "fixed",
  "growth_rate": 1.7,
  "growth_timing": "plateau",
  "final_latent_dim": 128,
  "n_growth_events": 6,
  "growth_epochs": [6, 12, 19, 26, 34, 42],
  "linear_probe_acc": 0.381,
  "knn1_acc": 0.312,
  "knn5_acc": 0.334,
  "ordering_score_rho": 0.71,
  "ordering_score_grouped_rho": 0.95,
  "sparsity_mean_inactive": 23.1,
  "final_intrinsic_dim": 15.2,
  "train_loss_curve": [],
  "val_loss_curve": [],
  "id_curve": []
}
```

Committing to this structure now means Week 6 figure generation is just loading JSONs — no manual copy-pasting from logs.

### Reproduction targets

| Model | CIFAR-10 linear probe | CIFAR-10 ID (final) |
|-------|----------------------|---------------------|
| AE | ~30% | ~10 |
| PCA-AE | ~33% | ~12 |
| Dev-AE (fixed) | ~38% | ~15 |

Run 5 seeds each. If your mean is within ±2% and your variance is similar to their shaded bands in Fig 2A, you're good. Do not proceed to Week 2 until this is confirmed.

---

## Week 2 — Growth Schedule Ablation

**Success criterion:** You have a complete heatmap showing ordering score and accuracy as a function of growth rate × trigger timing. Clear degradation at extremes.

### Reduced experiment grid

| Axis | Values | Rationale |
|------|--------|-----------|
| Growth rate | 1.1, 1.4, 1.7, 2.5 | Covers slow/moderate/original/aggressive |
| Timing | every-5, every-10, every-20, loss-plateau | Covers fast/medium/slow/adaptive-baseline |
| Datasets | CIFAR-10, CIFAR-100 | Both from Day 1 |
| Seeds | **5** | Sufficient for ablation, upgrade to 10 in Week 4 |

Total: 4 × 4 × 2 × 5 = **160 runs**. Each ~20 min on one GPU → ~53 GPU-hours. Parallelise across your HPC as a SLURM array.

### Key hypotheses to confirm

- **GR-1.1 + every-5**: too fast — ordering score near 0, units don't specialise before next growth
- **GR-2.5 + every-20**: too slow — good ordering but underutilises capacity, lower accuracy
- **GR-1.7 + plateau**: original paper's sweet spot on CIFAR-10
- **CIFAR-100 diverges from CIFAR-10**: harder dataset likely benefits from slower growth — this is your first novel finding

### SLURM array structure

```bash
#SBATCH --array=0-159
python run_experiment.py \
  --config configs/ablation_growth_rate.yaml \
  --growth_rate  ${GROWTH_RATES[$((SLURM_ARRAY_TASK_ID % 4))]} \
  --timing       ${TIMINGS[$((SLURM_ARRAY_TASK_ID / 4 % 4))]} \
  --dataset      ${DATASETS[$((SLURM_ARRAY_TASK_ID / 16 % 2))]} \
  --seed         $((SLURM_ARRAY_TASK_ID / 32))
```

---

## Week 3 — ID + GV Trigger Implementation

**Success criterion:** Both triggers run for 60 epochs on CIFAR-10 without errors and produce at least 2 growth events. Smoke test on 1 seed before full runs.

### ID Trigger — final specification

```python
class IDTrigger(BaseTrigger):
    """
    Grow when TwoNN intrinsic dimensionality stops increasing.
    Trigger: delta_ID < delta_threshold for `patience` consecutive epochs.
    """
    def __init__(self, start_dim, max_dim, growth_rate=1.7,
                 min_epochs_per_stage=3, patience=3,
                 delta_threshold=0.1, sample_size=1000):
        super().__init__(start_dim, max_dim, growth_rate, min_epochs_per_stage)
        self.patience = patience
        self.delta_threshold = delta_threshold
        self.sample_size = sample_size
        self.prev_id = None
        self.bad_epochs = 0

    def should_grow(self, metrics):
        id_val = metrics.get("intrinsic_dim")
        if id_val is None:
            return False
        if self.prev_id is None:
            self.prev_id = id_val
            return False
        delta = id_val - self.prev_id
        self.prev_id = id_val
        if delta < self.delta_threshold:
            self.bad_epochs += 1
        else:
            self.bad_epochs = 0
        return self.can_grow() and self.bad_epochs >= self.patience

    def next_dim(self, metrics):
        self.bad_epochs = 0
        self.prev_id = metrics.get("intrinsic_dim")  # don't reset, continue tracking
        return self.grow_to(next_size(self.current_dim, self.max_dim, self.growth_rate))
```

Note `prev_id` is set to the current value (not `None`) after growth — you want the ID curve to continue smoothly, not restart.

### GV Trigger — final specification

The key addition vs. the scaffold is computing gradient variance in the solver. This must be accumulated per batch and averaged per epoch:

```python
# In train_one_epoch — after loss.backward(), before optimizer.step()
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    grad_vars = []

    for x, _ in loader:
        x = x.to(device)
        optimizer.zero_grad()
        x_hat = model(x)
        loss = loss_fn(x_hat, x)
        loss.backward()

        # Compute bottleneck gradient variance before step
        grads = []
        for name, param in model.named_parameters():
            if ('encoder_fc' in name or 'decoder_fc' in name) \
               and param.grad is not None:
                grads.append(param.grad.detach().flatten())
        if grads:
            grad_vars.append(torch.cat(grads).var().item())

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader), float(np.mean(grad_vars)) if grad_vars else 0.0
```

`train_one_epoch` now returns `(train_loss, gradient_variance)`. Update `train_with_trigger` accordingly.

### Smoke test protocol

Before queuing full runs for each trigger, run this manually:

```bash
python run_experiment.py \
  --config configs/adaptive_id_trigger.yaml \
  --dataset cifar10 --seed 0 --epochs 60
```

Confirm in the logs:
- At least 2 growth events fire
- Bottleneck reaches ≥ 50 units by epoch 40
- No NaN in loss or ID values
- Final linear probe accuracy ≥ 25% (sanity floor)

---

## Week 4 — Full Trigger Experiments

**Success criterion:** ID trigger beats fixed schedule (original 1.7/plateau) on both datasets. That's the paper's core claim.

### Final experiment grid

| Method | Datasets | Seeds |
|--------|---------|-------|
| AE (baseline) | CIFAR-10, CIFAR-100 | 10 |
| PCA-AE (baseline) | CIFAR-10, CIFAR-100 | 10 |
| Dev-AE fixed (original) | CIFAR-10, CIFAR-100 | 10 |
| Dev-AE fixed (best from Week 2) | CIFAR-10, CIFAR-100 | 10 |
| Dev-AE ID trigger | CIFAR-10, CIFAR-100 | 10 |
| Dev-AE GV trigger | CIFAR-10, CIFAR-100 | 10 |

Total: 6 × 2 × 10 = **120 runs** (~40 GPU-hours)

### The decisive comparison

Your paper lives or dies on this table. By end of Week 4 it should look something like:

| Method | CIFAR-10 LP | CIFAR-10 Ordering ρ | CIFAR-100 LP | CIFAR-100 Ordering ρ |
|--------|------------|---------------------|--------------|----------------------|
| AE | ~30% | ~0.1 | ~? | ~0.1 |
| PCA-AE | ~33% | ~0.4 | ~? | ~0.4 |
| Dev-AE (fixed) | ~38% | ~0.7 | ~? | ~? |
| Dev-AE (ID) | **?** | **?** | **?** | **?** |
| Dev-AE (GV) | **?** | **?** | **?** | **?** |

If ID trigger beats fixed on ordering score on both datasets, your thesis is confirmed.

---

## Week 5 — Gap 6 MI Analysis (Lightweight)

**Only start this if Week 4 results are clean and confirmed.**

### kNN-based MI estimator (not MINE)

Use `sklearn`'s k-NN MI estimator — it's stable, fast, requires no hypertuning, and is peer-reviewed:

```python
from sklearn.feature_selection import mutual_info_classif

def group_mi_analysis(model, loader, device, group_boundaries):
    """
    For each neuron group, estimate I(group_activations; class_label).
    Returns array of shape (n_groups,).
    """
    encodings, labels = extract_features(model, loader, device)
    # encodings: (N, 128), labels: (N,)
    
    group_mis = []
    for start, end in group_boundaries:
        group_acts = encodings[:, start:end]
        # mutual_info_classif returns MI per feature; take mean over group
        mi_per_unit = mutual_info_classif(group_acts, labels,
                                          discrete_features=False,
                                          random_state=42)
        group_mis.append(float(np.mean(mi_per_unit)))
    
    return np.array(group_mis)
```

Run this on your final Dev-AE models for CIFAR-10 and CIFAR-100. Plot `I(group; class)` vs. group index for both datasets. The expected result is the curve peaks earlier for CIFAR-10 (coarser class structure) and later for CIFAR-100 (finer class structure). This is Gap 6 explained.

This takes roughly 2 days to implement and run — well within Week 5.

---

## Week 6 — Results Consolidation + Figures

### Figure plan (6 main figures, 8-page budget)

| Figure | Content | Section |
|--------|---------|---------|
| 1 | Schematic: Dev-AE + ID/GV trigger overview | Methods |
| 2 | Ablation heatmap: ordering ρ and accuracy vs. growth rate × timing | Experiments |
| 3 | Trigger signal curves: ID curve with growth events marked | Experiments |
| 4 | Main results table (visualised as grouped bar chart) | Experiments |
| 5 | MI analysis: I(group; class) per dataset | Analysis |
| 6 | CIFAR-100 vs. CIFAR-10 ordering score comparison | Analysis |

### Aggregation script

Write `analysis/aggregate_results.py` that:
1. Loads all JSONs from `results/raw/`
2. Groups by `(dataset, trigger, growth_rate, growth_timing)`
3. Computes mean ± std across seeds
4. Outputs one CSV per experiment group to `results/processed/`
5. Flags any runs with NaN metrics (to catch silent failures)

Do this before attempting any figures. Clean data first.

---

## Week 7 — Paper Writing

### Writing order

1. Methods (you know exactly what you did)
2. Results (transcribe figures and tables)
3. Introduction (frame the problem now you know the answer)
4. Related Work (keep to 0.5 pages — NLDL is 5-8 pages total)
5. Discussion + Conclusion
6. Abstract last

### Section budget (8 pages)

| Section | Pages |
|---------|-------|
| Abstract | 0.15 |
| Introduction | 0.75 |
| Related Work | 0.5 |
| Methods | 1.5 |
| Experiments | 2.5 |
| Analysis (MI + CIFAR-100 comparison) | 0.85 |
| Discussion + Conclusion | 0.5 |
| References | overflow (no limit) |

### NLDL-specific checklist

- [ ] Double-blind: no names, affiliations, or self-citations without blinding
- [ ] Every reference has a DOI (NLDL requirement — check every arXiv paper for published version)
- [ ] LaTeX template from `https://github.com/SFI-Visual-Intelligence/nldl`
- [ ] OpenReview account created for all authors
- [ ] PDF checked for metadata stripping (author names can leak in PDF metadata)

---

## Week 8 — Revision + Submission

**Days 1-3:** Full draft complete, internal consistency pass
**Days 4-5:** Anonymisation audit + DOI check on all references
**Days 6-7:** LaTeX template compliance, page count, figure quality at print resolution (300 DPI minimum)
**Days 8-10:** Final proofread, submission

---

## Revised Compute Estimate

| Phase | Runs | GPU-hours |
|-------|------|-----------|
| Week 1 reproduction | 15 | ~5 |
| Week 2 ablation | 160 | ~55 |
| Week 4 full triggers | 120 | ~40 |
| Total | **295** | **~100** |

Comfortably within your HPC allocation given your existing nnU-Net workflows.

---

## Immediate Actions This Week

In order:

1. Apply all 9 fixes from the bug/warning audit checklist above
2. Run AE baseline on CIFAR-10, 5 seeds, confirm ~30% linear probe
3. Run Dev-AE fixed on CIFAR-10, 5 seeds, confirm ~38% linear probe
4. Run CIFAR-100 end-to-end sanity check (1 seed, AE only)
5. Commit frozen codebase to a tagged branch: `git tag v0.1-reproduced`
6. Queue Week 2 ablation array only after step 5

The tag is important — it's your reference point. Any experiment you run after this tag is on a stable, audited codebase.