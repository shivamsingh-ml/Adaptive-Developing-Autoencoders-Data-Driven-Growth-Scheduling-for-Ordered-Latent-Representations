Now you're getting into territory that is potentially **more novel than Dev-AE itself**.

The original Dev-AE assumption is:

```text
capacity should monotonically increase
6 → 10 → 17 → 29 → 50 → 85 → 128
```

But there's no guarantee the dataset actually needs 128 dimensions.

Your results already hint at this:

```text
CIFAR-10 final ID ≈ 12
CIFAR-100 final ID ≈ 12
```

yet you're ending at:

```text
latent_dim = 128
```

which suggests massive overcapacity.

---

## Dynamic Growth and Pruning

Instead of:

```text
grow only
```

you could have:

```text
grow
shrink
grow
shrink
```

based on learning signals.

Example:

```text
Epoch 0:
latent = 6

ID rises
→ grow to 10

ID rises
→ grow to 17

ID plateaus
→ stay

Many neurons inactive
→ shrink to 14

ID rises again
→ grow to 20
```

---

## Why this is interesting

Current Dev-AE asks:

> When should we add capacity?

Your idea asks:

> What is the minimum capacity needed at any point during learning?

That's much closer to biological development:

```text
synaptogenesis
↓
pruning
↓
specialization
```

Neuroscience actually observes:

```text
overgrowth
→ pruning
```

rather than monotonic growth.

---

## A simple implementation

Instead of physically rebuilding layers:

```python
latent_dim = 128
active_dim = 6
```

Use a mask:

```python
z = encoder(x)

mask[:active_dim] = 1
mask[active_dim:] = 0

z = z * mask
```

Growth:

```python
active_dim += k
```

Shrinkage:

```python
active_dim -= k
```

No model surgery required.

---

## What triggers shrinking?

### Option 1: Intrinsic Dimension

If:

```text
ID << active_dim
```

for several epochs:

```python
if intrinsic_dim < 0.5 * active_dim:
    shrink()
```

Example:

```text
active_dim = 64
ID = 12
```

Clearly overparameterized.

---

### Option 2: Neuron Utilization

Track:

```python
mean(|z_i|)
```

for each latent neuron.

If:

```text
neuron inactive for 10 epochs
```

remove it.

Very similar to network pruning.

---

### Option 3: Reconstruction Gain

Measure:

```text
improvement from last growth
```

If adding neurons did not improve reconstruction:

```python
undo_growth()
```

This is quite elegant.

---

## Even more interesting

You could frame the system as:

```text
Adaptive Capacity Controller
```

with actions:

```text
GROW
SHRINK
HOLD
```

and signals:

```text
Intrinsic Dimension
Gradient Variance
Neuron Activity
Validation Loss
```

Now you're approaching:

```text
Neural Architecture Adaptation
```

rather than just developing autoencoders.

---

## Research novelty

Ranking ideas we've discussed:

| Idea                                   | Novelty   |
| -------------------------------------- | --------- |
| Fixed Dev-AE                           | Low       |
| ID-triggered growth                    | Medium    |
| GV-triggered growth                    | Medium    |
| Medical datasets                       | Medium    |
| U-Mamba + ID growth                    | High      |
| Grow + Shrink bottleneck               | High      |
| Adaptive architecture growth + pruning | Very High |

The **grow-and-shrink bottleneck** is probably the most interesting next-step extension of Dev-AE because it directly challenges the paper's core assumption that capacity should only increase.

In fact, if your experiments showed:

```text
Dev-AE:
6 → 10 → 17 → 29 → 50 → 85 → 128

Adaptive:
6 → 10 → 17 → 14 → 18 → 15 → 20
```

with equal or better performance, that would be a very compelling result. It would suggest the network can discover its own appropriate latent dimensionality instead of following a predetermined expansion schedule.
