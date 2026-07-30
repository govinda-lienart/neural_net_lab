# The Whole Story — one fish crop's journey through training

Everything in the Neural Net Lab, connected into one narrative. No code. Read it
top to bottom and every term you've met should click into place.

Setting: AquaMind Stage 6.4. Five fish. A folder of curated crops. A frozen
DINOv2 backbone and a small head you're about to train.

---

## Before anything happens: the desk

A model is a **mixing desk**. Every **weight** is a fader, and its **value** is
where that fader currently sits. Nothing else is in there — no images, no fish,
no rules. Just fader positions.

Your desk has two sections:

| section | faders | state at the start |
|---|---|---|
| **backbone** (DINOv2) | ~21 million | already positioned by someone else's training — **frozen** ❄️ |
| **head** (yours) | 384 × 5 = 1,920 | scattered at **random** 🎲 |

**Why random and not all identical?** If every fader started in the same place,
every fader would be equally involved, receive identical feedback, and take
identical steps — forever. A million identical faders is just *one* fader
copied a million times. The random scatter is what lets them become different
from each other, and therefore specialise. It's called *symmetry breaking*.

---

## Step 1 — a crop becomes numbers

A `.jpg` is a compressed recipe, not a picture. Opening and transforming it
produces a **tensor** of shape `(3, 224, 224)` — three stacked grids (red,
green, blue), each 224×224. That's 150,528 numbers describing one fish.

Sixteen crops get stacked into a **batch**: `(16, 3, 224, 224)`.

---

## Step 2 — forward pass: the batch flows through the desk

The batch enters the backbone. Millions of frozen faders transform it, and out
the other side comes a **fingerprint**: 384 numbers per crop, capturing what the
fish looks like.

Those 384 numbers hit your head, where 1,920 faders turn them into **5 scores**,
one per fish. Highest score wins — that's the **guess**.

> In the lab's Network tab this is shrunk to 6 inputs and 5 outputs so you can
> actually see the 30 wires. Real shape: 384 inputs, 5 outputs, 1,920 wires.

---

## Step 3 — the loss: one number for the whole desk

Compare each guess to the truth. Not just right/wrong — **graded**:

```
guessed fish 3, truth fish 3, confident   →  0.05   😎
guessed fish 3, truth fish 3, unsure      →  0.90   😅
guessed fish 1, truth fish 3, confident   →  4.60   😱
```

Average across the 16 crops in the batch → **one number**. That's the **loss**.

Three things this number is *not*:

- not per-weight — all 21 million share this single score
- not an average of weights — it's an average over **crops**
- not a change — it's the value right now

Why graded rather than just counting mistakes? Counting gives whole numbers:
20 wrong, 19, 19, 19… flat, then a jump. Flat has no tilt, and no tilt means no
direction to move. The graded loss shifts smoothly when a fader moves a hair —
and a smooth shift is a **slope** you can follow.

### How the number is actually computed (`nn.CrossEntropyLoss`)

Three steps, from the five raw scores:

```
1. e^score for each        turns any score, even negative, into a positive number
2. divide by their total    → five percentages that add to 100%   (steps 1+2 = softmax)
3. loss = −ln(the percentage given to the TRUE fish)
```

Only the true fish's percentage matters; the other four are ignored. And the
shape of `−ln` is the point: 100% → loss 0, 10% → 2.30, 1% → 4.61. It punishes
**confident wrongness** hardest.

Two practical notes:

- `−ln` rather than `log₁₀` is mostly convention — a different base just scales
  every loss by a constant and changes nothing about which way is downhill.
  Logs matter because they turn *multiplying* probabilities into *adding* them,
  which stops a batch of small numbers underflowing to zero.
- This is why `label_map` converts fish IDs 1–5 into slots 0–4: the loss needs
  to know *which position* in the output list to look up.

### 📌 Sanity check: what loss should you expect?

With 5 fish, a model that knows nothing gives everyone 20%:

```
−ln(0.20) = 1.61      ← the clueless baseline
```

| loss | meaning |
|---|---|
| ≈ 1.61 | knows nothing — or isn't learning 🎲 |
| ≈ 0.7 | getting somewhere |
| ≈ 0.1 | confident and usually right 😎 |
| 2.5+ | worse than guessing — actively misled 😬 |

A fresh random head **should** start near 1.61. If it's still there after several
epochs, something is wrong — frozen head, bad learning rate, or broken labels.
This is the cheapest sanity check you have on day one of Stage 6.4.

---

## Step 4 — the landscape (a picture, never actually drawn)

Imagine freezing every fader but one, sweeping that single fader across all its
possible positions, and recording the loss at each. You'd get a **hill**.

- **up–down axis** = the whole model's loss
- **left–right axis** = one weight's value

Every weight has its own hill. Steep hill = this weight matters a lot. Flat
hill = the model barely notices it.

**At any moment, every weight sits at the same height** — because there's only
one loss. But each stands on differently tilted ground. *One score, millions of
different corrections.* (That's the One Loss, Many Weights tab.)

Nobody ever draws these hills. Drawing one would mean re-running your whole
dataset hundreds of times, for every weight. All the model ever knows is where
it's standing and how tilted the ground is there — like walking in fog. 🌫️

**What decides a hill's shape?** The architecture, the loss function, and above
all **your data**. Different crops → different landscape → different valleys.
Which is why a mislabelled crop is so damaging: it puts a lie in the terrain,
and gradient descent will faithfully roll into the wrong valley. It has no way
to know. This is what `curate_crops.py` and the global-fragment guardrail are
really protecting. 🚩

---

## Step 5 — backward pass: measuring every tilt at once

The **gradient** of a weight is the tilt of its hill where it stands. Two facts
folded into one number:

| the **sign** | which way is downhill |
| the **size** | how steep it is here |

And that size is itself two things multiplied:

> **tilt = how steep the hill × how far from the valley**

So a weight can sit far from its valley yet get a small correction (flat hill),
or sit close yet get a large one (steep hill).

**How are 21 million tilts computed without trying anything?** Not by testing
nearby positions — that would mean millions of full dataset runs per step.
Instead, **backpropagation** walks the network backwards, assigning blame:

> **blame = how loud a signal this weight handled × how much blame arrived from downstream**

```
weight A:  signal 0.02 × downstream 3.0  =  0.06   barely involved 😴
weight B:  signal 5.00 × downstream 3.0  =  15.0   heavily involved 😰
```

Football: the team lost 3–0 — one shared score. The striker who took five shots
gets a large correction; the sub who came on in the 89th minute gets almost
none. Nobody *decides* this; it falls out of how involved each player was.

**Two things follow from position in the network:**

- **Depth** — blame is multiplied at every station on the way back, so by the
  time it reaches early layers it's often faint (*vanishing gradients*).
- **Fan-out** — a weight feeding many downstream units collects blame from all
  of them.

Which is exactly why you freeze the backbone: those layers are already
excellent *and* receive the faintest blame. You train the part that both needs
it and can actually hear the feedback. 🎯

---

## Step 6 — the update: the only line that moves anything

```
new value  =  old value  −  learning rate  ×  tilt
              ────────      ─────────────     ────
              where it      YOU choose        backprop
              was           this              measured this
```

The **learning rate** is a plain multiplier: how much of the suggested step you
actually take.

```
tilt 7.00  ×  lr 0.15  =  move 1.05     3.50 → 2.45
```

- too big → the ball leaps past the valley and bounces higher each time 💥
- too small → it works, just painfully slowly 🐢

Typical real values: `1e-3` for a fresh head, `1e-4` fine-tuning, `1e-5` for
nudging a pretrained backbone. The learning rate is a **hyperparameter** — the
model never adjusts it and has no idea it exists.

Then the tilts are **thrown away** (`zero_grad()`). They were a measurement of
one moment, useless the instant the fader moved.

**Two distinctions worth keeping sharp:**

- The **value** is *set*; the **tilt** is *computed*. Values are saved in
  `brain.pt`; gradients are discarded every step.
- Backprop **measures** the slope, it doesn't change it. The hill never moves —
  only the ball walks.

---

## Step 7 — the diary

The loss gets logged:

```python
mlflow.log_metric("train_loss", loss.item(), step=i)
```

**The gradients are not on that chart.** Nothing about them is. There are
millions and they're all different; the loss is one number, so it's the only
thing that *can* be plotted.

Same vertical axis as the hills (loss), completely different horizontal one:

| | up–down | left–right |
|---|---|---|
| a hill 🏔️ | loss | one weight's value |
| MLflow 📉 | loss | **time (steps)** |

The hill is the terrain; MLflow is the trail log. Or: gradients are the gym
sessions, the chart is the scale reading. The gym is why the number drops, and
the gym is never on the chart. 🏋️

---

## Step 8 — round and round

One batch through and back = one **step**. All your crops seen once = one
**epoch**.

```
7,300 crops ÷ batch 16  =  456 steps  =  1 epoch
× 50 epochs             =  22,800 steps  =  22,800 nudges of every fader
```

Batches are **reshuffled at the start of each epoch**, not per step — within one
epoch every crop is seen exactly once. Shuffling matters especially for you:
`sorted(glob.glob(...))` groups your crops by fish, so without `shuffle=True`
the first batch would be 16 crops of fish 1 and nothing else, and the model
would thrash. 🃏

Why many small steps rather than one careful big one? All 7,300 crops at once
wouldn't fit in memory, and 456 rough steps beat one perfect step anyway.

---

## Step 9 — done

Loss drops steeply, bends, then flattens. **Flat is the finish line** — not
zero. A loss of zero means the model memorised your crops rather than learned
your fish, and it would fail on a fresh one. 🚩

```python
torch.save(model.state_dict(), "brain.pt")
```

Inside that file: the **fader positions**. Millions of numbers saying where each
dial ended up. No images, no fish, no logic — just settings. A trained model is
a good set of fader positions, found by nudging. That's the whole mystery.

---

## The four lines, if you remember nothing else

```
1. loss      = how wrong the model is            (one number)
2. gradient  = which way each weight should move (millions of numbers)
3. lr        = how big a step to take            (you pick it)
4. repeat
```

---

## Vocabulary check ✅

You should be able to say each of these in one sentence:

**weight · value · loss · gradient · learning rate · hyperparameter · forward
pass · backward pass · backpropagation · step · batch · epoch · shuffling ·
loss vs accuracy · initialisation · symmetry breaking · frozen backbone ·
transfer learning · vanishing gradients · `zero_grad()` · `state_dict()`**

If one of them is fuzzy, that's the next thing to poke at in the lab — not the
whole story again.
