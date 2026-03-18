# TensorFlow Core Learning Algorithms — Beginner's Guide

A hands-on introduction to 4 fundamental Machine Learning algorithms using TensorFlow and simple, self-created datasets.

---

## Table of Contents

1. [What This Project Covers](#what-this-project-covers)
2. [Setup & Installation](#setup--installation)
3. [Algorithm 1 — Linear Regression](#algorithm-1--linear-regression)
4. [Algorithm 2 — Classification](#algorithm-2--classification)
5. [Algorithm 3 — Clustering](#algorithm-3--clustering)
6. [Algorithm 4 — Hidden Markov Model](#algorithm-4--hidden-markov-model)
7. [Key Concepts Glossary](#key-concepts-glossary)
8. [How to Experiment](#how-to-experiment)
9. [Common Errors & Fixes](#common-errors--fixes)

---

## What This Project Covers

| # | Algorithm | Question It Answers | Output Type |
|---|-----------|---------------------|-------------|
| 1 | Linear Regression | "What number will this be?" | A number (e.g. price) |
| 2 | Classification | "Which category does this belong to?" | A label (e.g. pass/fail) |
| 3 | Clustering | "How can I group these?" | Group IDs (no labels needed) |
| 4 | Hidden Markov Model | "What comes next in this sequence?" | Probabilities / expected values |

> **Supervised vs Unsupervised:**
> - Algorithms 1 & 2 are **supervised** — your data has labels (you already know the answers during training).
> - Algorithm 3 is **unsupervised** — your data has no labels (the model finds patterns on its own).
> - Algorithm 4 uses **probability distributions** — you define the rules, and the model reasons over them.

---

## Setup & Installation

### Requirements

- Python 3.8 or higher
- pip (Python package manager)

### Install all dependencies

```bash
pip install tensorflow tensorflow-probability scikit-learn pandas numpy matplotlib
```

> **Version tip:** If TensorFlow Probability throws errors, pin it:
> ```bash
> pip install tensorflow==2.11.0 tensorflow-probability==0.19.0
> ```

### Verify installation

```python
import tensorflow as tf
import tensorflow_probability as tfp
print("TensorFlow:", tf.__version__)
print("TFP:", tfp.__version__)
```

---

## Algorithm 1 — Linear Regression

**File:** `linear_regression.py`

### What it does

Predicts a **continuous number** (house price) from an input feature (house size). It finds the best-fit straight line through the data points.

### The math (simplified)

```
price = (weight × size) + bias
```

The model learns the best `weight` and `bias` by minimising the difference between its predictions and the actual prices.

### The data

```python
house_sizes  = [500, 700, 800, ..., 3000]   # input  (feature)
house_prices = [150, 200, 220, ..., 660]    # output (label)
```

We split this into:
- **Training set** — first 10 rows (model learns from these)
- **Test set** — last 5 rows (we check accuracy on these)

### Key parts explained

```python
# 1. Feature column — tells TensorFlow what kind of data "size" is
feature_columns = [tf.feature_column.numeric_column('size')]

# 2. Input function — feeds data to the model in small batches
def make_input_fn(data_df, label_series, num_epochs=10, ...):
    ...

# 3. Model — a linear estimator (draws the best straight line)
model = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# 4. Train
model.train(train_input_fn)

# 5. Evaluate
result = model.evaluate(test_input_fn)

# 6. Predict
predictions = list(model.predict(predict_input_fn))
```

### What "epochs" means

One epoch = the model sees the **entire training dataset once**. We use multiple epochs (e.g. 10) so the model can improve its line of best fit over multiple passes.

```
Epoch 1: Model makes rough guesses → adjusts weights
Epoch 2: Model makes better guesses → adjusts again
...
Epoch 10: Model has converged to the best line it can find
```

### Expected output

```
Predictions:
  House size 1300 sq ft → predicted price: $330k
  House size 2100 sq ft → predicted price: $490k
  House size 3500 sq ft → predicted price: $695k
```

A plot `linear_regression.png` is saved showing the data points and the predicted line.

---

## Algorithm 2 — Classification

**File:** `classification.py`

### What it does

Predicts **which category** something belongs to. Here: will a student **pass or fail** based on study hours and sleep hours?

### The data

```python
data = {
    'study_hours': [1, 2, 1.5, 3, 4, 5, 6, ...],
    'sleep_hours': [4, 3, 5,   4, 5, 6, 7, ...],
    'passed':      [0, 0, 0,   0, 1, 1, 1, ...]   # 0=fail, 1=pass
}
```

### The model — Deep Neural Network (DNN)

Unlike a straight line in regression, a DNN learns complex, non-linear patterns. It has layers of "neurons" that each learn different features.

```
Input layer:   [study_hours, sleep_hours]
                        ↓
Hidden layer 1: 10 neurons  ← learns basic patterns
                        ↓
Hidden layer 2: 6 neurons   ← learns complex combinations
                        ↓
Output layer:  [prob_fail, prob_pass]
```

```python
model = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 6],   # 2 hidden layers with 10 and 6 neurons
    n_classes=2             # 2 possible outputs: pass or fail
)
```

> **Try this:** Change `hidden_units=[10, 6]` to `[20, 10, 5]` (3 layers) and see if accuracy improves. More neurons = more capacity to learn, but also more risk of overfitting.

### Training vs Testing data

```python
train_df = df.sample(frac=0.75)   # 75% of data → model learns from this
test_df  = df.drop(train_df.index) # 25% of data → we test accuracy on this
```

We **never** let the model see the test data during training. This ensures we're measuring real-world performance, not just memorisation.

### Expected output

```
Test Accuracy: 85.0%

Predictions for new students:
  Student 1 | Study: 1.0h, Sleep: 3.0h → FAIL ✗ (92% confident)
  Student 2 | Study: 7.0h, Sleep: 8.0h → PASS ✓ (97% confident)
  Student 3 | Study: 4.0h, Sleep: 4.0h → PASS ✓ (61% confident)
```

---

## Algorithm 3 — Clustering

**File:** `clustering.py`

### What it does

Groups data points into clusters **without any labels**. The algorithm finds natural groupings on its own. Here: segment customers by age and spending habits.

### Why no TensorFlow here?

TensorFlow's built-in KMeans has known bugs in recent versions. We use `sklearn.cluster.KMeans` instead — same algorithm, more stable.

### The K-Means algorithm (step by step)

```
Step 1: Randomly place K centre points in the data
Step 2: Assign every data point to its nearest centre
Step 3: Move each centre to the average of its assigned points
Step 4: Repeat steps 2-3 until nothing changes
```

### Choosing K — the Elbow Method

How do you know how many groups to use? Plot inertia (how tight the clusters are) for K=1 to 7. The "elbow" of the curve is your best K.

```python
for k in range(1, 8):
    km = KMeans(n_clusters=k)
    km.fit(df)
    inertias.append(km.inertia_)

# Plot this — the bend in the curve tells you the ideal K
```

A plot `elbow.png` is saved to help you visualise this.

### The data

We simulate 3 types of customers:
- **Young, low spenders** — age ~25, spending score ~20
- **Middle-aged, high spenders** — age ~40, spending score ~70
- **Senior, medium spenders** — age ~60, spending score ~45

```python
young_low   = np.column_stack([np.random.normal(25, 3, 20), np.random.normal(20, 5, 20)])
middle_high = np.column_stack([np.random.normal(40, 4, 20), np.random.normal(70, 8, 20)])
senior_mid  = np.column_stack([np.random.normal(60, 5, 20), np.random.normal(45, 6, 20)])
```

### Expected output

```
Cluster centres:
         age  spending_score
cluster
0       24.8            19.7
1       39.6            70.2
2       59.8            44.9

New customer (age=35, spending=65) → belongs to Group 1
```

A plot `clustering.png` is saved showing the 3 coloured groups.

---

## Algorithm 4 — Hidden Markov Model

**File:** `hmm.py`

### What it does

Models **sequences** where the true state is hidden. Here: we can't directly observe whether today is a "hot state" or "cold state" day, but we can observe the temperature — and use that to reason about the hidden states.

### The 3 components you must define

#### 1. Initial distribution
What is the probability of starting in each state?

```python
initial_dist = tfd.Categorical(probs=[0.8, 0.2])
# 80% chance day 1 is cold, 20% chance it's hot
```

#### 2. Transition distribution
Given today's state, what's the probability of tomorrow's state?

```python
transition_dist = tfd.Categorical(probs=[
    [0.7, 0.3],   # if cold today → 70% cold tomorrow, 30% hot
    [0.2, 0.8]    # if hot today  → 20% cold tomorrow, 80% hot
])
```

#### 3. Observation distribution
Given a hidden state, what temperature do we observe?

```python
observation_dist = tfd.Normal(
    loc=[0., 15.],      # cold days average 0°C, hot days average 15°C
    scale=[5., 10.]     # cold days range ±5°C, hot days range ±10°C
)
```

### Putting it together

```python
model = tfd.HiddenMarkovModel(
    initial_distribution=transition_dist,
    transition_distribution=transition_dist,
    observation_distribution=observation_dist,
    num_steps=7    # predict 7 days ahead
)

mean_temps = model.mean()   # expected temperature each day
sample     = model.sample() # one simulated week
```

### Expected output

```
Expected temperature for each day of the week:
  Mon:   2.1°C  ████
  Tue:   4.8°C  ██████
  Wed:   7.2°C  ████████
  Thu:   9.1°C  █████████
  Fri:  10.8°C  ██████████
  Sat:  12.0°C  ████████████
  Sun:  13.0°C  █████████████

One simulated week of temperatures:
  Mon:  -3.2°C  (❄️  cold)
  Tue:  18.4°C  (🔥 hot)
  ...
```

> **Why do temperatures rise across the week?** Because we start 80% cold. Over time, the transition probabilities push the expected state toward equilibrium between cold and hot.

---

## Key Concepts Glossary

| Term | Plain English Meaning |
|------|-----------------------|
| **Feature** | An input variable (e.g. house size, study hours) |
| **Label** | The answer we want to predict (e.g. price, pass/fail) |
| **Training data** | Data the model learns from |
| **Test data** | Data the model has never seen — used to measure real accuracy |
| **Epoch** | One full pass through the entire training dataset |
| **Batch size** | How many examples the model sees before updating its weights |
| **Loss** | How wrong the model's predictions are (lower = better) |
| **Accuracy** | % of correct predictions (higher = better) |
| **Overfitting** | Model memorises training data but fails on new data |
| **Hidden layer** | A layer inside a neural network (between input and output) |
| **Neuron** | A single unit in a neural network layer that learns a weight |
| **Cluster** | A natural grouping of similar data points |
| **Inertia** | How tight/compact the clusters are (lower = better) |
| **Hidden state** | A state in an HMM that you cannot directly observe |
| **Transition prob.** | The chance of moving from one state to another |

---

## How to Experiment

The best way to learn is to **change things and see what happens**. Here are ideas for each algorithm:

### Linear Regression
- Add more data points with different price ranges
- Add a second feature (e.g. number of bedrooms) and see if accuracy improves
- Change `num_epochs` from 10 to 100 — does the loss decrease?

### Classification
- Add a third class (e.g. "borderline") and change `n_classes=3`
- Try `hidden_units=[5]` (simpler model) vs `[50, 30, 10]` (complex) — which is more accurate?
- Deliberately put a student with 10 study hours and 1 sleep hour — what does the model predict?

### Clustering
- Change `n_clusters=3` to `n_clusters=2` or `5` and see how groups change
- Add a third feature (e.g. income) and observe how clusters shift
- What happens if you add outlier customers with extreme values?

### Hidden Markov Model
- Change `num_steps=7` to `num_steps=30` — do temperatures stabilise?
- Flip the initial distribution to `[0.2, 0.8]` (mostly hot) — does the weekly pattern reverse?
- Change the hot day mean from `15` to `35` — how do expected temperatures change?

---

## Common Errors & Fixes

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `ModuleNotFoundError: tensorflow` | TF not installed | `pip install tensorflow` |
| `ModuleNotFoundError: tensorflow_probability` | TFP not installed | `pip install tensorflow-probability` |
| `AttributeError: DNNClassifier` | Old TF version | Use TF 2.x: `pip install tensorflow>=2.9` |
| `ValueError: incompatible shapes` | Feature column mismatch | Check column names match `feature_columns` |
| `numpy` version conflict with TF | Incompatible numpy | `pip install numpy==1.23.5` |
| HMM `InvalidArgumentError` | TF/TFP version mismatch | Pin: `pip install tensorflow==2.11 tensorflow-probability==0.19` |

---

## Project Structure

```
project/
│
├── README.md                  ← You are here
├── linear_regression.py       ← Algorithm 1
├── classification.py          ← Algorithm 2
├── clustering.py              ← Algorithm 3
├── hmm.py                     ← Algorithm 4
│
└── outputs/
    ├── linear_regression.png  ← Line of best fit plot
    ├── elbow.png              ← K selection plot
    └── clustering.png         ← Customer segments plot
```

---

## Learning Path Recommendation

If this is your first time with ML, go in this order:

```
1. Run linear_regression.py  → understand features, labels, training
2. Run classification.py     → understand neural networks, accuracy
3. Run clustering.py         → understand unsupervised learning
4. Run hmm.py                → understand probabilistic models
5. Modify each script        → change data, parameters, and observe
6. Add your own dataset      → replace the dummy data with real data you care about
```

---

*Built for learning. Every script is self-contained and runs independently.*
