# PCA — Unsupervised ML Technique

**Type:** Unsupervised Learning — Dimensionality Reduction  
**Category:** Data Transformation (not a predictive model)  
**Library:** sklearn.decomposition

---

## What is PCA?

PCA (Principal Component Analysis) is a dimensionality reduction technique that compresses a dataset with many features into fewer dimensions while retaining as much information as possible.

It does not predict anything. It transforms your data into a new shape — finding the directions where your data varies the most and projecting everything onto those directions.

---

## The core intuition

Imagine you have a dataset with 50 features. Many of those features are correlated — they carry overlapping information. PCA finds the directions of maximum variance in your data and creates new axes along those directions called **principal components**.

The first principal component captures the most variance. The second captures the next most. And so on. You then keep only the top K components — discarding the rest — and end up with a compressed but information-rich version of your data.

---

## The math behind it — simply

PCA is built on three concepts:

**Covariance matrix** — measures how every feature relates to every other feature. If two features move together, they are correlated. PCA uses this matrix to understand the structure of your data.

**Eigen decomposition** — breaks the covariance matrix into eigenvectors and eigenvalues. Eigenvectors are the special directions (principal components). Eigenvalues are how much variance each direction holds.

**Variance explained** — each eigenvalue tells you what fraction of the total variance that principal component captures. You sort them largest to smallest and keep enough to explain 95% of the total variance.

The flow looks like this:

```
Raw data
  → Standardize features
    → Compute covariance matrix
      → Eigen decomposition
        → Sort eigenvectors by eigenvalue
          → Keep top K components
            → Transform data into K dimensions
```

---

## Why standardization is mandatory before PCA

PCA is sensitive to the scale of features. A feature measured in thousands (like salary) will dominate over a feature measured in single digits (like age) — not because it is more important, but simply because its numbers are larger.

Standardizing all features to mean 0 and standard deviation 1 levels the playing field before PCA runs.

---

## Key hyperparameter

| Parameter | Role |
|---|---|
| `n_components` | Number of principal components to keep |

**How to choose n_components:** Use `explained_variance_ratio_` after fitting. Pick the value of K where the cumulative explained variance crosses 95%. This is the standard industry approach.

---

## Applications

| Domain | Use case |
|---|---|
| Healthcare | Compressing gene expression data before classification |
| Computer vision | Face recognition using eigenfaces |
| Finance | Reducing correlated financial indicators |
| NLP | Compressing word embedding dimensions |
| Manufacturing | Sensor data compression before anomaly detection |
| General ML | Speeding up slow models on high-dimensional data |

---

## When to use PCA

- Dataset has many correlated features
- Model training is too slow due to high dimensionality
- You want to visualize high-dimensional data in 2D or 3D
- You need to remove noise from data before modeling

## When NOT to use PCA

- You need interpretable features — principal components are combinations of original features and cannot be explained to stakeholders
- Your features are already independent — PCA gives no benefit
- You are using tree-based models like LightGBM or XGBoost — these handle high dimensions natively and PCA often hurts performance
- Your dataset has very few features — unnecessary overhead

---

## Key takeaways

- PCA is a transformer, not a predictive model — it has no target variable
- Always standardize your data before applying PCA
- Choose `n_components` based on cumulative explained variance — 95% is the standard threshold
- Larger eigenvalue = that principal component captures more variance = more important
- PCA is best used before algorithms that are sensitive to dimensionality like SVM, KNN, and Logistic Regression
