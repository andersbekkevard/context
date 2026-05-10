---
title: "Module 10: Recommended Exercises - Solutions"
subtitle: "TMA4268 Statistical Learning V2025"
author: "Kenneth Aase, Stefanie Muff, Sara Martino — Department of Mathematical Sciences, NTNU"
date: "March 26, 2025"
---

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, cut_tree
from ISLP.cluster import compute_linkage
import pyreadr
```

# Problem 1

Get the data by loading it.

```python
result = pyreadr.read_r("pca-examples.rdata")

# We will work with nyt.frame
nyt_data = result["nyt.frame"]
```

Compute the PCA. We drop the first column (the class label) and apply `PCA()` directly. Note that `prcomp` in R does not scale by default; the equivalent in `sklearn` is `PCA()` (which centers but does not scale).

```python
X = nyt_data.drop(columns=["class.labels"]).values
nyt_pca = PCA()
nyt_scores = nyt_pca.fit_transform(X)
```

## Default biplot

`sklearn` has no built-in biplot, so we build one manually as in the ISLP lab. Plotting all 4431 loading vectors would put too much information on the graph; we should select only a few to display.

```python
i, j = 0, 1  # which components
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(nyt_scores[:, i], nyt_scores[:, j])
ax.set_xlabel(f"PC{i+1}")
ax.set_ylabel(f"PC{j+1}")
for k in range(nyt_pca.components_.shape[1]):
    ax.arrow(0, 0,
             nyt_pca.components_[i, k],
             nyt_pca.components_[j, k])
```

Let's pick some words with high PC1 weight and some words with high PC2 weight. Only looking at the graph we can see that PC1 is associated with music and PC2 with art.

Remember that when selecting the high weights, we need to take the absolute value, since we don't care if they are negative or positive.

```python
# Each row of pcaUS.components_ is a loading vector; transpose so that
# rows correspond to features (words), columns to PCs — this matches R's
# nyt_pca$rotation layout.
nyt_loading = nyt_pca.components_[:2].T  # shape (n_features, 2)
feature_names = nyt_data.drop(columns=["class.labels"]).columns

top_pc1 = np.argsort(-np.abs(nyt_loading[:, 0]))[:6]
top_pc2 = np.argsort(-np.abs(nyt_loading[:, 1]))[:6]
informative_idx = np.unique(np.concatenate([top_pc1, top_pc2]))

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(nyt_scores[:, 0], nyt_scores[:, 1])
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
for k in informative_idx:
    ax.arrow(0, 0, nyt_loading[k, 0], nyt_loading[k, 1])
    ax.text(nyt_loading[k, 0],
            nyt_loading[k, 1],
            feature_names[k])
```

## Proportion of variance explained (PVE)

The numbers below show that although the graphs based on PC1 and PC2 give some insight about document types, the first two PCs explain only a small portion of the variability contained in the data.

```python
pr_var = nyt_pca.explained_variance_
pr_var
```

We can then compute the proportion of variance explained by each principal component (this is also available directly as `nyt_pca.explained_variance_ratio_`).

```python
pve = nyt_pca.explained_variance_ratio_
pve
```

We can plot the PVE explained by each component, as well as the cumulative PVE, as follows:

```python
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ticks = np.arange(nyt_pca.n_components_) + 1

ax = axes[0]
ax.plot(ticks, pve, marker="o")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Proportion of Variance Explained")
ax.set_ylim([0, 1])

ax = axes[1]
ax.plot(ticks, pve.cumsum(), marker="o")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Cumulative Proportion of Variance Explained")
ax.set_ylim([0, 1])
```

# Problem 2

The answer is on page 517 of the book (Second edition), with the explanation around Equation (12.18). (In the First edition, it is on page 388, Equation (10.12)).

# Problem 3

$k$-means clustering:

```python
km_out = KMeans(n_clusters=2, random_state=1, n_init=20).fit(X)
```

## Cluster assignments

```python
km_out.labels_
```

## Plot the data

To plot the data we need to use the PCA projections. Below we will use the plot based on PCA with true labels (A for art and M for music) and compare with the plot that colors the points according to the $k$-means clustering.

```python
labels = nyt_data["class.labels"].values
is_art = labels == "art"
is_music = labels == "music"

# PCA with true labels
fig, ax = plt.subplots(figsize=(8, 8))
for idx in np.where(is_art)[0]:
    ax.text(nyt_scores[idx, 0], nyt_scores[idx, 1], "A",
            ha="center", va="center")
for idx in np.where(is_music)[0]:
    ax.text(nyt_scores[idx, 0], nyt_scores[idx, 1], "M",
            ha="center", va="center")
ax.set_xlim(nyt_scores[:, 0].min() - 0.05, nyt_scores[:, 0].max() + 0.05)
ax.set_ylim(nyt_scores[:, 1].min() - 0.05, nyt_scores[:, 1].max() + 0.05)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
```

PCA with true labels but colored by $k$-means clustering:

```python
cluster_colors = np.array(["C1", "C2"])  # one color per cluster
km_colors = cluster_colors[km_out.labels_]

fig, ax = plt.subplots(figsize=(8, 8))
for idx in np.where(is_art)[0]:
    ax.text(nyt_scores[idx, 0], nyt_scores[idx, 1], "A",
            color=km_colors[idx], ha="center", va="center")
for idx in np.where(is_music)[0]:
    ax.text(nyt_scores[idx, 0], nyt_scores[idx, 1], "M",
            color=km_colors[idx], ha="center", va="center")
ax.set_xlim(nyt_scores[:, 0].min() - 0.05, nyt_scores[:, 0].max() + 0.05)
ax.set_ylim(nyt_scores[:, 1].min() - 0.05, nyt_scores[:, 1].max() + 0.05)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
```

# Problem 4

## Perform hierarchical clustering

We use Euclidean distance and complete linkage.

```python
HClust = AgglomerativeClustering
hc_complete = HClust(distance_threshold=0,
                     n_clusters=None,
                     linkage="complete").fit(X)
linkage_complete = compute_linkage(hc_complete)
```

## Plot dendrogram

```python
cargs = {"color_threshold": -np.inf,
         "above_threshold_color": "black"}

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
dendrogram(linkage_complete,
           ax=ax,
           labels=labels,
           leaf_font_size=9,
           **cargs)
ax.set_title("Complete Linkage")
```

## Cluster assignment

```python
hc_clusters = cut_tree(linkage_complete, n_clusters=2).ravel()
```

## Plot the data

Based on the cluster assignment above and the plots below we see that hierarchical clustering performs worse than $k$-means for the same number of clusters ($K=2$).

```python
# PCA with true labels
fig, ax = plt.subplots(figsize=(8, 8))
for idx in np.where(is_art)[0]:
    ax.text(nyt_scores[idx, 0], nyt_scores[idx, 1], "A",
            ha="center", va="center")
for idx in np.where(is_music)[0]:
    ax.text(nyt_scores[idx, 0], nyt_scores[idx, 1], "M",
            ha="center", va="center")
ax.set_xlim(nyt_scores[:, 0].min() - 0.05, nyt_scores[:, 0].max() + 0.05)
ax.set_ylim(nyt_scores[:, 1].min() - 0.05, nyt_scores[:, 1].max() + 0.05)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
```

PCA with true labels but colored by hierarchical clustering:

```python
hc_colors = cluster_colors[hc_clusters]

fig, ax = plt.subplots(figsize=(8, 8))
for idx in np.where(is_art)[0]:
    ax.text(nyt_scores[idx, 0], nyt_scores[idx, 1], "A",
            color=hc_colors[idx], ha="center", va="center")
for idx in np.where(is_music)[0]:
    ax.text(nyt_scores[idx, 0], nyt_scores[idx, 1], "M",
            color=hc_colors[idx], ha="center", va="center")
ax.set_xlim(nyt_scores[:, 0].min() - 0.05, nyt_scores[:, 0].max() + 0.05)
ax.set_ylim(nyt_scores[:, 1].min() - 0.05, nyt_scores[:, 1].max() + 0.05)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
```

Let's see if single linkage or average linkage performs better.

```python
hc_single = HClust(distance_threshold=0,
                   n_clusters=None,
                   linkage="single").fit(X)
hc_average = HClust(distance_threshold=0,
                    n_clusters=None,
                    linkage="average").fit(X)
linkage_single = compute_linkage(hc_single)
linkage_average = compute_linkage(hc_average)
```

# Plot dendrogram

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
dendrogram(linkage_single,
           ax=axes[0],
           labels=labels,
           leaf_font_size=9,
           **cargs)
axes[0].set_title("Single Linkage")
dendrogram(linkage_average,
           ax=axes[1],
           labels=labels,
           leaf_font_size=9,
           **cargs)
axes[1].set_title("Average Linkage")

# Divide into clusters
hc_clusters_single = cut_tree(linkage_single, n_clusters=2).ravel()
hc_clusters_average = cut_tree(linkage_average, n_clusters=2).ravel()
```

Plot clusters in PC1 and PC2-dimension:

```python
fig, axes = plt.subplots(2, 1, figsize=(8, 14))

for ax_, clust in zip(axes, [hc_clusters_single, hc_clusters_average]):
    cols = cluster_colors[clust]
    for idx in np.where(is_art)[0]:
        ax_.text(nyt_scores[idx, 0], nyt_scores[idx, 1], "A",
                 color=cols[idx], ha="center", va="center")
    for idx in np.where(is_music)[0]:
        ax_.text(nyt_scores[idx, 0], nyt_scores[idx, 1], "M",
                 color=cols[idx], ha="center", va="center")
    ax_.set_xlim(nyt_scores[:, 0].min() - 0.05, nyt_scores[:, 0].max() + 0.05)
    ax_.set_ylim(nyt_scores[:, 1].min() - 0.05, nyt_scores[:, 1].max() + 0.05)
    ax_.set_xlabel("PC1")
    ax_.set_ylabel("PC2")
```

Doesn't seem to provide a better fit than complete.
