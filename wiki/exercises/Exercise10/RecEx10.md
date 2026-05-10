---
title: "Module 10: Recommended Exercises"
subtitle: "TMA4268 Statistical Learning V2025"
author: "Kenneth Aase, Stefanie Muff, Sara Martino — Department of Mathematical Sciences, NTNU"
date: "March 26, 2025"
---

# PCA on New York Times stories

This exercise is based on the New York Time stories example (code and data) on the [lecture](http://www.stat.cmu.edu/~cshalizi/490/10/) of Brian Junker and Cosma Shalizi about Principal components and factor analysis.

## Data Exploration

New stories were randomly selected from the [New York Times Annotated Corpus](https://catalog.ldc.upenn.edu/LDC2008T19). There are 57 stories about art and 45 about music on the dataset available.

The `nyt_data` is a dataset containing these 102 stories, which represent the 102 observations. The first column contains class labels of the stories (art and music) and, and the other stories contain the counts of each word in a given story (weight by the inverse document frequency and normalized by vector length).

The New York Times stories dataset are contained in the file `pca-examples.rdata` which is already included in the directory. You can still download it from Google drive (https://drive.google.com/open?id=1vaK9GDvMw4Hsuv0T1jHeq9ZyrqLhJ6MR) and store in the directory of your file. The `pca-examples.rdata` is an R data file; one convenient way to read it from Python is via `pyreadr`.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyreadr

result = pyreadr.read_r("pca-examples.rdata")

# We will work with nyt.frame
nyt_data = result["nyt.frame"]  # dimension: (102, 4432)
```

```python
nyt_data.info()
nyt_data["class.labels"].value_counts()
```

Let's check some word samples:

```python
rng = np.random.default_rng(0)
nyt_data.columns[rng.choice(nyt_data.shape[1], 30, replace=False)]
```

Let's check some values in the dataset. We have many zeroes, as most words do not appear in most stories.

```python
row_idx = rng.choice(nyt_data.shape[0], 5, replace=False)
col_idx = rng.choice(nyt_data.shape[1], 10, replace=False)
nyt_data.iloc[row_idx, col_idx].round(3)
```

# Problem 1

- For the New York Times stories (`nyt_data`) dataset:
    - Create a biplot and explain the type of information that you can extract from the plot.
    - Create plots for the proportion of variance explained (PVE) and cumulative PVE. Describe what type of information you can extract from the plots.


# Problem 2

Show that the algorithm below is guaranteed to decrease the value of the objective

$$
\underset{C_1, \ldots, C_k}{\text{minimize}}\left\{\sum_{k=1}^{K} \frac{1}{|C_k|} \sum_{i, i' \in C_k} \sum_{j=1}^{p}(x_{ij} - x_{i'j})^2 \right\}
$$

at each step.

![](kmeans_algo.pdf)

# Problem 3

Perform $k$-means clustering in the New York Times stories dataset.

# Problem 4

Perform hierarchical clustering in the New York Times stories dataset.
