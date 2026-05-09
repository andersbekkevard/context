# Recommended exercise 1

## Get the data

The pca-examples.rdata can be downloaded from the Blackboard. 

```r
# Modify the location based on your filesystem
load("../10_unsupervised_learning/datasets/pca-examples.Rdata")

# We will work with nyt.frame
nyt_data = nyt.frame
```

## Compute the PCA

```r
nyt_pca = prcomp(nyt_data[,-1])
```

## Default biplot

Too much information on the graph, we should select only a few loading vectors to display.

```r
biplot(nyt_pca, scale=0, cex = 0.5)
```

Lets pick some words with high PC1 weight and some words with high PC2 weight. Only looking at the graph we can see that PC1 is associated with music and PC2 with art.

```r
nyt_loading = nyt_pca$rotation[, 1:2]
informative_loadings = rbind(
  head(nyt_loading[order(nyt_loading[,1], decreasing = TRUE),]),
  head(nyt_loading[order(nyt_loading[,2], decreasing = TRUE),])
)

biplot(x = nyt_pca$x[,1:2], y= informative_loadings, scale=0, cex = 0.5)
```

## PVE 

The numbers below show that although the graphs based on PC1 and PC2 give some insight about document types, the first two PCs explain only small portion of the variability contained in the data.

```r
pr.var=nyt_pca$sdev^2
pr.var
```

We can then compute the proportion of variance explained by each principal component

```r
pve=pr.var/sum(pr.var)
pve
```

We can plot the PVE explained by each component, as well
as the cumulative PVE, as follows:

```r
plot(pve, 
     xlab="Principal Component", 
     ylab="Proportion of Variance Explained", 
     ylim=c(0,1),
     type='b')
plot(cumsum(pve), 
     xlab="Principal Component", 
     ylab="Cumulative Proportion of Variance Explained", 
     ylim=c(0,1),
     type='b')
```

# Recommended exercise 2

The answer is on page 388 of the book, with the explanation around Equation (10.12)

# Recommended exercise 3

## Perform k-means

```r
km.out=kmeans(nyt_data[,-1],2,nstart=20)
```

## Cluster assignments

```r
km.out$cluster
```

## Plot the data

To plot the data we need to use the PCA projections. Below we will use the plot based on PCA with true labels (A for art and M for music) and compare with the plot that color the points according to the k-means clustering.

```r
# PCA with true labels
plot(nyt_pca$x[,1:2],type="n")
points(nyt_pca$x[nyt_data[,"class.labels"]=="art",1:2],pch="A")
points(nyt_pca$x[nyt_data[,"class.labels"]=="music",1:2],pch="M")

# PCA with true labels but colored by k-means clustering
plot(nyt_pca$x[,1:2],type="n")
points(nyt_pca$x[nyt_data[,"class.labels"]=="art",1:2],pch="A",col=(km.out$cluster+1)[nyt_data[,"class.labels"]=="art"])
points(nyt_pca$x[nyt_data[,"class.labels"]=="music",1:2],pch="M",col=(km.out$cluster+1)[nyt_data[,"class.labels"]=="music"])
```

# Recommended exercise 4

## Perform hierarchical clustering

I will use euclidean distance and complete linkage

```r
hc.complete=hclust(dist(nyt_data[,-1]), method="complete")
str(hc.complete)
```

## Plot dendograms

```r
plot(hc.complete,main="Complete Linkage", labels=as.character(nyt_data[,1]), xlab="", sub="", cex=.9)
```

## Cluster assignment

```r
hc.clusters=cutree(hc.complete, 2)
```

## Plot the data

Based on the cluster assignment above and the plots below we see that hierarchical clustering performs worse than k-means for the same number of clusters (K=2)

```r
# PCA with true labels
plot(nyt_pca$x[,1:2],type="n")
points(nyt_pca$x[nyt_data[,"class.labels"]=="art",1:2],pch="A")
points(nyt_pca$x[nyt_data[,"class.labels"]=="music",1:2],pch="M")

# PCA with true labels but colored by k-means clustering
plot(nyt_pca$x[,1:2],type="n")
points(nyt_pca$x[nyt_data[,"class.labels"]=="art",1:2],pch="A",col=(hc.clusters+1)[nyt_data[,"class.labels"]=="art"])
points(nyt_pca$x[nyt_data[,"class.labels"]=="music",1:2],pch="M",col=(hc.clusters+1)[nyt_data[,"class.labels"]=="music"])
```
