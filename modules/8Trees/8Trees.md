## Learning material for this module

* James et al (2021): An Introduction to Statistical Learning. Chapter 8.  
* All the material presented on these module slides.

## What will you learn?

You will get to know

* Decision trees
    * Regression trees  
    * Classification trees  

* Pruning trees

* Bagging

* Variable importance

* Random forests

... and how to apply all that.

## Example 1 (from chapter 8.1; `Hitters` data)

* Baseball players' salaries may depend on their experience (in years) and the number of hits.

* High salaries (yellow, red) vs low salaries (blue, green), salaries given on $\log$-scale. How can these be stratified for prediction of the salary? 

![Hits](hits.png){width=60%}

## Main idea of tree-based methods

* Divide the area into rectangles with similar salaries. 

* Idea: Derive a set of decision (splitting) rules for segmenting the predictor space into a number of finer and finer regions. 

* All points in the same region will be given the same predictive value (the mean of all values in that square, or a majority vote).

Visualisaztion in two dimensions:

![ISLR Figure 8.1](../../ISLR/Figures/Chapter8/82.png){width=50%}

* CARTs (Classification and regression trees) are usually drawn upside down, where the top node is called the *root*.

* The series of splitting rules can be visualized with a _regression tree_.

* The following tree (which corresponds to the split in the previous slide) has three _leaves_ (terminal nodes), and two _internal nodes_.

![ISLR Figure 8.1](../../ISLR/Figures/Chapter8/81.png){width=45%}

## Interpretation

* Years is the most important factor in determining Salary, and players with less experience earn lower salaries than more experienced players.

* Given that a player is less experienced, the number of Hits that he made in the previous year seems to play little role in his Salary.

* But among players who have been in the major leagues for five or more years, the number of Hits made in the previous year does affect Salary, and players who made more Hits last year tend to have higher salaries.

* Compared to a regression model, it is easy to display, interpret and explain.

# Regression tree (continous outcome)

Assume that we have a dataset consisting of $n$ pairs $(\boldsymbol{x}_i,y_i)$, $i=1,\ldots,n$, and each predictor is ${\boldsymbol{x}}_i=(x_{i1},x_{i2},...,x_{ip})$. The aim is to predict $y_i$.

Two steps:

1. Divide the predictor space into non-overlapping regions $R_1,R_2,\ldots,R_J$.

2. For every observation that falls into region $R_j$ we make the same prediction - which is the mean of the responses for the training observations that fall into $R_j$.

**But**: _How to divide the predictor space_ into non-overlapping regions $R_1,R_2,\ldots,R_J$?

We could try to minimize the RSS (residual sums of squares) on the training set given by

$$
\text{RSS}=\sum_{j=1}^J \sum_{i \in R_j}(y_i-\hat{y}_{R_j})^2,
$$

where $\hat{y}_{R_j}$ is the mean response for the training observations in region $j$. The mean $\hat{y}_{R_j}$ is also the predicted value for a new observations that falls into region $j$. 

To do this we need to consider every partition of the predictor space, and compute the RSS for each partition.

**But**: An exhaustive search over all possible splits is _computationally infeasible_! \footnote{In fact, constructing optimal binary decision trees is an NP-complete problem (Hyafil and Rivest, 1976).}

## Recursive binary splitting

* A _greedy_ approach is taken (aka top-down) - called _recursive binary splitting_: Find a split that minimizes RSS at each step\footnote{This does not necessarily give the optimal global solution, but will give the best solution at each split, given what is done previously.}.

* Start at the top of the tree and divide the predictor space into two regions $R_1(j,s)=\{x \mid x_j<s\}$ and $R_2(j,s)=\{x \mid x_j\geq s\}$, by making a decision rule for one of the predictors $x_1, x_2,...,x_p$.

* We thus need to find the (predictor) $j$ and (splitting point) $s$ that minimize
$$\sum_{i: x_i \in R_1(j,s)}(y_i-\hat{y}_{R_1})^2+\sum_{i: x_i \in R_2(j,s)}(y_i -\hat{y}_{R_2})^2 \ ,$$
where $\hat{y}_{R_1}$ and $\hat{y}_{R_2}$ are the mean responses for the training observations in $R_1(j,s)$ and $R_2(j,s)$ respectively. This way we get the two first branches in our decision tree.

* We repeat the process to make branches further down in the tree. 

* For every iteration we let each single split depend on _only one of the predictors_, giving us two new branches. 

* This is done _successively_ and in each step we choose the split that gives the _best split at that particular step_, _i.e.,_ the split that gives the smallest RSS. 

* However, this time, instead of splitting the entire predictor space, we split only _one of the previously identified regions_.

* Continue splitting the predictor space until we reach some _stopping criterion_. For example we stop when a region contains less than 10 observations or when the reduction in the RSS is smaller than a specified limit.

**Q**: Why is this algorithm called _greedy_?

## Regression tree: ozone example

Consider the `ozone` dataset from the `ElemStatLearn` library. The dataset consists of 111 observations on the following variables:

* `ozone` : the concentration of ozone in ppb
* `radiation`: the solar radiation (langleys)
* `temperature` : the daily maximum temperature in degrees F
* `wind` : wind speed in mph

```r
library(ElemStatLearn)
library(RColorBrewer)
myozone=ElemStatLearn::ozone
colnames(myozone)=c("ozone","radi","temp","wind")
```

```r
head(myozone)
```

Let's fit a regression tree with `ozone` as our response variable and `temperature` and `wind` as predictors.

```r
set.seed(300)
ozone.trainID = sample(1:111, 75)
ozone.train = myozone[ozone.trainID, ]
ozone.test = myozone[-ozone.trainID,]
```

Use the default settings in the tree function:

```r
library(tree)
ozone.tree = tree(ozone~temp+wind, data=ozone.train)
```

```r
plot(ozone.tree,type="uniform")
text(ozone.tree)
```

* We see that `temperature` is the "most important" predictor for predicting the ozone concentration. 
Observe that we can split on the same variable several times.

* Focus on the regions $R_j$, $j=1,\ldots, J$. What is $J$ here? 

### Tree- vs region plot

```r
par(mfrow=c(1,2),pty="s")
o.class = cut(ozone.train$ozone, breaks= 7)
col.oz = brewer.pal(9, "BuGn")
palette(col.oz[3:9])
plot(ozone.tree,type="uniform")
text(ozone.tree,cex=0.8)
plot(wind~temp, col=o.class, data=ozone.train, pch=20)
partition.tree(ozone.tree, add=TRUE,cex=0.8)
```

**Q**:

* Explain the connection between the tree and the region plot.
* Advantages and disadvantages of letting each single split depend on only one of the predictors?
* Does our tree include interactions between variables?

## R: function `tree` in library `tree`

* R package `tree` by Brian D. Ripley [@tree2019].

* Note: The default choice for a function to minimize is the deviance, and for normal data (as we may assume for regression), the deviance is proportional to the RSS.

* A competing R function is `rpart`, explained in <https://cran.r-project.org/web/packages/rpart/vignettes/longintro.pdf>

### Stopping criterion

* When building the tree, we can (in principle) split until each leaf corresponds to one data point.

* Usually we use a less stringent stopping criterion, like the minimal number of nodes per region and/or the minimum reduction in the RSS.

* For example, the default in `tree()`: is given by `mincut=5, minsize=10, mindev=0.01`, thus a minimal number of observations in a node is 5, the smallest nodes that are potentially split is 10, and the minimal reduction in deviance (RSS) equal to 0.01 times the deviance of the root note. 

* We could make the ozone tree much deeper:

```r
ozone.tree2 = tree(ozone~temp+wind, 
                  data=ozone.train, 
                  control = tree.control(75, mincut = 2, minsize = 4, mindev = 0.001)
                  )
```

With the above command, the tree would become very fine. Overfitting?

```r
plot(ozone.tree2)
text(ozone.tree2,cex=0.8)
```

## Tree performance

Test the predictive performance of our regression tree by using a training and a test set. But:

_How do we know if our tree performs good, or if there would be trees with a better predictive performance?_

# Pruning 

* If we have a dataset with many predictors or choose a stringent stopping criterion, we may fit a (too) large tree. $\rightarrow$ _overfitting_? 

* A smaller tree with fewer splits leads to fewer regions $R_1, \ldots, R_J$ with more observations. $\rightarrow$ Lower variance and better interpretation, but more bias.

* One possible alternative to the process described above is to grow the tree only so long as the decrease in the RSS due to each split exceeds some (high) threshold.

* This strategy will result in smaller trees, but is _too short-sighted_: a seemingly worthless split early on in the tree might be followed by a very good split — that is, a split that leads to a large reduction in RSS later on.

## Cost complexity pruning

* Better idea: to grow a very large tree $T_0$, and then _prune_ it back in order to obtain a _subtree_.

* _Cost complexity pruning_ is used for this: We try to find a subtree $T\subset T_0$ that (for a given value of $\alpha$) minimizes
$$
C_{\alpha}(T)=Q(T)+\alpha |T|,
$$
where $Q(T)$ is our cost function, $|T|$ is the number of terminal nodes in tree $T$. The parameter $\alpha$ is then a parameter penalizing the number of terminal nodes, ensuring that the tree does not get too many branches. 

* For regression trees (continous outcome) we choose 
$$Q(T)=\sum_{m=1}^{|T|}\sum_{x_i\in R_m}(y_i - \hat{y}_{R_m})^2 \ .$$

* For $\alpha>0$ we get a pruned tree (note: the same pruned tree within a small range of $\alpha$).

* For $\alpha=0$ we get $T_0$. 

* As $\alpha$ increases we get shorter and shorter trees.

So, which value of $\alpha$ is best?

* We can use $K$-fold cross-validation to find out!

* Importantly, by increasing $\alpha$, branches get pruned in a _hierarchical (nested) fashion_.

Please study this [note from Bo Lindqvist in MA8701 in 2017 - Advanced topics in Statistical Learning and Inference](https://www.math.ntnu.no/emner/TMA4268/2018v/notes/CART1MA87012017BoLindqvist.pdf) for an example of how we perform cost complexity pruning in detail\footnote{see also link on course website}. Alternatively, this method, with proofs, are given in @Ripley, Section 7.2.

## Building a regression tree: Algorithm 8.1

1. Use recursive binary splitting to grow a large tree on the training data, stopping only when each terminal node has fewer than some minimum number of observations.

2. Apply cost complexity pruning to the large tree in order to obtain a sequence of best subtrees, as a function of $\alpha$.

3. Use $K$-fold cross-validation to choose $\alpha$. That is, divide the training observations into $k$ folds. For each $k = 1,\ldots, K$:
      + Repeat Steps 1 and 2 on all but the $k$th fold of the training data.
      + Evaluate the mean squared prediction error on the data in the left-out $k$th fold, as a function of $\alpha$.
      + Average the results for each value of $\alpha$, and pick $\alpha$ to minimize the average error.

4. Return the subtree from Step 2 that corresponds to the chosen value of $\alpha$.

### Pruning the ozone tree

Let us start with the tree with many leaves, and then prune it with 5-fold CV. We can then plot the CV error as a function of tree size (instead of $\alpha$ -- why?):

```r
set.seed(563)
ozone.tree <- tree(ozone~temp+wind, 
                  data=ozone.train, 
                  control = tree.control(nrow(myozone), 
                                         mincut = 2, 
                                         minsize = 4, 
                                         mindev = 0.001)
                  )
cv.ozone <- cv.tree(ozone.tree,K=5)
plot(cv.ozone$dev ~  cv.ozone$size,type= "b", lwd=2, col="red", xlab="Tree Size", ylab="Deviance")
```

Interestingly, a tree with 5 leaves performs best - which corresponds to the original choice:

```r
prune.ozone <- prune.tree(ozone.tree,best=5)
plot(prune.ozone)
text(prune.ozone, pretty =0)
```

# Classification trees (binary or categorical outcome)

* Now allow for $K\geq 2$ number of classes for the response.

* Building a decision tree in this setting is similar to building a regression tree for a quantitative response, but there are two main differences: _the prediction_ and _the splitting criterion_.

**1) The prediction:**   

* In the regression case we use the mean value of the responses in $R_j$ as a prediction for an observation that falls into region $R_j$. 

* For the _classification case_, however, we have two possibilities: 

    * **Majority vote**: Predict that the observation belongs to the most commonly occurring class of the training observations in $R_j$.  
    
    * Estimate the **probability** that an observation $x_i$ belongs to a class $k$, $\hat{p}_{jk}(x_i)$, given as the proportion of class $k$ training observations in region $R_j$. Region $j$ has $N_j$ observations and with $n_{jk}$ observations lying in class k:
    
    $$\hat{p}_{jk} = \frac{1}{N_j} \sum_{i:x_i \in R_j} I(y_i = k)=\frac{n_{jk}}{N_j}.$$ 

**2) The splitting criterion:**  We do not use RSS as a splitting criterion for a qualitative variable. Instead we can use some _measure of impurity_ of the node. For leaf node $j$ and class $k=1,\ldots, K$:

* **Gini index**:
$$
G=\sum_{k=1}^K \hat{p}_{jk}(1-\hat{p}_{jk}) \ ,
$$
which is small if all of the $\hat{p}_{jk}$'s are close to 0 or 1.

* **Cross entropy**:
$$
D=-\sum_{k=1}^K \hat{p}_{jk}\log\hat{p}_{jk} \ .
$$
Since $0\leq\hat{p}_{jk}\leq 1$, it follows that $0\leq -\hat{p}_{jk}\log\hat{p}_{jk}$, with values near zero if $\hat{p}_{jk}$ is close to 0 or 1. 

When making a split in our classification tree, we want to minimize the Gini index or the cross-entropy.

Why don't we just minimize the misclassification error $$E= 1- \max_k{\hat{p}_{jk}} \ ?$$

**A**: 

* The misclassification error is not sufficiently sensitive for tree _growing_.

* The Gini index and Entropy are _measure of impurity_. They are more sensitive to changes in the node probabilities. 

* Example for two classes: Assume we have two classes with 400 nodes each, written as $(400,400)$. Now we can choose between two splits:
    + Split 1: (100,300) and (300,100)  
    + Split 2: (200,400) and (200,0).

Both splits result in 25% misclassification. Which of these splits is better? Probably the second one, because it produces a _pure node_.

* Moreover, The Gini index and Entropy are differentiable (preferred for numerical optimization!)

```r
p=seq(0.001,0.999,length=1000)
gini=2*p*(1-p)
ent=-p*log(p)-(1-p)*log(1-p)
misc= pmin(1-p,p)
plot(p,misc,xlab="p",ylab="",ylim=c(0,0.5),type="l",main="K=2")
lines(p,ent*(0.5/max(ent)),col="red")
lines(p,gini,col="blue")
legend(0.3,0.2,legend=c("misclassification","cross-entropy","Gini"),fill=c("black","red","blue"))
```

## Example: Detection of Minor Head Injury

(Artificial data)

* Data from patients that enter hospital. The aim is to quickly assess whether a patient has a brain injury or not.

* Patients are investigated and (possibly) asked questions.

* _Our job_: To build a good model to predict quickly if someone has a brain injury. The method should be 

    + **easy** to interpret for the medical personell that are not skilled in statistics, and 
    
    + **fast**, such that the medical personell quickly can identify a patient that needs treatment. 

The dataset includes data about 1321 patients and is a modified and smaller version of the (simulated) dataset `headInjury` from the `DAAG` library. 

```r
library(DAAG)
options(digits=6)
headInjury2=read.table("headInjury2.txt",header=TRUE)
colnames(headInjury2)=c("amnesia","bskullf","GCSdecr","GCS.13","GCS.15","risk","consc","oskullf","vomit","brain.injury","age")

n=nrow(headInjury2)
set.seed(1)
train=sample(1:nrow(headInjury2),850)
test=setdiff(1:nrow(headInjury2),train)
headInjury2$brain.injury=factor(headInjury2$brain.injury)
for (i in 1:9) headInjury2[[i]]=as.factor(headInjury2[[i]])
head(headInjury2)
```

### Split with cross-entropy

We use the 850 training samples to get a tree using cross-entropy (deviance):

```r
tree.HIClass=tree(brain.injury~.,
                  data=headInjury2,
                  subset=train,split="deviance")
summary(tree.HIClass)
```

**Remark**: the deviance ($-2\log(L)$) is a scaled version of the cross entropy: 
$$-2\sum_{k=1}^K n_{jk} \log\hat{p}_{jk}\ , \text{where} \quad   \hat{p}_{jk}=\frac{n_{jk}}{N_j}\ , $$ thus `split="deviance"` implies that we split according to the entropy criterion.

```r
plot(tree.HIClass,type="proportional")
text(tree.HIClass,pretty=1)
```

* With `type="proportional" ` (default), the length of branches are proportional to the decrease in impurity.

* The classification tree has two terminal nodes with factor "0" originating from the same branch. Why do we get this "unnecessary" split? 

### Split with Gini index

The same analysis with the Gini index:

```r
tree.HIClassG=tree(brain.injury~.,headInjury2,
                  subset=train,split="gini")
summary(tree.HIClassG)
```

```r
plot(tree.HIClassG,type="proportional")
text(tree.HIClassG,pretty=1)
```

This is a very bushy tree!

### Checking predictions

We also use the classification tree to predict the status of the patients in the test set. 

With the deviance: 

```r
library(caret)
tree.pred=predict(tree.HIClass,headInjury2[test,],type="class")
(confMat <- confusionMatrix(tree.pred,reference=headInjury2[test,]$brain.injury)$table)
1 - sum(diag(confMat))/sum(confMat[1:2, 1:2])
```

With the Gini-index:

```r
tree.predG=predict(tree.HIClassG,headInjury2[test,],type="class")
(confMatG <- confusionMatrix(tree.predG,reference=headInjury2[test,]$brain.injury)$table)
1 - sum(diag(confMatG))/sum(confMatG[1:2, 1:2])
```

## Group discussion

Study the confusion matrices on the previous slides:

* Which algorithm makes more errors?

* Is one type of mistakes more severe than the other? 

* Discuss if/how it is possible to change the algorithm in order to decrease the number of severe mistakes.

## Building a classification tree: Algorithm 8.1

* Like for regression trees, we would like to find classification trees that are good at prediction.

* Algorithm 8.1 (including the pruning step) can be used with misclassification, the Gini index, or cross-entropy instead of the RSS as quality measure.

### Finding an optimal classification tree

We prune the classification tree that was built with the deviance criterion. We use the misclassification error to do so\footnote{While trees should be built using the deviance or the Gini index, pruning is typically done with the misclassification criterion when the aim is predictive accuracy.}:

```r
set.seed(1)
cv.head=cv.tree(tree.HIClass, FUN= prune.misclass)
plot(cv.head$size,cv.head$dev,type="b",
                   xlab="Terminal nodes",ylab="Misclassifications")
```

The function `cv.tree` automatically does $10$-fold cross-validation. `dev` denotes the number of misclassifications.

```r
print(cv.head)
```

* We have done cross-validation on our training set of 850 observations. According to the plot, the number of misclassifications is as low for 7, 8 or 9 nodes, so we choose 7 terminal nodes as the smallest model (lowest variance). 

* We prune the classification tree according to this value:

```r
prune.HIClass=prune.misclass(tree.HIClass,best=7) 
```

```r
plot(prune.HIClass)
text(prune.HIClass,pretty=1)
```

$\rightarrow$ No unnecessary splits left, and we have a simple and interpretable decision tree. 

How is the predictive performance of the model affected?

```r
tree.pred.prune <- predict(prune.HIClass,headInjury2[test,],type="class")
(confMat <- confusionMatrix(tree.pred.prune,headInjury2[test,]$brain.injury)$table)
1 - (sum(diag(confMat))/sum(confMat[1:2,1:2]))
```

$\rightarrow$ We see that the misclassification rate is as small as before, thus the pruned tree is as good as the original tree for the test data.

$\rightarrow$ If you want, you can apply the same pruning procedure to the very bushy Gini-grown tree.

### Questions:

Discuss the _bias-variance tradeoff of a regression tree_ when increasing/decreasing the number of terminal nodes, i.e: 

* What happens to the bias?
* What happens to the variance of a prediction if we reduce the tree size? 

### Other issues: Categorical predictors

* If a predictor has $q$ unordered levels, there are $2^{q-1} -1$ possible partitions into groups.  

* In case of a _binary outcome_: 
    * Order the predictor classes according to the proportion in outcome class 1. 
    * Then, use it as an ordered predictor.  
$\rightarrow$ This gives the optimal split according to Gini index.

* For _multicategory and continuous outcomes_, this is not possible. Try to avoid predictors with very many levels!

### Other issues: Missing predictor values

* Predictor variables can have missing instances. Discarding those observations can lead to serious depletion of the training data.

* We can _impute_ missing values, e.g., by taking the mean over the observed values. 

**Two better solutions for trees:** 

* Idea 1: Make a separate category for "missing". 

* Idea 2: Use the observed values in each variable for finding the split, but also create so-called "surrogate splits": Store next-best split options using other variables at that point. 

# Trees versus linear models

* What if we have $x_1$ and $x_2$ and the true class boundary (two classes) is linear in $x_1$, $x_2$ space. How can we do that with our binary recursive splits?

* What about a rectangular boundary?

![Linear boundary ISL Figure 8.7](../../ISLR/Figures/Chapter8/87.png){width=50%}

# From trees to forests

**Advantages (+)**

* Trees automatically select variables.
* Tree-growing algorithms scale well to large $n$, growing a tree greedily.
* Trees can handle mixed features (continouos, categorical) seamlessly.
* Small trees are easy to interpret and explain to people.
* Some believe that decision trees mirror human decision making.
* Trees can be displayed graphically.

**Disadvantages (-)**

* Large trees are not easy to interpret.
* Trees do not generally have good prediction performance (high variance).
* Trees are not very robust, a small change in the data may cause a large change in the final estimated tree.

## What is next?

Several of the above listed disadvantages can be addressed by 

* **Bagging**: grow many trees (from bootstrapped data) and average - to get rid of the non-robustness and high variance by averaging.

* **Random forests**: inject more randomness (and even less variance) by just allowing a random selection of predictors to be used for the splits at each node.

* **Boosting**: make one tree, then another based on the residuals from the previous, repeat. The final predictor is a weighted sum of these trees.  
$\rightarrow$ See more extensive discussion in module 9.

In addition:

* **Variable importance** - to see which variables make a difference (now that we have many trees).

## Leo Breiman - the inventor of CART, bagging and random forests

See Wikipedia entry: https://en.wikipedia.org/wiki/Leo_Breiman

> Breiman's work helped to bridge the gap between statistics and computer science, particularly in the field of machine learning. His most important contributions were his work on classification and regression trees and ensembles of trees fit to bootstrap samples. Bootstrap aggregation was given the name bagging by Breiman. Another of Breiman's ensemble approaches is the random forest. 

# Bagging

* Decision trees often suffer from high variance. By this we mean that the trees are sensitive to small changes in the predictors: If we change the observation set, we may get a very different tree. 

* Another way to understand "high variance" is that, if we split our training data into two parts and fit a tree on each, we might get rather different decision trees.

* To reduce the variance of decision trees we can apply _bootstrap aggregating_ (_bagging_), invented by Leo Breiman in 1996 [@Breiman1996].

### High variance in decision trees -- Illustration

* Let's draw a new training set (`train2`) and use the deviance criterion to grow the tree. 

* The two classification trees contructed from $850$ different random subsets each:

```r
set.seed(33)
N=dim(headInjury2)[1]
train2=sample(1:N,850)
tree.HIClass2=tree(brain.injury~.,
                   data=headInjury2,subset=train2,split="deviance") 
par(mfrow=c(1,2))
plot(tree.HIClass);text(tree.HIClass,pretty=0)
plot(tree.HIClass2);text(tree.HIClass2,pretty=0)
```

* We get two rather different trees.

## Recall: Variance for independent datasets

* Assume we have $B$ _i.i.d._ observations of a random variable $X$ each with the same mean and with variance $\sigma^2$. We calculate the mean $\bar{X} = \frac{1}{B} \sum_{b=1}^B X_b$. The variance of the mean is
$$\text{Var}(\bar{X}) = \text{Var}\Big(\frac{1}{B}\sum_{b=1}^B X_b \Big) = \frac{1}{B^2} \sum_{b=1}^B \text{Var}(X_b) = \frac{\sigma^2}{B}.$$
By averaging we get reduced variance. This is the basic idea!

* For decision trees, if we have $B$ training sets, we could estimate $\hat{f}_1({\boldsymbol x}),\hat{f}_2({\boldsymbol x}),\ldots, \hat{f}_B({\boldsymbol x})$ and average them as
$$ \hat{f}_{avg}({\boldsymbol x})=\frac{1}{B}\sum_{b=1}^B \hat{f}_b({\boldsymbol x}) \ .$$

However, we do not have many independent dataset - so we use _bootstrapping_ to construct $B$ datasets.

## Bagging = Bootstrap aggregating

* Bootsrapping: Draw _with replacement_ $n$ observations from our sample - and that is our first _bootstrap sample_.

* We repeat this $B$ times and get $B$ bootstrap samples - that we use as our $B$ datasets.

* For each bootstrap sample $b=1,\ldots, B$ we construct a decision tree, $\hat{f}^{*b}(x)$. 

* For a _regression tree_, we take the average of all of the predictions and use this as the final result:
$$
\hat{f}_{bag}(x)=\frac{1}{B}\sum_{b=1}^B \hat{f}^{*b}(x).
$$

* For a _classification tree_ we record the predicted class (for a given observation $x$) for each of the $B$ trees and use the most occurring classification (_majority vote_) as the final prediction.

## Out-of-bag error estimation

* Recall (module 5) that the probability that an observation is in the bootstrap sample is approximately $1-e^{-1}$=`r round(1-exp(-1),2)` ($\approx 2/3$).

* When an observation is left out of the bootstrap sample it is not used to build the tree, and we can use this observation as a part of a "test set" to measure the predictive performance and error of the fitted model, $\hat{f}^{*b}(x)$. 

* For $B$ bootstrap samples, observation $i$ will be outside the bootstrap sample in approximately $B/3$ of the fitted trees. Obtain a single prediction by averaging (regression problem) or taking the majority vote/mean probability (classification problem) of the $B/3$ predictions.

**Terminology:**

* The observations left out are referred to as the _out-of-bag_ (OOB) observations.

* The measured error of the $B/3$ predictions is called the _out-of-bag error_. 

## Example

We can do bagging by using the function `randomForest()` in the `randomForest` library.

```r
library(randomForest)
set.seed(1)
r.brain.bag <- randomForest(brain.injury~.,
                 data=headInjury2,subset=train,
                 mtry=10,ntree=500,importance=TRUE)
r.brain.bag$confusion
1-sum(diag(r.brain.bag$confusion))/sum(r.brain.bag$confusion[1:2,1:2])
```

The variable `mtry=10` because we want to consider all $10$ predictors in each split of the tree. The variable `ntree=500` because we want to average over $500$ trees.

Predictive performance of the bagged tree on unseen test data:

```r
yhat.bag=predict(r.brain.bag,newdata=headInjury2[test,])
misclass.bag=table(yhat.bag,headInjury2[test,]$brain.injury)
print(misclass.bag)
1-sum(diag(misclass.bag))/(sum(misclass.bag))
```

Note: The misclassification rate has increased slightly for the bagged tree (as compared to our previous full and pruned tree). **Typically, we would expect an improvement!**

## Variable importance plots

* Drawback of bagging: It becomes difficult to interpret the results. Instead of having one tree, the resulting model consists of many trees (it is an _ensemble method_). 

* _Variable importance plots_ show _the relative importance of the predictors:_ the predictors are sorted according to their importance, such that the top variables have a higher importance than the bottom variables. 

* There are in general two types of variable importance plots: 

    * variable importance based on decrease in node impurity.
    * variable importance based on randomization.

### Variable importance based on node impurity

*Importance* relates to _total decrease in the node impurity, over splits for a predictor_.

**Regression trees:** 

* The importance of each predictor is calculated using the RSS. 
* The algorithm records the total amount that the RSS is decreased due to splits for each predictor (there may be many splits for one predictor for each tree). 
* This decrease in RSS is then averaged over the $B$ trees.

**Classification trees:** 

* The importance of each predictor is calculated using the Gini index. 
* The importance is the mean decrease (over all $B$ trees) in the Gini index by splits of a predictor.

**R-hint**: `varImpPlot()` function (or `importance`) in `randomForest` with `type=2`.

### Variable importance based on randomization

Variable importance based on randomization is calculated using the OOB sample. 

* Computations are carried out for one bootstrap sample at a time. 
* Each time a tree is grown the OOB sample is used to test the predictive power of the tree. 
* Then for one predictor at a time, repeat the following: 
     + permute the OOB observations for the $j$th variable $x_j$ and calculate the new OOB error. 
     + If $x_j$ is important, permuting its observations will decrease the predictive performance. 
* The difference between the two is averaged over all trees (and normalized by the standard deviation of the differences). 

**R-hint**: `varImpPlot()` (or `importance()`) function with `type=1`.

### Example 1: Auto data (regression tree)

The data relates fuel consumption (`mpg`) to 10 aspects of automobile desing and performance.

```r
data(mtcars)
mtcars.rf <- randomForest(mpg ~ ., data=mtcars, ntree=1000, 
                          keep.forest=FALSE,
                          mtry=10,
                          importance=TRUE)
varImpPlot(mtcars.rf,pch=20,main="")
```

(Randomization: left, node purity: right)

### Example 2: Head injury (classification tree)

```r
varImpPlot(r.brain.bag,pch=20)
```

(Randomization: left, node purity: right)

# Random Forests

* If there is a strong predictor in the dataset, the decision trees produced by each of the bootstrap samples in the bagging algorithm becomes very similar: Most of the trees will use the same strong predictor in the top split. 

* We have seen this for our example trees for the minor head injury example, the predictor `GCS.15` was chosen in the top split every time. This is probably the case for a large amount of the bagged trees as well.

* Not optimal, as we get $B$ trees that are highly correlated. $\rightarrow$ No large reduction in variance by averaging $\hat{f}^{*b}(x)$ when the correlation between the trees is high. 

## The effect of correlation on the variance of the mean

* The variance of the average of $B$ observations of _i.i.d._ random variables $X$, each with variance $\sigma^2$ is $\frac{\sigma^2}{B}$. 

* But if we have $B$ _i.i.d._ observations of a random variable $X$ with a positive correlation $\rho$ such that $\text{Cov}(X_i, X_j) = \rho \sigma^2, \quad i \neq j,$ then

$$ \begin{aligned} \text{Var}(\bar{X}) &= \text{Var}\Big( \frac{1}{B}\sum_{i=1}^B X_i \Big) =\\
&= \sum_{i=1}^B \frac{1}{B^2} \text{Var} (X_i) + 2 \sum_{i=2}^B \sum_{j=1}^{i-1} \frac{1}{B^2} \text{Cov} (X_i, X_j) \\
&= \frac{1}{B} \sigma^2 + 2 \frac{B(B-1)}{2}\frac{1}{B^2} \rho \sigma^2 \\
&= \frac{1}{B} \sigma^2 + \rho \sigma^2 - \frac{1}{B} \rho \sigma^2 \\
&= \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2  
\end{aligned}$$

Check: $\rho=0$ and $\rho=1$? 

* _Random forests_ provide an improvement over bagged trees by a small tweak that _decorrelates the trees_. This reduces the variance when we average the trees.

* As in bagging, we build a number of decision trees on bootstrapped training samples.

* But each time a split in a tree is considered, a _random selection_ of $m$ predictors is chosen as split candidates from the full set of $p$ predictors. The split is allowed to use only one of those m predictors.

* A fresh selection of $m$ predictors is taken at each split, typically 
    * $m\approx \sqrt p$ (classification),  
    * $m=p/3$ (regression).

Generally, $m$ should be chosen small if the predictors are very correlated.

### How many trees?

* The number of trees, $B$, is not a tuning parameter, and the best is to choose it large enough (as large as "necessary"). An increase in $B$ will not lead to overfitting.

* Increasing $B$ will not change the goodness of fit measure.

* To find out which number $B$ is sufficient, we do _not_ need to run cross-validation, but can again use the OOB error (works best for bagging and random forests).

ISLR Figure 8.10, gene expression dataset with 15 classes and 500 predictors:

![ISLR Figure 8.10, gene expression dataset with 15 classes and 500 predictors](../../ISLR/Figures/Chapter8/810.png){width=80%}

## Example

We decorrelate the brain injury classification trees by using the `randomForest()` function again, but this time we set `mtry=3` (instead for `mtry=10`). This means that the algorithm only considers three of the predictors in each split. We choose $3$ because we have $10$ predictors in total and $\sqrt{10}\approx 3$. 

```r
set.seed(1)

r.brain.rf = randomForest(brain.injury~.,
                data=headInjury2,subset=train,
                mtry=3,ntree=500,importance=TRUE)
```

We check the predictive performance as before, using the test set:

```r
yhat.rf=predict(r.brain.rf,newdata=headInjury2[test,])

misclass.rf=table(yhat.rf,headInjury2[test,]$brain.injury)
print(misclass.rf)

1-sum(diag(misclass.rf))/(sum(misclass.rf))
```

The misclassification rate is slightly decreased compared to the bagged tree (and to the pruned tree).

By using the `varImpPlot()` function we can study the importance of each predictor.

```r
varImpPlot(r.brain.rf,pch=20, main="")
```

As expected `GCS.15` (GCS=15 at 2h) is a strong predictor along with `bskullf` (basal skull facture) and `age`. This means that most of the trees will have these predictors in the top split.

# Gradient boosting

* _Boosting_ is an alternative approach for improving the predictions resulting from a so-called _weak learner_.

* Typically, boosting is applied to regression and classification trees, because they are (on their own) not very good prediction methods.

* Main idea: The trees are grown _sequentially_ so that each tree is grown using information from the previous tree. 

* Since each tree improves the previous model, the number of trees $B$ is a tuning parameter (in contrast to bagging and random forest).

$\rightarrow$ We will look at gradient boosting in Module 9!

# Example: Boston dataset

(ISLR book, Sections 8.3.2 to 8.3.4.)

Remember the dataset: The aim is to predict the median value of owner-occupied homes (in 1000\$)

We first run through trees, bagging and random forests - before arriving at boosting. 

```r
library(MASS)
set.seed(1)
train = sample(1:nrow(Boston), nrow(Boston)/2)
head(Boston)
```

### Regression tree

```r
tree.boston=tree(medv~.,Boston,subset=train,control = tree.control(nrow(Boston), mindev = 0.005)) 
summary(tree.boston); 
```

```r
plot(tree.boston)
text(tree.boston,pretty=0)
```

Remember: 

* The `tree()` function has a built-in default stopping criterion. 
* You can change this with the `control` option, for example by setting `control = tree.control(mincut = 2, minsize = 4, mindev = 0.001)`. Here we used `mindev=0.005`. 

### Need to prune?

```r
cv.boston=cv.tree(tree.boston)
plot(cv.boston$size,cv.boston$dev,type='b')
```

It looks like a tree with 6 leaves would work well.

### Pruning

So we are pruning to a 6-node tree here:

```r
prune.boston=prune.tree(tree.boston,best=6)
plot(prune.boston)
text(prune.boston,pretty=0)
```

### Test error for full tree

We calculate the test error for the pruned tree:

```r
yhat=predict(prune.boston,newdata=Boston[-train,])
boston.test=Boston[-train,"medv"]
plot(yhat,boston.test, pch=20)
abline(0,1)
```

```r
mean((yhat-boston.test)^2)
```

### Bagging

Remember: For bagging you can use the `randomForest()` function, but include all variables (here `mtry=13`).

```r
library(randomForest)
set.seed(1)
bag.boston=randomForest(medv~.,data=Boston,subset=train,mtry=13,importance=TRUE)
bag.boston
```

### Test error for bagged tree

```r
yhat.bag = predict(bag.boston,newdata=Boston[-train,])
plot(yhat.bag, boston.test,pch=20)
abline(0,1)
```

```r
bag.boston=randomForest(medv~.,data=Boston,subset=train,mtry=13,ntree=25)
yhat.bag = predict(bag.boston,newdata=Boston[-train,])
mean((yhat.bag-boston.test)^2)
```

### Random forest

Let's go from bagging to a random forest, using 4 ($\approx p/3$) randomly selected predictors for each tree:

```r
set.seed(1)
rf.boston=randomForest(medv~.,data=Boston,subset=train,mtry=4,importance=TRUE)
yhat.rf = predict(rf.boston,newdata=Boston[-train,])
mean((yhat.rf-boston.test)^2)
```

It's interesting to see how the prediction error further decreased with respect to simple bagging.

### Variable importance

```r
importance(rf.boston)
```

Interpretation?

And the variable importance plots

```r
varImpPlot(rf.boston,main="")
```

To understand what this means, please check again the meaning of the variables by typing `?Boston`. 

### Boosting

See Module 9.

# Further reading

* [Videoes on YouTube by the authors of ISL, Chapter 8](https://www.youtube.com/playlist?list=PL5-da3qGB5IB23TLuA8ZgVGC8hV8ZAdGh), and corresponding [slides](https://lagunita.stanford.edu/c4x/HumanitiesScience/StatLearning/asset/trees.pdf).

* [Solutions to exercises in the book, chapter 8](https://rstudio-pubs-static.s3.amazonaws.com/65564_925dfde884e14ef9b5735eddd16c263e.html)

# References
