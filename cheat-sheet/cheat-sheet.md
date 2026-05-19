### m09 Boosting

### Derivations
- OLS = MLE på vektorform og univariate
- Utledning av Bias Variance Tradeoff



### Misc
- Df for all relevant models. Specifically "moving beyond linearity"
- Logistic = sigmoid, and its inverse is the logit (And how they all look)
- En enkel tabell på når subset selection vs ridge vs lasso vs elasticnet er best, and also for PCA/PCR/PLS


### Prewritten explanation
- The following question and explanation will (most likely) come on the exam since the prof loves it. Therefore I need to have it on lock. This is a verbose version, but this general idea or argument must be here.
(1 %) In one or two sentences, give a reason why one might prefer the label bias–variance decomposition over bias–variance trade-off. (You may, for example, appeal to the squared form of the bias term, the role of a clever model choice such as regularization, or the over-parameterized regime in which both bias and variance can simultaneously decrease.) 

Solution (1 %) (ii) “Decomposition” is preferred because the identity above holds exactly for every estimator — it is an algebraic equality, not a constraint forcing one quantity to grow when the other shrinks. The label “trade-off” is misleading because (a) the bias term is squared, so a small absolute increase in bias contributes very little to MSE while variance can shrink a lot, and (b) changing the model class (e.g. adding regularization, or moving to the over-parameterized / double-descent regime of large neural networks) can lower variance without a matching increase in bias. Any one of these two angles, stated clearly, earns the full 1 P.

### Book lookups
See book-lookups.md


### Statsbook (Anders)
- Regneregler Cov, Var, osv, bare litt snacks fra statheftet. Spesielt, summen for variansen til en sum av korrelerte variabler.
