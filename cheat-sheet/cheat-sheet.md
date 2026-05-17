### m09 Boosting
Will want to go through the slide deck and fetch all formulas.
/Users/andersbekkevard/dev/school/stat_laer/context/archive/modules/9TreeBoosting/9TreeBoosting.Rmd

### Derivations
- Utledning Corr^2 = R2 for enkel linreg (this can be found somewhere in the repo)
- Utledning av (12.18), og "Show that the algorithm below is guaranteed to decrease the value of the objective at each step"
- OLS = MLE på vektorform og univariate

### Vector math
- Scalar: Kan transponere (Anders)
- Regneregler for vektorderivasjon. d/db(bTXb) og d/db(aTb). Nødvendige for OLS-utledning.
- Regneregler for E[], Var[], Cov[] etc for matriser, som grunnleggende byggeblokker
- Formel for å finne cov-matrise på X:
    - Center (-1xmeanT)
    - (Her ville man standardisert for PCA)
    - 1/(n-1) X_cT X_c


### Misc
- Which models allow for p>n
- Df for all relevant models. Specifically "moving beyond linearity"
- When in the course do we use standardization, and when do we use centering (PCA, ridge/lasso, nns, etc etc.). This NEEDS to be an exhaustive list in order for Anders to trust it.
- Logistic = sigmoid, and its inverse is the logit (And how they all look)
- What are all the hyperparameters for each of the models we use in the course (e.g., for trees they vary, different for AdaBoost, Random Forests and XGBoost). In addition, we should have a tiny note for each on how it affects the bias variance tradeoff.
- En enkel tabell på når subset selection vs ridge vs lasso vs elasticnet er best, and also for PCA/PCR/PLS


### Prewritten explanation
- The following question and explanation will (most likely) come on the exam since the prof loves it. Therefore I need to have it on lock. This is a verbose version, but this general idea or argument must be here.
(1 %) In one or two sentences, give a reason why one might prefer the label bias–variance decomposition over bias–variance trade-off. (You may, for example, appeal to the squared form of the bias term, the role of a clever model choice such as regularization, or the over-parameterized regime in which both bias and variance can simultaneously decrease.) [!Snakk med chat om denne også, så du kan besvare forskjellige formuleringer]

Solution (1 %) (ii) “Decomposition” is preferred because the identity above holds exactly for every estimator — it is an algebraic equality, not a constraint forcing one quantity to grow when the other shrinks. The label “trade-off” is misleading because (a) the bias term is squared, so a small absolute increase in bias contributes very little to MSE while variance can shrink a lot, and (b) changing the model class (e.g. adding regularization, or moving to the over-parameterized / double-descent regime of large neural networks) can lower variance without a matching increase in bias. Any one of these two angles, stated clearly, earns the full 1 P.

### Book lookups
See book-lookups.md


### Statsbook (Anders)
- Regneregler Cov, Var, osv, bare litt snacks fra statheftet. Spesielt, summen for variansen til en sum av korrelerte variabler.
