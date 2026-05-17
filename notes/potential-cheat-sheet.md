# Potential cheat sheet
### Ideas for cheat sheet entries
En enkel tabell på når subset selection vs ridge vs lasso vs elasticnet er best, og kanskje også PCA/PCR/PLS
- Df for "moving beyond linearity", evt finne ut hvor det står i boken.
- Df generelt
- Which models allow for p>n
- Utledning Corr^2 = R2 for enkel linreg
- Utledning av (12.18), og "Show that the algorithm below is guaranteed to decrease the value of the objective at each step"
- Ulike former for optimeringsproblemer: (argmin, minimize subject to, etc)
- OLS = MLE på vektorform?
- Adaboost slide 13 m09-boosting-slides
- Regneregler Cov, Var, osv, bare litt snacks fra statheftet. Spesielt, summen for variansen til en sum av korrelerte variabler.
- Hyperparametre/tunings for alle modeller, spesielt trær. Også hvordan de funker bias/variance -teknisk. Også "Stochastic GBM".
- Algo 10.3 i ESL
- Noe på AUC
- Sigmoid = (Logistic function), samt hva logit er (logistic^-1)

- **Viktig**, boosting boosting boosting. Ikke i ISLP. Og teknisk. Og funker som faen. Må være på formelark.
- Regneregler for vektorderivasjon. d/db(bTXb) og d/db(aTb). Nødvendige for OLS-utledning.
- Regneregler for E[], Var[], Cov[] etc for matriser, som grunnleggende byggeblokker
- Scalar: Kan transponere
- Når må man dele på std, og når må man trekke fra mu? PCA, Ridge/Lasso
- Formel for å finne cov-matrise på X:
    - Center (-1xmeanT)
    - (Her ville man standardisert for PCA)
    - 1/(n-1) X_cT X_c



### Lærdommer/tips
- Alle antagelsene "by idependence", "by linearity if E[]" etc.

(1 %) In one or two sentences, give a reason why one might prefer the label bias–variance decomposition over bias–variance trade-off. (You may, for example, appeal to the squared form of the bias term, the role of a clever model choice such as regularization, or the over-parameterized regime in which both bias and variance can simultaneously decrease.) [!Snakk med chat om denne også, så du kan besvare forskjellige formuleringer]

Solution (1 %) (ii) “Decomposition” is preferred because the identity above holds exactly for every estimator — it is an algebraic equality, not a constraint forcing one quantity to grow when the other shrinks. The label “trade-off” is misleading because (a) the bias term is squared, so a small absolute increase in bias contributes very little to MSE while variance can shrink a lot, and (b) changing the model class (e.g. adding regularization, or moving to the over-parameterized / double-descent regime of large neural networks) can lower variance without a matching increase in bias. Any one of these two angles, stated clearly, earns the full 1 P.



