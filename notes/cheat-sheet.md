Har boka. Burde ha liste over:
- Ting Benji mener er viktig på eksamen/som vil bli spurt om
- Liste over ting som ikke kommer så tydelig frem fra boken/som ville krevd mye lesing. Feks en enkel tabell på når subset selection vs ridge vs lasso vs elasticnet er best, og kanskje også PCA/PCR/PLS. +Kort om hva benji mener om hver (for sjarmpoeng):
- Df for "moving beyond linearity", evt finne ut hvor det står i boken.
- Df generelt
- Which models allow for p>n
- Utledning Corr^2 = R2 for enkel linreg
- Utledning av (12.18)
- Formlene for f i forskjellige typer nns.
- Ulike former for optimeringsproblemer: (argmin, minimize subject to, etc)
- OLS = MLE på vektorform?
- Adaboost slide 13 m09-boosting-slides

- **Viktig**, boosting boosting boosting. Ikke i ISLP. Og teknisk. Og funker som faen. Må være på formelark.
- Regneregler for vektorderivasjon. d/db(bTXb) og d/db(aTb). Nødvendige for OLS-utledning.
- Regneregler for E[], Var[], Cov[] etc for matriser, som grunnleggende byggeblokker
- Scalar: Kan transponere
- Når må man dele på std, og når må man trekke fra mu? PCA, Ridge/Lasso
- Formel for å finne cov-matrise på X:
    - Center (-1xmeanT)
    - (Her ville man standardisert for PCA)
    - 1/(n-1) X_cT X_c


Whats up med:
https://www.deeplearningbook.org/?





(1 %) In one or two sentences, give a reason why one might prefer the label bias–variance decomposition over bias–variance trade-off. (You may, for example, appeal to the squared form of the bias term, the role of a clever model choice such as regularization, or the over-parameterized regime in which both bias and variance can simultaneously decrease.) [!Snakk med chat om denne også, så du kan besvare forskjellige formuleringer]

Solution (1 %) (ii) “Decomposition” is preferred because the identity above holds exactly for every estimator — it is an algebraic equality, not a constraint forcing one quantity to grow when the other shrinks. The label “trade-off” is misleading because (a) the bias term is squared, so a small absolute increase in bias contributes very little to MSE while variance can shrink a lot, and (b) changing the model class (e.g. adding regularization, or moving to the over-parameterized / double-descent regime of large neural networks) can lower variance without a matching increase in bias. Any one of these two angles, stated clearly, earns the full 1 P.
