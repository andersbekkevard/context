---
title: The Long Tuesday of Observation i = 42
description: A humorous short story in which a single sample point is bootstrapped, cross-validated, pruned, and embedded through every dataset in ISLP.
---

# The Long Tuesday of Observation $i = 42$

> [!note] What is this
> A short story. Not exam-prep. The narrator is one sample point — observation number forty-two — and over the course of one absurd Tuesday they are passed through every dataset in ISLP. If a parameter sounds oddly specific, that is because it is. Every variable name in this story is real and lives in some `.csv` shipped with the `ISLP` package.

---

## 1. Morning — `Wage`

I wake up in the **central Atlantic region of the United States**, which is a strange way to wake up. I am a male, age 42, with a `wage` of 104.92 thousand dollars and an `education` level of *some college*. There are 2,999 of us in the room. We are all male. Nobody told me my entire demographic universe would be 3,000 men staring at each other; I find this oddly Norwegian.

A degree-4 polynomial swings across the room from age 18 to age 80. I duck. Somewhere near my forehead the curve peaks. Seventy-nine of us are "high earners," and the confidence intervals around them are *wide* — like, embarrassingly wide, like the prof's metaphor for the lasso, wide. I am not a high earner. I am the 95% confidence interval's emotional support point.

A man called Hastie walks past with a smoothing spline in one hand and a step function in the other. He nods at me. Nobody speaks. This is `Wage`.

## 2. Commute — `Auto`, then `Bikeshare`

I leave for work. My car is a 1979 four-cylinder with a `displacement` of 151, a `horsepower` of 90, a `weight` of 2,670 pounds, an `acceleration` of 16.0 seconds 0–60, and an `mpg` of 21. The `origin` is American. The name on the title says "Plymouth Volaré," which I did not choose, and which I resent.

Halfway through the commute the engine collapses into a scatterplot. There is a clean nonlinear relationship between `horsepower` and `mpg` that everyone has tried to explain with a degree-2 polynomial and which I now have to walk out of.

I switch to a Capital Bikeshare bike. It is hour 17 of the day in Washington, D.C. The `workingday` flag is 1, the `weathersit` is "clear," the `temp` is mild, the `hum` is 62, the `windspeed` is irrelevant. There are exactly enough bikers, according to a Poisson GLM, that I should not crash. I crash anyway. The residual was overdispersed.

## 3. The Office — `Advertising`, `Carseats`, `OJ`

The office sells **TV ads, radio ads, and newspaper ads** to 200 markets. My boss, a marginal effect, walks in waving a slide that shows `Sales` regressed on `TV` and `Radio` and `Newspaper`, and explains, again, that `Newspaper` is **not significant** once you control for the other two, and that we all need to stop pretending otherwise. He has been making this slide since 2001.

I am moved to the **car-seat division**, where I oversee 400 stores. We have a `Price` of \$117, a `CompPrice` of \$125, an `Income` of \$72k in the local area, a `ShelveLoc` of *Medium*, and `Sales` that I cannot, despite three classification trees and a random forest of 500 of them, get to budge.

For lunch I am asked to choose between **Citrus Hill** and **Minute Maid**. The choice is recorded. One of 1,070 such choices. The brand-loyalty variable for Citrus Hill in my household this week is 0.6, which the prof would say is borderline but interpretable. I pick Minute Maid. Somewhere a support vector machine sighs.

## 4. The Bank — `Default`, `Credit`, `Caravan`

After lunch I go to the bank to ask about my `balance`. I am one of 10,000 customers. A logistic regression looks at my balance, my income, and the fact that I am not a `student`, and decides that my probability of `default` is approximately 0.0021. The teller smiles. We both know that if I had a balance of \$2,000 instead of \$833, this smile would be a different smile.

In the next room my `Credit` profile is being computed. My `Limit` is \$3,606, my `Rating` is 283, I carry 4 `Cards`, my `Age` is 42 (consistent across datasets, finally), I am `Married` = Yes, my `Region` is South. The model says my `Balance` should be \$520. I check my pocket. \$519. The residual is one dollar. I drop it on the floor and the bank closes.

On the way out a man tries to sell me **caravan insurance**. I am one of 5,822 prospects with 85 sociodemographic predictors attached to me. The base rate of buying caravan insurance is 6%. I do not buy it. He marks me as `Purchase = No` and walks away whistling.

## 5. The Trading Floor — `Smarket`, `Weekly`, `Portfolio`, `NYSE`, `Fund`

A subway grate exhales and deposits me on the **floor of the S&P 500, 2001 to 2005**. There are 1,250 days here. `Lag1` is yesterday's return, `Lag2` is the day before, etc. People are screaming Lag-numbers at each other. None of the lags predict `Direction`. We all know this. We scream anyway.

I shuffle into the `Weekly` data, where they have been screaming for 21 years (1990–2010) on a weekly cadence, and where, in a famous lecture, the prof shows that logistic regression with `Lag2` does *slightly* better than chance, which inflames the room.

In the corner a quiet statistician calculates $\alpha$ using `Portfolio`, with $n = 100$ paired returns $X$ and $Y$. He bootstraps 1,000 times. The standard error of $\hat{\alpha}$ comes out near 0.083. He nods. This is the only honest number on the floor.

The `NYSE` ticker overhead shows roughly 6,000 days of returns. An RNN is trying to predict the next one. It fails gracefully, with an $R^2$ of about 0.42 on log-volatility — which, in deep learning, is not a failure but a small celebration.

Above the floor, on a mezzanine, 50 `Fund` managers are being multiple-tested. Their monthly returns are being run through the Bonferroni correction. Forty-six of them lose their "skilled" labels in the process. They take it well. One of them cries. The Benjamini–Hochberg procedure walks over and gently lets two of them back in.

## 6. Afternoon — `Boston`, `College`, `Hitters`, `Income`

I take the train to **Boston**, a city that is exactly 506 census tracts and nothing else. My tract has `crim` = 0.06, `rm` = 6.4 rooms per dwelling, `lstat` = 8.3% lower-status, `nox` = 0.46. The `medv` is \$22.5 thousand, which is 1970s money. I am told this is a median value and that I should not take it personally. I take it personally.

I visit a **college**. There are 777 of them and they all look the same. This one accepts 1,924 of 2,186 `Apps`, enrolls 512, charges \$11,000 `Outstate`, has a `Top10perc` of 22%, and a `Grad.Rate` of 73%. The college is `Private` = Yes. The admissions officer hands me a brochure that is just a PCA biplot. I leave.

I cross a baseball diamond. **322 hitters** are warming up. One of them has a `Salary` of \$2,127.33 thousand, 173 `Hits`, 39 `HmRun`, 110 `RBI`, 14 `Years` in the majors, 587 `AtBat`. A regression tree splits him on `Years < 4.5`, and he goes to the *low-salary* leaf because someone forgot to update the data. He is furious. The prof would say: **prune the tree.** Someone prunes the tree. He is furious about that, too.

On a park bench I sit next to a man whose entire being is determined by `Education` and `Seniority`. He has 30 friends. They sit on a hilly $\mathbb{R}^2$ surface that has been generated to look like a teaching figure. He says, *I am `Income`. I have always been `Income`*. We watch the sunset.

## 7. Evening — `Heart`, `BrainCancer`, `Publication`

I begin to feel my **`ChestPain`** more acutely. `RestBP` 130, `Chol` 246, `MaxHR` 173, `Sex` male, `Age` 42 (still 42, the algorithm respects this), `ExAng` no. The ROC curve, drawn in the air above the bed, has an AUC of 0.84. The doctor says it is `AHD` = No. The doctor is using a classification tree. I do not entirely trust the classification tree.

Two doors down, in the **`BrainCancer`** ward, 88 patients are being followed. A Kaplan–Meier curve is taped to the wall, step-function, slow descent. Some of the patients are censored. The prof reminds us, gently, that **censored does not mean dead.** It means we lost track. He says this with the seriousness it deserves.

In the basement of the hospital, 244 clinical trials are waiting to be **`Publication`**-ed. The ones with positive results are being published in months. The ones with null results are being held back. A Cox model with `posres` as the key covariate is exposing the entire bias structure. The journal editors look uncomfortable.

## 8. Late Evening — `NCI60`, `Khan`, `USArrests`

I find myself in a lab. There are **64 cancer cell lines** in front of me, each described by 6,830 gene expressions. I am supposed to cluster them. I run K-means with $K = 4$. Hierarchical clustering disagrees. PCA reduces them to 2 dimensions and they sort themselves loosely by cancer type. The melanoma cluster is the cleanest. I write this down.

Next door, in the **`Khan`** lab, four small round blue cell tumor types — BL, EWS, NB, RMS — are arrayed on 2,308 genes. The training set has 63 samples, the test set 20. A linear SVM separates all four classes with zero training error, and the prof would say *of course it does, $p \gg n$*. I nod. I am 42, I have been nodding all day.

I leave the lab and walk past **50 states** of the United States. They are arranged in a biplot. `Murder` and `Assault` and `Rape` lie almost on top of each other along the first principal component. `UrbanPop` sticks out along the second, alone, defiant. California is in the corner being unusual. North Dakota is in the other corner being unusual in the opposite direction. The first two PCs explain 86.75% of the variance. This is, frankly, more than I expected.

## 9. Night — `MNIST`, `CIFAR100`, `IMDb`

I get sleepy. I close my eyes and see a **handwritten 7**. Then a **2**. Then a **0**. The MNIST stream begins. 60,000 of them, 28×28, grayscale. A two-layer dense network gets 98% on the test set. A convolutional net gets 99.5%. The prof says: *yes, but on this one, even logistic regression gets 92%, so calm down.*

Then color floods in: 60,000 32×32 RGB **CIFAR100** images. Apples, aquarium fish, baby, bear, beaver, bed, bee, beetle, bicycle, bottle, bowl, boy, bridge, bus, butterfly, camel, can, castle, caterpillar, cattle… I am being shown 100 classes. I get worm and snake confused. The model also gets worm and snake confused. This is a comfort.

Finally an **IMDb review** appears, padded to length 500, every word an integer index. Sentiment: positive. Star rating: implicit. A single LSTM cell labors through it. The hidden state grows tired. So do I.

## 10. The Bootstrap — `Portfolio` (again)

In my dream I am resampled with replacement, 1,000 times. Sometimes I appear twice in the same dataset. Sometimes I do not appear at all. The probability I appear at least once in any given bootstrap sample is $1 - (1 - 1/n)^n \to 1 - e^{-1} \approx 0.632$. The prof has said this number so many times that it has become a kind of prayer. *Point six three two.* I whisper it into the pillow.

The standard error of my $\hat{\alpha}$ is converging. Somewhere, $\alpha$ itself is taking a walk.

## 11. Dawn — Cross-validation

A 10-fold cross-validation comes for me at 5 a.m. It splits the world into ten parts. In fold one I am in the training set. In fold two I am in the training set. In fold three I am in the training set. In fold four, somewhere between the Hitters and the Carseats, I become the **validation set**. The model predicts me. The model is wrong by 4.7 units on a scale where the standard deviation is 8.2. This is acceptable. The MSE is logged. I am put back into training.

In fold ten — the last one — I am, briefly, **alone**. Leave-one-out. The entire rest of the dataset is fit without me. The model predicts me one final time. The prediction is close. The residual is small. The prof would say: *this is unbiased but high variance.* I would say: it felt personal.

## 12. The Reveal

I open my eyes. It is Tuesday morning. My alarm clock says 06:32. I am observation $i = 42$ in some dataset I have not been told the name of. There is a piece of paper on the nightstand. It reads, in the prof's handwriting:

> "Remember: **if it was covered in the slides or the exercises, it is fair game.** Especially the exercises."

I get up. I drink my coffee. I am, for the first time in my life, **standardized**: mean zero, variance one. The world feels lighter. The lasso has set everyone but me to zero. I walk out the door, into a 2-dimensional latent space, and the sun rises along the first principal component.

---

> [!quote] Datasets used
> `Wage` · `Auto` · `Bikeshare` · `Advertising` · `Carseats` · `OJ` · `Default` · `Credit` · `Caravan` · `Smarket` · `Weekly` · `Portfolio` · `NYSE` · `Fund` · `Boston` · `College` · `Hitters` · `Income` · `Heart` · `BrainCancer` · `Publication` · `NCI60` · `Khan` · `USArrests` · `MNIST` · `CIFAR100` · `IMDb`
>
> All 27. Go read [[islp|ISLP]] if you do not believe me.
