# ECO 368 — the book

A reading companion for Econ 368, *Data Science for Economists*, written to the order we teach the course.

## Why it exists

Students read Géron, *Hands-On Machine Learning with Scikit-Learn and PyTorch* (O'Reilly, 2025). It is an excellent engineering book, but it is not written for economists: its examples are images and housing prices, it says little about data acquisition, experiments, or measurement, and it is long. This book is the course as taught: our order, our economic questions, our datasets, and Python that runs. It sits alongside Géron, the way the Econ 250 book sits alongside Keller.

## Sources, and what we take from each

| Source | What it gives us |
|---|---|

| Géron, *Hands-On Machine Learning with Scikit-Learn and PyTorch* (2025) | the order and level of Part III; which methods are standard |
| Géron's companion notebooks (github.com/ageron/handson-mlp, Apache 2.0) | reference implementations; any adapted snippet is credited in a code comment |
| ISLP (James, Witten, Hastie, Tibshirani & Taylor 2023) | the statistical framing of bias, variance and resampling |
| Kohavi, Tang & Xu (2020) | the practice of online experiments |
| Gentzkow, Kelly & Taddy (2019) | the text-as-data framework |
| This course's own slides and notebooks | most of the examples and data, already checked |

**Everything in the book is written from scratch.** Géron's text and figures are copyrighted; we use the book to decide *what* to cover and in *what order*, never for sentences or figures. Code is our own unless a comment says it is adapted from the Apache-licensed notebooks.

## Rules the book follows

1. **Every number is computed, not typed.** Chapters are Quarto files in `src/` whose code runs when the book is built (`python3 build.py`). If a number in the prose comes from data, it comes from an inline `{python}` expression.
2. **It matches what students were shown.** Same examples, datasets and notation as the Monday slides and Wednesday notebooks.
3. **Economics first.** Every method opens with an economic question and closes with what an economist would do with the answer.
4. **Python appears as it is taught.** pandas, scikit-learn, statsmodels and PyTorch, in the style of the work-through notebooks. No library a student has not installed.
5. **Written in the first person**, like the lecture notes, not as a reference work.
6. **Undergraduate level.** At most one key equation per idea; derivations go in a short *Going further* box at the end of a section.

## Chapter template

```
# Chapter N — Title

> **The question:** one sentence, in economic terms.

## (teaching sections, each ending in a worked example with a Solution)
## In Python        the full pipeline on the chapter's dataset, run end to end
## What goes wrong  the 4–6 mistakes students actually make
## Exercises        5–8 problems, a mix of interpretation and code
### Answers         worked answers, computed
```

## Table of contents

### Part I — Tools for economic data

| # | Chapter | Covers | Course | Géron ch. |
|---|---|---|---|---|
| 1 | What data science is for | description, prediction and causation; what "big" data changes | L01 | 1 |
| 2 | Python for economists | types, lists and dictionaries, functions, NumPy, a regression by hand | L02 | — |
| 3 | Reproducible projects with Git | folders, scripts, paths, commits, branches, pull requests | L03–L04 | — |
| 4 | Wrangling, bigger data, and pictures | tidy data, pandas verbs, merges, Parquet/DuckDB, charts that inform | L05–L07 | — |
| 5 | Mapping opportunity | the Opportunity Atlas, rank-rank mobility, geopandas, choropleths | L08–L09 | — |
| 6 | Getting data from the web | HTTP, JSON, APIs and keys, HTML and scraping, the ethics | L10–L11 | — |

### Part II — Experiments and uncertainty

| # | Chapter | Covers | Course | Géron ch. |
|---|---|---|---|---|
| 7 | A/B tests | randomization, difference in means, power, CUPED, peeking, many metrics | L12 | — |
| 8 | Resampling: simulation and the bootstrap | Monte Carlo, coverage, the bootstrap, permutation tests | L14 | — |

### Part III — Machine learning

| # | Chapter | Covers | Course | Géron ch. |
|---|---|---|---|---|
| 9 | An end-to-end machine learning project | the landscape, train/test, pipelines, cross-validation, the test set is sacred | L15 | 1–2 |
| 10 | Training linear models | OLS as a predictor, gradient descent, polynomials and splines, learning curves | L13 | 4 |
| 11 | Regularization and model selection | ridge, LASSO, elastic net, choosing the penalty by cross-validation | L16 | 4 |
| 12 | Classification | logistic regression, other classifiers, confusion matrices, thresholds and costs | L17 | 3–4 |
| 13 | Support vector machines | the widest street, soft margins, kernels | L19 | App. C |
| 14 | Decision trees | splits, impurity, depth, why trees overfit | L18 | 5 |
| 15 | Ensembles: forests and boosting | bagging, random forests, gradient boosting, importance and SHAP | L18 | 6 |
| 16 | Dimensionality reduction | the curse of dimensionality, PCA, a development index | L21 | 7 |
| 17 | Clustering | k-means, choosing k, hierarchical clustering, Gaussian mixtures | L21 | 8 |
| 18 | Neural networks | from logit to a neuron, backpropagation, training, when nets lose to trees | L20 | 9–11 |

### Part IV — Language

| # | Chapter | Covers | Course | Géron ch. |
|---|---|---|---|---|
| 19 | Text as data | regex, bag of words, tf-idf, dictionaries, text regression | L22 | — |
| 20 | Embeddings and topic models | topic models, word and sentence embeddings, similarity | L23 | 14 |
| 21 | Transformers, LLMs, and how they are told to behave | attention, pretraining and fine-tuning, constitutions, LLMs as annotators | L24 | 15 |

