# Data Science for Economists

**Teaching notes for Econ 368 · Bates College · Winter 2027**

Hitanshu Pandit · Department of Economics

---

**For reading I recommend Aurélien Géron, *Hands-On Machine Learning with Scikit-Learn and PyTorch* (O'Reilly, 2025).** This is not a replacement for it. It is my teaching notes written up, in the order we teach the course, with economic examples and Python you can run.

Use it for three things: to **read** the week's material before Monday, to **run** the code (every block works in the course environment or Google Colab), and to work the **exercises** at the end of each chapter. Géron is where you go for the fuller treatment and more engineering detail; this is where you go for how we did it in class.

## Contents

| | Chapter | Covers |
|---|---|---|
| | **Part I — Tools for economic data** | |
| 1 | [What data science is for](01_what_data_science_is_for.md) | description, prediction and causation; what "big" data changes |
| 2 | [Python for economists](02_python_for_economists.md) | types, lists and dictionaries, functions, NumPy, a regression by hand |
| 3 | [Reproducible projects with Git](03_reproducible_projects.md) | folders, scripts, paths, commits, branches, pull requests |
| 4 | [Wrangling, bigger data, and pictures](04_wrangling_and_pictures.md) | tidy data, pandas verbs, merges, **missing values**, Parquet/DuckDB, charts that inform |
| 5 | [Mapping opportunity](05_mapping_opportunity.md) | the Opportunity Atlas, rank-rank mobility, geopandas, choropleths |
| 6 | [Getting data from the web](06_data_from_the_web.md) | HTTP, JSON, APIs and keys, HTML and scraping, the ethics |
| | **Part II — Experiments and uncertainty** | |
| 7 | [A/B tests](07_ab_tests.md) | randomization, difference in means, power, CUPED, peeking, many metrics |
| 8 | [Resampling: simulation and the bootstrap](08_resampling.md) | Monte Carlo, coverage, the bootstrap, permutation tests |
| | **Part III — Machine learning** | |
| 9 | [An end-to-end machine learning project](09_end_to_end_project.md) | the landscape, train/test, pipelines, cross-validation, the test set is sacred |
| 10 | [Training linear models](10_training_linear_models.md) | OLS as a predictor, gradient descent, polynomials and splines, learning curves |
| 11 | [Regularization and model selection](11_regularization.md) | ridge, LASSO, elastic net, choosing the penalty by cross-validation |
| 12 | [Classification](12_classification.md) | logistic regression, other classifiers, confusion matrices, thresholds and costs |
| 13 | [Support vector machines](13_support_vector_machines.md) | the widest street, soft margins, kernels |
| 14 | [Decision trees](14_decision_trees.md) | splits, impurity, depth, why trees overfit |
| 15 | [Ensembles: forests and boosting](15_ensembles.md) | bagging, random forests, gradient boosting, importance and SHAP |
| 16 | [Dimensionality reduction](16_dimensionality_reduction.md) | the curse of dimensionality, PCA, a development index |
| 17 | [Clustering](17_clustering.md) | k-means, choosing k, hierarchical clustering, Gaussian mixtures |
| 18 | [Neural networks](18_neural_networks.md) | from logit to a neuron, backpropagation, training, when nets lose to trees |
| | **Part IV — Language** | |
| 19 | [Text as data](19_text_as_data.md) | regex, bag of words, tf-idf, dictionaries, text regression |
| 20 | [Embeddings and topic models](20_embeddings_and_topics.md) | topic models, word and sentence embeddings, similarity |
| 21 | [Transformers, LLMs, and how they are told to behave](21_transformers_and_llms.md) | attention, pretraining and fine-tuning, constitutions, LLMs as annotators |

Every chapter ends with **Economics reading**: published papers in which economists use the chapter's method, each with a question to think about.

## How to read it

Read with a laptop open. Every code block runs, top to bottom, and the numbers in the text are the numbers the code prints. Change something and run it again; that is how this material is learned.

The datasets repeat on purpose. Bay Area rents appear in Chapters 9, 10, 14 and 15; hotel bookings in Chapters 12 and 13; county childcare costs in Chapters 4 and 11. You are meant to notice that a new method is answering a different question about data you already understand.

## A note on accuracy

Every number in this book is computed by the code in it. If a number here disagrees with a slide, **the book is right and the slide is wrong**. Please tell me in class or open an issue.

---

© 2026 Hitanshu Pandit. Text released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); code under the MIT license.
