# Classical Machine Learning: Basic to Advanced, in Steps

A hands-on course that goes from math refreshers to complete end-to-end ML projects. Neural networks are
covered in a separate course.

Follow up with : https://github.com/suraj9321/deep-learning-cv-nlp-llm-zero-to-hero

## Index
- [Phase 0: Tools](#phase-0-tools)
- [Phase 1: Math refreshers](#phase-1-math-refreshers)
- [Phase 2: Data handling](#phase-2-data-handling)
- [Phase 3: ML foundations & supervised learning](#phase-3-ml-foundations--supervised-learning)
- [Phase 4: Unsupervised learning](#phase-4-unsupervised-learning)
- [Phase 5: Practical ML](#phase-5-practical-ml)
- [Phase 6: Capstone](#phase-6-capstone)

## How each module works
Each folder in `modules/` contains:

| File | What to do |
|---|---|
| `lesson.ipynb` | Read it and run every cell. Change values and re-run to build intuition. |
| `exercises.ipynb` | Replace each `None` / `TODO` with your code, then run the **check** cell below it (✅ = pass). |
| `solutions.ipynb` | Open it only after a real attempt, or to compare approaches. |

The teaching pattern is **intuition → a little math → from scratch in NumPy → scikit-learn**.

## Setup (one time)
```bash
git clone https://github.com/suraj9321/classical-machine-learning-basic-to-advanced-in-steps.git
cd classical-machine-learning-basic-to-advanced-in-steps
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab                         # opens in your browser
```
In VS Code, open a notebook and pick the `.venv` Python as the kernel.

Shared plotting helpers live in `common/plotting.py`. Notebooks import them with:
```python
import sys; sys.path.append("../..")
from common.plotting import plot_decision_boundary
```

## Roadmap & progress
Tick each box as you finish a module. ✅ built · ⏳ coming next

Resource key: 🎥 video · 📖 book chapter or guide · 📄 research paper. Watch the videos before or alongside the lesson.
Papers marked *optional* or *advanced* can be skipped on a first pass; the rest are short and readable.

### Phase 0: Tools
- [ ] ✅ `00_setup_and_tools`
  - Why NumPy: vectorization vs Python loops
  - Creating arrays: shape, dtype, `reshape`, `(n,)` vs `(n, 1)`
  - Indexing, slicing, boolean masks, fancy indexing, views vs copies
  - The `axis` argument (`axis=0` gives one result per column, `axis=1` one per row)
  - Broadcasting and standardizing features
  - pandas: Series/DataFrame, `loc` vs `iloc`, filtering, new columns, `groupby`
  - matplotlib and seaborn: line, scatter, histogram, subplots
  - **Resources**
    - 🎥 [Python NumPy Tutorial for Beginners (freeCodeCamp)](https://www.youtube.com/watch?v=QUT1VHiLmmI)
    - 🎥 [Complete Python Pandas Data Science Tutorial (Keith Galli)](https://www.youtube.com/watch?v=vmEHCJofslg)
    - 📖 [NumPy: the absolute basics for beginners (official docs)](https://numpy.org/doc/stable/user/absolute_beginners.html)
    - 📖 [Python Data Science Handbook, chapters 2–4 (free online)](https://jakevdp.github.io/PythonDataScienceHandbook/)
    - 📄 [Harris et al. (2020), *Array programming with NumPy*, Nature (optional)](https://doi.org/10.1038/s41586-020-2649-2)

### Phase 1: Math refreshers
- [ ] ✅ `01_linear_algebra`
  - Vectors as data points; addition and scaling
  - Dot product as a weighted sum (a linear model's prediction) and as similarity (cosine similarity)
  - L1/L2 norms and Euclidean distance
  - Matrices, transpose, matrix–vector product `X @ w`
  - Matrix multiplication shape rules, `X.T @ X`
  - Identity, inverse, `np.linalg.solve`, singular matrices and multicollinearity
  - Least squares with `np.linalg.lstsq` (a preview of linear regression)
  - Preview: eigenvalues and eigenvectors
  - **Resources**
    - 🎥 [Essence of Linear Algebra, chapters 1–4, 9 and 14 (3Blue1Brown) — highly recommended](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
    - 🎥 [Dot products and duality (3Blue1Brown)](https://www.youtube.com/watch?v=LyGKycYT2v0)
    - 📖 [*Mathematics for Machine Learning*, chapters 2–3 (free PDF)](https://mml-book.github.io/)
- [ ] ✅ `02_calculus_and_gradient_descent`
  - Derivatives as slopes; numerical derivatives
  - Common derivative rules
  - Gradient descent in 1-D; the effect of the learning rate
  - Partial derivatives, gradients, gradient checking
  - Gradient descent in 2-D, with contour plots and zig-zag paths
  - The chain rule
  - Fitting a line by minimizing MSE with gradient descent: your first trained model
  - **Resources**
    - 🎥 [Essence of Calculus, chapters 1–4 (3Blue1Brown)](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
    - 🎥 [Gradient Descent, Step-by-Step (StatQuest)](https://www.youtube.com/watch?v=sDv4f4s2SB8)
    - 📖 [*Mathematics for Machine Learning*, chapters 5 and 7.1](https://mml-book.github.io/)
    - 📄 [Ruder (2016), *An overview of gradient descent optimization algorithms* (optional, sections 1–3)](https://arxiv.org/abs/1609.04747)
- [ ] ✅ `03_probability_and_statistics`
  - Mean, median, variance, std, percentiles, outliers, `ddof`
  - Distributions: Bernoulli, binomial, uniform, normal
  - The normal distribution, the 68-95-99.7 rule, z-scores
  - Probability by simulation
  - Conditional probability, Bayes' rule, base rates (medical-test example)
  - Law of large numbers, central limit theorem, standard error
  - Covariance, Pearson correlation, correlation matrices, correlation ≠ causation
  - **Resources**
    - 🎥 [Calculating the Mean, Variance and Standard Deviation (StatQuest)](https://www.youtube.com/watch?v=SzZ6GpcfoQY)
    - 🎥 [The Normal Distribution, Clearly Explained (StatQuest)](https://www.youtube.com/watch?v=rzFX5NWojp0)
    - 🎥 [Bayes theorem, the geometry of changing beliefs (3Blue1Brown)](https://www.youtube.com/watch?v=HZGCoVF3YvM)
    - 🎥 [But what is the Central Limit Theorem? (3Blue1Brown)](https://www.youtube.com/watch?v=zeJD6dqJ5lo)
    - 🎥 [Probability is not Likelihood (StatQuest)](https://www.youtube.com/watch?v=pYxNSUDSFH4)
    - 📖 [Seeing Theory: interactive visual probability (Brown University)](https://seeing-theory.brown.edu/)

### Phase 2: Data handling
- [ ] ⏳ `04_pandas_and_eda`
  - Loading data, inspecting dtypes and summaries
  - Cleaning: missing values, duplicates, type conversion
  - `groupby`, merging/joining, reshaping
  - Exploratory data analysis (EDA) on the titanic and penguins datasets
  - **Resources**
    - 🎥 [Pandas Tutorials playlist (Corey Schafer)](https://www.youtube.com/playlist?list=PL-osiE80TeTsWmV9i9c58mdDCSskIFdDS)
    - 📖 [10 minutes to pandas (official docs)](https://pandas.pydata.org/docs/user_guide/10min.html)
    - 📄 [Wickham (2014), *Tidy Data*, Journal of Statistical Software](https://doi.org/10.18637/jss.v059.i10)
- [ ] ⏳ `05_visualization_for_ml`
  - Distribution plots, pairplots, correlation heatmaps
  - Spotting outliers, skew and data leakage
  - **Resources**
    - 📖 [Python Data Science Handbook, chapter 4: Visualization with Matplotlib](https://jakevdp.github.io/PythonDataScienceHandbook/)
    - 📄 [Kaufman et al. (2012), *Leakage in Data Mining*, ACM TKDD](https://doi.org/10.1145/2382577.2382579)

### Phase 3: ML foundations & supervised learning
- [ ] ⏳ `06_ml_fundamentals`
  - What ML is: supervised vs unsupervised learning, features and labels
  - Train/test split and the ML workflow
  - Overfitting, underfitting, the bias-variance tradeoff
  - **Resources**
    - 🎥 [A Gentle Introduction to Machine Learning (StatQuest)](https://www.youtube.com/watch?v=Gv9_4yMHFhI)
    - 🎥 [Machine Learning Fundamentals: Bias and Variance (StatQuest)](https://www.youtube.com/watch?v=EuBBz3bI-aA)
    - 📖 [*An Introduction to Statistical Learning (ISLP)*, chapter 2 (free PDF)](https://www.statlearning.com/)
    - 📄 [Domingos (2012), *A Few Useful Things to Know About Machine Learning* — must-read](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)
    - 📄 [Breiman (2001), *Statistical Modeling: The Two Cultures* (optional)](https://doi.org/10.1214/ss/1009213726)
- [ ] ⏳ `07_linear_regression`
  - Closed-form solution (normal equation) and gradient descent from scratch
  - scikit-learn `LinearRegression`
  - Metrics: MSE, RMSE, MAE, R²
  - **Resources**
    - 🎥 [Linear Regression, Clearly Explained (StatQuest)](https://www.youtube.com/watch?v=nk2CQITm_eo)
    - 📖 [ISLP chapter 3: Linear Regression](https://www.statlearning.com/)
- [ ] ⏳ `08_logistic_regression_and_classification`
  - Sigmoid, log-loss, a from-scratch implementation, then scikit-learn
  - Confusion matrix, precision, recall, F1, ROC-AUC
  - **Resources**
    - 🎥 [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8)
    - 🎥 [The Confusion Matrix (StatQuest)](https://www.youtube.com/watch?v=Kdsp6soqA7o)
    - 🎥 [ROC and AUC, Clearly Explained (StatQuest)](https://www.youtube.com/watch?v=4jRBRDbJemM)
    - 📖 [ISLP chapter 4: Classification](https://www.statlearning.com/)
    - 📄 [Fawcett (2006), *An introduction to ROC analysis*, Pattern Recognition Letters](https://doi.org/10.1016/j.patrec.2005.10.010)
    - 📄 [Saito & Rehmsmeier (2015), *The Precision-Recall Plot Is More Informative than the ROC Plot…*, PLOS ONE](https://doi.org/10.1371/journal.pone.0118432)
- [ ] ⏳ `09_model_evaluation_and_regularization`
  - Cross-validation and learning curves
  - Polynomial features
  - Ridge, Lasso, ElasticNet
  - **Resources**
    - 🎥 [Machine Learning Fundamentals: Cross Validation (StatQuest)](https://www.youtube.com/watch?v=fSytzGwwBVw)
    - 🎥 [Regularization Part 1: Ridge (L2) Regression (StatQuest)](https://www.youtube.com/watch?v=Q81RR3yKn30)
    - 🎥 [Regularization Part 2: Lasso (L1) Regression (StatQuest)](https://www.youtube.com/watch?v=NGf0voTMlcs)
    - 📖 [ISLP chapters 5 (Resampling) and 6 (Regularization)](https://www.statlearning.com/)
    - 📄 [Tibshirani (1996), *Regression Shrinkage and Selection via the Lasso*](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)
    - 📄 [Hoerl & Kennard (1970), *Ridge Regression* (optional, historical)](https://doi.org/10.1080/00401706.1970.10488634)
    - 📄 [Zou & Hastie (2005), *Regularization and Variable Selection via the Elastic Net* (optional)](https://doi.org/10.1111/j.1467-9868.2005.00503.x)
- [ ] ⏳ `10_preprocessing_and_pipelines`
  - Scaling, one-hot and ordinal encoding, imputation
  - `ColumnTransformer` and `Pipeline`
  - Avoiding data leakage
  - **Resources**
    - 📖 [scikit-learn guide: Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html)
    - 📖 [scikit-learn guide: Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)
    - 📄 [Kaufman et al. (2012), *Leakage in Data Mining* (re-read now)](https://doi.org/10.1145/2382577.2382579)
- [ ] ⏳ `11_knn_and_naive_bayes`
  - KNN from scratch, distance metrics, the curse of dimensionality
  - Gaussian and multinomial Naive Bayes (text example)
  - **Resources**
    - 🎥 [K-nearest neighbors, Clearly Explained (StatQuest)](https://www.youtube.com/watch?v=HVXime0nQeI)
    - 🎥 [Naive Bayes, Clearly Explained (StatQuest)](https://www.youtube.com/watch?v=O2L2Uv9pdDA)
    - 📄 [Cover & Hart (1967), *Nearest Neighbor Pattern Classification* (optional, historical)](https://doi.org/10.1109/TIT.1967.1053964)
- [ ] ⏳ `12_decision_trees`
  - Gini impurity and entropy, a from-scratch tree for small cases
  - Depth, pruning, feature importance
  - **Resources**
    - 🎥 [Decision and Classification Trees, Clearly Explained (StatQuest)](https://www.youtube.com/watch?v=_L39rN6gz7Y)
    - 📖 [ISLP chapter 8: Tree-Based Methods](https://www.statlearning.com/)
    - 📄 [Quinlan (1986), *Induction of Decision Trees* (ID3)](https://doi.org/10.1007/BF00116251)
- [ ] ⏳ `13_ensembles`
  - Bagging and random forests
  - Boosting intuition and gradient boosting
  - XGBoost and LightGBM
  - **Resources**
    - 🎥 [Random Forests Part 1 (StatQuest)](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
    - 🎥 [Gradient Boost Part 1: Regression Main Ideas (StatQuest)](https://www.youtube.com/watch?v=3CC4N4z3GJc)
    - 🎥 [XGBoost Part 1: Regression (StatQuest)](https://www.youtube.com/watch?v=OtD8wVaFm6E)
    - 📄 [Breiman (2001), *Random Forests*, Machine Learning](https://doi.org/10.1023/A:1010933404324)
    - 📄 [Friedman (2001), *Greedy Function Approximation: A Gradient Boosting Machine* (advanced)](https://doi.org/10.1214/aos/1013203451)
    - 📄 [Chen & Guestrin (2016), *XGBoost: A Scalable Tree Boosting System*](https://arxiv.org/abs/1603.02754)
    - 📄 [Ke et al. (2017), *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*](https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
    - 📄 [Grinsztajn et al. (2022), *Why do tree-based models still outperform deep learning on tabular data?*](https://arxiv.org/abs/2207.08815)
- [ ] ⏳ `14_svm`
  - Margins, kernels, the C and gamma parameters
  - Visualized decision boundaries
  - **Resources**
    - 🎥 [Support Vector Machines Part 1: Main Ideas (StatQuest)](https://www.youtube.com/watch?v=efR1C6CvhmE)
    - 🎥 [Support Vector Machines in Python from Start to Finish (StatQuest)](https://www.youtube.com/watch?v=8A7L0GsBiLQ)
    - 📖 [ISLP chapter 9: Support Vector Machines](https://www.statlearning.com/)
    - 📄 [Cortes & Vapnik (1995), *Support-Vector Networks*](https://doi.org/10.1007/BF00994018)

### Phase 4: Unsupervised learning
- [ ] ⏳ `15_clustering`
  - k-means from scratch, the elbow and silhouette methods
  - Hierarchical clustering, DBSCAN
  - **Resources**
    - 🎥 [StatQuest: K-means clustering](https://www.youtube.com/watch?v=4b5d3muPQmA)
    - 🎥 [StatQuest: Hierarchical Clustering](https://www.youtube.com/watch?v=7xHsRkOdVwo)
    - 📖 [ISLP chapter 12: Unsupervised Learning](https://www.statlearning.com/)
    - 📄 [Ester et al. (1996), *A Density-Based Algorithm for Discovering Clusters…* (DBSCAN)](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf)
    - 📄 [Rousseeuw (1987), *Silhouettes: A graphical aid to… cluster analysis*](https://doi.org/10.1016/0377-0427(87)90125-7)
- [ ] ⏳ `16_dimensionality_reduction`
  - PCA from scratch (covariance matrix and eigenvectors), explained variance
  - t-SNE for visualization
  - **Resources**
    - 🎥 [Principal Component Analysis (PCA), Step-by-Step (StatQuest)](https://www.youtube.com/watch?v=FgakZw6K1QQ)
    - 🎥 [t-SNE, Clearly Explained (StatQuest)](https://www.youtube.com/watch?v=NEaUSP4YerM)
    - 📄 [Shlens (2014), *A Tutorial on Principal Component Analysis* — very readable](https://arxiv.org/abs/1404.1100)
    - 📄 [van der Maaten & Hinton (2008), *Visualizing Data using t-SNE*, JMLR](https://www.jmlr.org/papers/v9/vandermaaten08a.html)
    - 📄 [Wattenberg et al. (2016), *How to Use t-SNE Effectively* (Distill, interactive)](https://distill.pub/2016/misread-tsne/)

### Phase 5: Practical ML
- [ ] ⏳ `17_tuning_and_real_world_issues`
  - GridSearchCV and RandomizedSearchCV
  - Imbalanced data: class weights, SMOTE
  - Model interpretation: permutation importance
  - Saving models with joblib
  - **Resources**
    - 📖 [*Interpretable Machine Learning* (Molnar), chapter on permutation feature importance](https://christophm.github.io/interpretable-ml-book/)
    - 📄 [Bergstra & Bengio (2012), *Random Search for Hyper-Parameter Optimization*, JMLR](https://www.jmlr.org/papers/v13/bergstra12a.html)
    - 📄 [Chawla et al. (2002), *SMOTE: Synthetic Minority Over-sampling Technique*](https://arxiv.org/abs/1106.1813)

### Phase 6: Capstone
- [ ] ⏳ `18_capstone_project`
  - End-to-end project on California housing (regression) or Adult income (classification)
  - EDA → preprocessing pipeline → baseline → several models → tuning → final evaluation → write-up
  - **Resources**
    - 📖 [Google, *Rules of Machine Learning*](https://developers.google.com/machine-learning/guides/rules-of-ml)
    - 📄 [Sculley et al. (2015), *Hidden Technical Debt in Machine Learning Systems*](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)
    - 📄 [Domingos (2012), *A Few Useful Things to Know About Machine Learning* (re-read with fresh eyes)](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)

## Suggested pace
About 3–5 hours per module: about 1 hour for the lesson, 1–2 hours for exercises, and the rest re-doing the parts that felt shaky.
Don't rush the math modules. Everything later builds on `@`, `axis`, gradients and probabilities.

## Tips
- Before every matrix operation, say the **shapes** out loud. Most bugs are shape bugs.
- When stuck on an exercise for more than 20 minutes, re-read the matching lesson section before opening the solution.
- Commit your progress with git after each module: `git add -A && git commit -m "finished module 01"`.
