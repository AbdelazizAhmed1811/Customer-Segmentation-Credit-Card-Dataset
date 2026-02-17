# Customer Segmentation — Task Checklist

## Phase 1: Data Exploration & Cleaning

- [x] Load dataset and inspect structure (`df.info()`, `df.head()`)
- [x] Check missing values and their percentages
- [x] Investigate **why** values are missing (MNAR vs MCAR analysis)
- [x] Decide how to handle missing values (impute / drop)
- [x] Check for duplicates
- [x] Box plots grouped by feature range subgroups
- [x] Distribution analysis for each column (histograms / KDE)
- [x] Pairwise analysis (correlation heatmap, scatter plots)
- [ ] Investigate outliers and decide how to deal with them

## Phase 2: Preprocessing

- [x] Drop `CUST_ID` and `TENURE`
- [x] Handle missing values (MINIMUM_PAYMENTS → 0, drop 1 CREDIT_LIMIT row)
- [x] Log transform high-value monetary features
- [x] Feature scaling (StandardScaler)

## Phase 3: Clustering

- [x] Find optimal number of clusters (GMM BIC/AIC, Silhouette, Dendrogram)
- [x] Fit GMM model + DBSCAN
- [x] Visualise clusters across PCA, t-SNE, UMAP, PCA→UMAP

## Phase 4: Segment Analysis

- [ ] Profile each cluster (mean feature values)
- [ ] Business insights and recommendations
