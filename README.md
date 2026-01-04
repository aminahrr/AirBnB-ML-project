# Airbnb Price Classification - NYC
The main task we will be working on is supervised learning, classification. We will define target as  price: ‘low’, 'medium’, ‘high’ using quantile-based bins or possibly cluster-based bins, derived using  unsupervised learning, to explore underlying trends. 

## Team Members
### Part 1 -
- Asja Bašović (A) - Basic EDA, Data Cleaning, Feature Engineering
- Esma Kaderić (B) - Train-Test Split, Missing Data, Feature Selection  
- Amina Hrustić (C) - Full EDA, Clustering
### Part 2
- Asja Bašović  - 
- Esma Kaderić  - 
- Amina Hrustić - 
...

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-team/airbnb-project.git
cd airbnb-project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Place the Airbnb NYC dataset in:
```
data/raw/airbnb_nyc.csv
```

### 4. Run the notebook
```bash
jupyter notebook airbnb_classification.ipynb
```

## Project Structure
```
airbnb-ml-project/
│           
├── data/
│   ├── raw/
│   │   └── ML_dfc.csv          # Dataset (not in git)
│   └── processed/ 
│ 
├── models/
├── notebooks/
│   ├── 01_data_preprocessing.ipynb #Asjas part is pasted here
│   ├── 02_feature_engineering_clustering.ipynb
│   └── 03_modeling_feature_selection.ipynb
├── README.md
├── report/
│   └── Report.pdf  
└──  requirements.txt               # Python dependencies
```

## Workflow
## SECTION 2 — DATA EXPLORATION (FIRST LOOK)

**2.1** Quick preview + structure

* `df.head()`
* `df.shape`
* `df.columns`
* `df.info()`

**2.2** Numeric summary statistics

* `df.describe().T`

**2.3** Missing values overview

* `msno.bar(df)`
* `msno.matrix(df)`
* *Note:* Some features (e.g., `host_response_rate`) are strings due to `%` signs, limiting visualization.

**2.4** Cardinality audit for categorical/object columns

* Identify object columns:

  * `obj_cols = df.select_dtypes(include=["object"]).columns.tolist()`
  * `len(obj_cols)`
  * `obj_cols[:20]`
* Build cardinality table:

  * Columns: `column`, `n_unique`, `missing_count`, `missing_pct`, `top_1`, `top_1_pct`
  * `cardinality_audit = cardinality_audit.sort_values("n_unique", ascending=False).reset_index(drop=True)`
  * `display(cardinality_audit.head(30))`
* High cardinality inspection (`n_unique >= 100`), limited to first 10 columns with `value_counts().head(10)`

*Findings:*

* `amenities` has 358,667 unique combinations → OHE infeasible.
* `neighbourhood_cleansed` (225) and `property_type` (82) → bucket rare categories.
* `host_response_time` missing 41.6% → treat missing as “Unknown”.

**2.5** Early numeric-only view *(limited due to formatting issues)*

* `df_numeric = df.select_dtypes(include=['number'])`
* `df_numeric.columns`

**2.6** Outlier + distribution diagnostics *(pre-cleaning)*

* Helper function: `plot_feature_grid(df, cols, kind="box", ncols=5, figsize=None, bins=30, title_prefix=None)`
* Boxplots and histograms for sets of numeric features:

  *Set 1:*

  * `cols_1 = ['bathrooms','bedrooms','beds','review_scores_rating','review_scores_accuracy', 'review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value']`
  * `plot_feature_grid(df, cols_1, kind="box")`
  * `plot_feature_grid(df, cols_1, kind="hist")`
  * *Notes:* Most listings small, few extreme values; review scores highly positive and skewed.

  *Set 2:*

  * `cols_2 = ['accommodates','minimum_nights','maximum_nights','availability_30','availability_365','number_of_reviews','number_of_reviews_l30d','calculated_host_listings_count','reviews_per_month']`
  * `plot_feature_grid(df, cols_2, kind="box")`
  * `plot_feature_grid(df, cols_2, kind="hist")`

**2.7** Side-by-side plots + interpretability notes

* Helper: `plot_hist_and_box_with_notes(df, feature, bins=40)` → histogram + boxplot + quick stats

* Run for: `["bathrooms","review_scores_rating","availability_365"]`

* Custom-styled boxplots for: `["beds","number_of_reviews"]`

  * Horizontal boxplots with color customizations
  * Title: *Outlier Analysis of Selected Listing Features — Beds and Total Number of Reviews*
  * Notes: Both features right-skewed, many outliers; most listings have low values, few extreme cases.


## SECTION 3 — MECHANICAL CLEANING (NO LEAKAGE)

**Purpose:** Standardize types/formats **before splitting**; avoid any learned transforms/encoding here.

**3.1** Identify non-numeric columns needing standardization

* `df_non_numeric = df.select_dtypes(exclude=['number'])`
* `df_non_numeric.info()`
* `df_non_numeric.head()`

*Findings:*

* `last_scraped` → convert to datetime
* Boolean columns (`host_is_superhost`, `host_has_profile_pic`, `host_identity_verified`, `has_availability`, `instant_bookable`) → map `'t'/'f'` to `1/0`
* String percentages/price (`host_response_rate`, `host_acceptance_rate`, `price`) → strip `%`/`$` and cast numeric
* Other string types deferred to avoid leakage (e.g., OHE)

**3.2** Convert `last_scraped` to datetime

* `df['last_scraped'] = pd.to_datetime(df['last_scraped'], format='%Y-%m-%d')`
* `df.head()`

**3.3** Convert boolean-like columns (`'t'/'f'`) to `1/0`

* `bool_cols = ['host_is_superhost','host_has_profile_pic','host_identity_verified','has_availability','instant_bookable']`
* Map: `df[col] = df[col].map({'f': 0, 't': 1})`

**3.4** Strip `%` and `$` then cast to numeric

* `df['host_response_rate'] = pd.to_numeric(df['host_response_rate'].str.rstrip('%'), errors='coerce')`
* `df['host_acceptance_rate'] = pd.to_numeric(df['host_acceptance_rate'].str.rstrip('%'), errors='coerce')`
* `df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)`
* `df.info()`
* `df.head()`

**3.5** Visualize cleaned target distribution (price)

* `sns.boxplot(x=df['price'])`
* `sns.histplot(x=df['price'])`

**3.6** Visualize cleaned features

* Boolean features: `['host_is_superhost','host_has_profile_pic','host_identity_verified','has_availability','instant_bookable']` → `plot_feature_grid(df, bool_cols, kind="count")`
* Symbol-containing features: `['host_response_rate','host_acceptance_rate']` →

  * `plot_feature_grid(df, symbol_cols, kind="box")`
  * `plot_feature_grid(df, symbol_cols, kind="hist")`



## SECTION 4 — TARGET FILTERING + TRAIN/TEST SPLIT (A)

**4.1** Drop rows with missing target (`price`)

* `df_for_split = df.dropna(subset=['price'])`
* `df_for_split.shape`
* `df_for_split.info()`
* *Note:* Remaining nulls handled later to avoid data leakage.

**4.2** Group-based split according to listing `id`

* `splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)`
* `train_idx, test_idx = next(splitter.split(df_for_split, groups=df_for_split['id']))`
* `df_train = df_for_split.iloc[train_idx].copy()`
* `df_test  = df_for_split.iloc[test_idx].copy()`

**4.3** Split validation checks

* Confirm no ID overlap:

  * `train_ids = set(df_train["id"])`
  * `test_ids = set(df_test["id"])`
  * `overlap_ids = train_ids.intersection(test_ids)`
  * `print("ID overlap count:", len(overlap_ids))`
  * `print("No overlap:", len(overlap_ids) == 0)`
* Confirm shapes:

  * `df.shape` (full before dropna)
  * `df_for_split.shape` (after dropna price)
  * `df_train.shape`
  * `df_test.shape`
* Target sanity checks (`price`):

  * Non-null counts in full/train/test
  * Train/Test min/max prices


## SECTION 5 — PREPROCESSING (IMPUTATION + SCALING + ENCODING)

**Important:**

* Exclude `price`, label columns, `id`, `host_id` from predictors.
* Fit preprocessor on TRAIN predictors only.

**5.0** Define predictor frames (drop target + identifiers)

* `DROP_COLS = ['price','id','host_id']`
* `X_train_df = df_train.drop(columns=DROP_COLS, errors='ignore').copy()`
* `X_test_df  = df_test.drop(columns=DROP_COLS, errors='ignore').copy()`
* `df_train.head()`

**5.1** Identify column groups on train predictors only + missingness audit

* Numeric columns: `numeric_cols = X_train_df.select_dtypes(include=['number']).columns.tolist()`

* Categorical columns: `categorical_cols = X_train_df.select_dtypes(include=['object']).columns.tolist()`

* Datetime columns: `datetime_cols = X_train_df.select_dtypes(include=['datetime64']).columns.tolist()`

* Display features in a table:

  * `features_df = pd.DataFrame({'Feature Name': numeric_cols + categorical_cols, 'Feature Type': ['Numeric']*len(numeric_cols) + ['Categorical']*len(categorical_cols)})`
  * `features_df`

* Visualization of groups:

  * `audit = pd.DataFrame({"column": X_train_df.columns, "dtype": X_train_df.dtypes.astype(str)})`
  * `audit["group"] = np.select([...], ["numeric","categorical","datetime"], default="other")`
  * `audit.sort_values(["group","column"]).reset_index(drop=True)`
  * `audit["group"].value_counts()`
  * `display(audit)`

* Numeric missing values summary:

  * `missing_numeric_df = X_train_df[numeric_cols].isna().agg(['sum','mean']).T.rename(columns={'sum':'Missing Count','mean':'Missing Percentage'})`
  * `missing_numeric_df['Missing Percentage'] *= 100`
  * Keep only features with missing values and sort by percentage

* Visualize missing values (top 15 columns by missing %):

  * `cols_show = X_train_df[numeric_cols].isna().mean().sort_values(ascending=False).head(16).index.tolist()`
  * `tmp = pd.DataFrame({c: X_train_df[c].isna().map({True:"Missing",False:"Present"}) for c in cols_show})`
  * `plot_feature_grid(tmp, cols_show, kind="count", ncols=4, title_prefix="Present vs Missing")`

* Horizontal stacked bar chart of present vs missing:

  * `summary = pd.DataFrame({"Present": X_train_df[cols_show].notna().sum(), "Missing": X_train_df[cols_show].isna().sum()}).sort_values("Missing")`
  * `ax = summary.plot(kind="barh", stacked=True, figsize=(10,6))`
  * `ax.set_xlabel("Count")`, `ax.set_ylabel("Feature")`, `ax.set_title("Missing vs Present counts (train)")`
  * `plt.tight_layout()`, `plt.show()`

*Note:* Missing values mostly in `review_scores_*` (~30%); keep all numeric columns and handle missing with median imputation.


## SECTION 6 — Feature Engineering

### 6.1 Days Since Last Scraped

* Converted `last_scraped` into a numeric feature `days_since_last_scraped` by calculating the difference in days from the most recent scrape date in the training set.
* This avoids using raw calendar dates, making the feature model-friendly and capturing recency information.

### 6.2 Neighborhood Bucketing

* Rare neighborhoods (fewer than 500 listings) are grouped into a single category called `"Other"` to reduce sparsity.
* The original `neighbourhood_cleansed` column is dropped, replaced by `neighbourhood_cleaned`.
* Visual checks confirm the distribution and correctness of the recoding.
* This ensures location information is preserved without creating too many sparse dummy variables.

### 6.3 Amenity Analysis

* Extracted all unique amenities from the `amenities` column to understand dimensionality and sparsity.
* Computed the frequency of each amenity across listings. Most amenities appear in only a small fraction of listings, confirming high sparsity.

### 6.4 Number of Amenities Feature

* Created `num_amenities` to capture the total count of amenities for each listing.
* Converts a high-dimensional, sparse categorical feature into a single numeric feature, preserving information while simplifying the model.

### 6.5 Frequency-Based Amenity Selection

* Selected amenities appearing in 2–70% of listings for feature engineering.
* Built one-hot features like `has_wifi`, `has_air_conditioning`, etc., for the selected amenities.
* Dropped the original `amenities` column after encoding.
* This balances informativeness with sparsity reduction.

### 6.6 Amenity Grouping

* Amenities grouped into **logical categories**:

  * `kitchen_score`
  * `comfort_score`
  * `entertainment_score`
  * `laundry_score`
  * `luxury_score`
* Created `parking_level` as an ordinal feature representing parking quality (0–3).
* Standalone amenities like `has_pets_allowed` or `has_elevator` are kept as individual features.
* Extra raw `has_` columns not part of groups or standalone lists are dropped.
* Diagnostic checks ensure non-amenity columns are untouched.

**Result:**

* Final amenity-related features include grouped scores, `parking_level`, standalone amenities, and `num_amenities`.
* Amenity features are condensed, interpretable, and ready for modeling.

### 6.7 Additional Feature Engineering

* Added **safe, non-leaky features**:

  * `scrape_month` from `last_scraped` to capture seasonal trends.
  * `host_reliability` as the average of response and acceptance rates.
  * Ratios for listing efficiency:

    * `beds_per_guest`
    * `bathrooms_per_bedroom`
    * `crowding` (guests per bedroom)
  * Review summary features:

    * `review_quality_mean`
    * `review_variance`
* These features improve model expressiveness without leaking future information.

---

**Summary:**
Feature engineering focused on extracting **temporal, spatial, and amenity-related information** while controlling sparsity and dimensionality. Neighborhoods and amenities were carefully recoded, grouped, and summarized into interpretable features. Additional safe features derived from host, capacity, and review information provide richer predictive signals. The dataset is now enhanced with both numeric and categorical engineered features, ready for preprocessing pipelines.


## SECTION 7 — Preprocessing Pipelines: Imputation, Scaling, and Encoding

### 7.1 Separating Predictors and Target

* The predictors (`X_train_raw` / `X_test_raw`) are obtained by dropping `price`, `id`, and `host_id` from the training and test sets.
* The target variable `y_train` and `y_test` is the `price` column.

### 7.2 Feature Type Identification

* Numeric columns are automatically identified (`float64` and `int64` types).
* Categorical columns are those with `object` type.
* Datetime columns are those with `datetime64` type.
* This allows separate preprocessing pipelines for each feature type.

### 7.3 Frequency Encoding of Categorical Features

* Categorical features are transformed using **frequency encoding**, where each category is replaced by its occurrence proportion in the training set.
* Optional **smoothing** is applied to reduce noise from rare categories.
* Original categorical columns are dropped, replaced by their `_freq` encoded counterparts.
* This ensures all categorical data becomes numeric and compatible with downstream pipelines.

### 7.4 & 7.5 Numeric Feature Pipeline

* Numeric columns (excluding datetime) undergo:

  * **Median imputation** to fill missing values.
  * **Standard scaling** to normalize distributions.
* These steps are applied via a `Pipeline` inside a `ColumnTransformer`, allowing easy integration with other preprocessing steps.

### 7.6 Reattaching Datetime Columns

* After numeric transformations, datetime columns are reattached to preserve temporal information.
* Final datasets (`X_train` / `X_test`) contain processed numeric features and original datetime features.

### 7.7 State Check and Diagnostics

* Shapes of `X_train` and `X_test` are confirmed.
* Presence of missing values is checked (should be none after imputation for numeric and frequency encoding for categorical features).
* Column type distribution is inspected to ensure numeric features are processed and datetime columns are retained.
* This step confirms that the preprocessing pipeline successfully prepared the data for modeling.

---

**Summary:**
The preprocessing pipeline cleanly separates numeric, categorical, and datetime features. Categorical variables are converted to numeric via frequency encoding, numeric features are imputed and scaled, and datetime columns are preserved. After processing, both training and test datasets are ready for modeling, with no missing values and all features in a usable format.


## SECTION 8 — Deep EDA 

### 8.1 Correlation Analysis on Numeric Features

* Correlation matrix computed for all numeric features in the training set.
* Top 10 features correlated with `price` were extracted.
* Visualization included:

  * Heatmap of the full numeric correlation matrix.
  * Heatmap of the top features versus `price`.

**Observations:**

* **Linearity problem:** Highest correlation with `price` is only 0.11 (`instant_bookable`), and other numeric features like `accommodates` and `bedrooms` are all below 0.10, indicating highly non-linear or noisy relationships.
* **Multicollinearity:** Strong correlations exist between `accommodates`, `beds`, `bedrooms`, and `bathrooms` (0.37–0.63), suggesting potential for feature combination or removal.

---

### 8.2 Categorical Features vs Price

* Focused on `room_type` (if present).
* Analyses performed:

  * Boxplots of price distribution by room type (capped at $5000 for clarity).
  * Bar charts showing distribution of room type categories.
  * Statistical summaries (count, mean, median, std).
  * Kruskal-Wallis H-test to check differences in price across categories.

**Insights:**

* **Hierarchy:** Entire homes/apartments and hotel rooms command higher prices; shared rooms are cheaper with low variance.
* **Imbalance:** Hotel rooms and shared rooms are rare categories.
* **Statistical significance:** Prices differ significantly between room types (p < 0.05).

---

### 8.3 Outlier Detection and Analysis

* Features analyzed: `price`, `accommodates`, `bedrooms`, `beds`, `minimum_nights`.
* Outliers defined using IQR (1.5 × IQR beyond Q1/Q3).
* For each feature, bounds, outlier counts, percentages, and outlier ranges were reported.

**Key takeaways:**

* Price outliers are extreme (long right tail).
* Capacity-related features show fewer extreme deviations but still need attention.

---

### 8.4 Price Outlier Investigation

* Price outliers further explored with descriptive statistics including `accommodates`, `bedrooms`, `bathrooms`, `neighbourhood_cleaned`, and `room_type`.
* Outliers mainly correspond to high-end listings, large homes, or rare neighborhoods.

---

### 8.5 Bivariate Analysis (Price vs Features)

* Initial scatterplots of `price` vs numeric features suffered from overlap due to extreme values.
* Improved visualizations:

  * Boxplots for `accommodates`, `bathrooms`, `beds`.
  * Violin plot for `bedrooms` (limited to common counts).
  * Hexbin plots for `number_of_reviews` and `review_scores_rating` (log scale for clarity).

**Findings:**

* **Capacity drives price:** `accommodates`, `bedrooms`, `bathrooms`, and `beds` show stair-step patterns; larger listings have higher median price and variance.
* **Number of reviews vs price:** High prices rarely have many reviews; high review counts mostly appear in budget to mid-range listings.

---

### 8.6 Neighbourhood Analysis

* Top 10 most expensive neighborhoods identified via median price.
* Visualizations:

  * Horizontal bar chart of top 10 median prices.
  * Histogram of all neighborhood median prices.

**Insights:**

* Neighborhood median prices are right-skewed: most are $100–$150, with a second bump at $200–$250 and a luxury tail.
* Location is a strong predictor for price.

---

### 8.7 Temporal Data Analysis

* Extracted temporal features from `last_scraped`:

  * `scrape_month`, `scrape_year`, `scrape_quarter`.
* Monthly price trends:

  * Prices lowest in February (~$130), peak June–September (~$154).
  * Listing volume is stable (~5–8% fluctuation).

**Implication:** `scrape_month` is predictive due to seasonal demand-driven price variations.

---

### 8.8 Analysis of Newly Engineered Features

#### 8.8.1 Categorical Features

* `parking_level` and `neighbourhood_cleaned` analyzed:

  * Price varies widely between neighborhood categories.
  * Strong predictor: premium neighborhoods have consistently higher quartiles.
  * Imbalance exists; “Other” category successfully captures rare neighborhoods.

#### 8.8.2 Continuous Features

* Features analyzed: `num_amenities`, `beds_per_guest`, `review_quality_mean`, `days_since_last_scraped`, `crowding`, `host_reliability`.
* Visualizations:

  * Boxplots for features with few unique values.
  * Hexbin plots for continuous features with many unique values.

**Insights:**

* **num_amenities:** Higher counts unlock higher price potential.
* **review_quality_mean:** High ratings necessary to charge high prices.
* **days_since_last_scraped:** No predictive power; should be dropped.
* **host_reliability:** Critical for premium pricing; small reductions below 100% reduce price potential.

#### 8.8.3 Seasonality (Scrape Month)

* Median price trends, listing volume, and violin plots confirm seasonality.
* Prices rise in summer months, consistent across room types.

---

### 8.9 Key Insights Summary

1. **Price distribution:**

   * Highly skewed (skewness ≈ 6.7)
   * Range: $0 to $10,000+
   * Median: ~$150
2. **Most correlated features with price:**

   1. `accommodates` (r ≈ 0.09)
   2. `bedrooms` (r ≈ 0.08)
   3. `beds` (r ≈ 0.07)
   4. `bathrooms` (r ≈ 0.07)
   5. `review_scores_rating` (r ≈ 0.05)
3. **Strong predictors:** Neighborhood, room type, capacity, number of amenities, review quality.
4. **Outliers:** Long right-tail in price; large homes and luxury neighborhoods.
5. **Seasonality:** `scrape_month` shows ~18% variation; important for demand-driven pricing.
6. **Engineered features:** `host_reliability`, `num_amenities`, and `beds_per_guest` show meaningful trends; `days_since_last_scraped` is non-informative.
7. **Visualization improvements:** Capped values, violin, and hexbin plots reveal true patterns previously obscured by extreme values.

---

**Conclusion:**
The deep EDA reveals that **location, capacity, amenities, host quality, and seasonality** are key drivers of price, while some raw features are weak or noisy. Non-linear relationships and extreme outliers indicate that tree-based models (Random Forest, XGBoost) will outperform linear models unless additional feature engineering is applied.


## SECTION 9 — Clustering for Target Creation

### 9.1 Prepare Price Data

Price values from the training set were extracted and standardized for clustering. StandardScaler was applied to normalize the scale, ensuring clustering methods are not biased by extreme values.

### 9.2 Target Creation Methods

**Method 1.1 — Quantile-Based Binning:**
Prices were split into three bins (low, medium, high) using 33% and 66% quantiles. This produces balanced classes but the “high” category is very wide due to extreme outliers.

**Method 1.2 — Equal-Width Binning on Log Scale:**
Price was log-transformed and split into 3 equal-width bins. This approach is useful for outlier detection but less suitable for low/medium/high segmentation because the distribution remains skewed.

**Method 2 — KMeans Clustering:**
KMeans was applied on standardized prices for 2–5 clusters. Metrics (inertia, silhouette, Davies-Bouldin) indicate K=2 as a natural split. K=3 poorly separates tiers; KMeans tends to isolate extreme outliers rather than produce balanced low/medium/high classes.

**Method 3 — Gaussian Mixture Model (GMM):**
GMM with 3 components creates meaningful tiers:

* Low ≈ $3–$275
* Medium ≈ $276–$1235
* High ≈ $1236–$50k

Class distribution is skewed (high ≈ 1.2%), so imbalanced classification handling is needed.

### 9.3 Comparison of Methods

* **Quantile bins** produce balanced classes suitable for training but have poor interpretability at the high end.
* **Equal-width bins** are not ideal for tiered classification.
* **KMeans** on raw price mostly isolates outliers; not suitable for 3-class segmentation.
* **GMM** generates meaningful tiers but results in class imbalance.

### 9.4 Selected Method and Encoding

Quantile-based binning was chosen for simplicity and balanced class distribution:

* Training set and test set were binned using 33% / 66% quantiles.
* Classes were encoded with `LabelEncoder` for model training.
* Class mapping ensures correct correspondence between labels and integer encoding.

**Summary:**
Quantile-based binning provides balanced low/medium/high price categories, making it the most practical choice for supervised classification despite extreme outliers. The encoded target is ready for modeling.


## SECTION 10 — FEATURE SELECTION

**Goal:** Reduce redundancy and validate that the engineered feature space is stable, while ensuring **no leakage** (all selection steps are fit/decided on **training data only**).

### 10.1 Data Cleanup (No Target Leakage)

* Dropped target-related columns from the original split dataframes:

  * `df_train.drop(columns=['price', 'price_quantile'], errors='ignore')`
  * `df_test.drop(columns=['price', 'price_quantile'], errors='ignore')`

* Defined **protected identifier columns** (kept only for reference/grouping, never used as predictors):

  * `protected_cols = ['id', 'host_id']`

* Excluded identifiers and datetime columns from feature selection / modeling:

  * `exclude_from_fs = protected_cols + datetime_cols`

* Removed `host_id` from the working train/test dataframes (not useful for modeling):

  * `df_train.drop(columns=['host_id'], errors='ignore')`
  * `df_test.drop(columns=['host_id'], errors='ignore')`

---

### 10.2 Variance Filter (Training Only)

* Applied a low-variance filter on the **numeric model matrix** (excluding protected + datetime columns):

  * `X_train_fs_num = X_train.drop(columns=exclude_from_fs, errors='ignore')`
  * `X_test_fs_num  = X_test.drop(columns=exclude_from_fs, errors='ignore')`
  * `VarianceThreshold(threshold=0.01)` fitted on `X_train_fs_num`

* Selected features were kept and applied consistently to test:

  * `selected_var_features = X_train_fs_num.columns[var_thresh.get_support()]`
  * `X_test_var = X_test_fs_num[selected_var_features]`

* Reattached excluded columns (for tracking/diagnostics only):

  * `X_train_var = concat([filtered_numeric, X_train[exclude_from_fs]])`
  * `X_test_var  = concat([filtered_numeric, X_test[exclude_from_fs]])`

*Result:* No meaningful features were removed by variance thresholding.

---

### 10.3 Correlation Filter (Training Only)

* Computed absolute Pearson correlations among retained numeric features:

  * `corr_features = X_train_var.drop(columns=exclude_from_fs, errors='ignore')`
  * `corr_matrix = corr_features.corr().abs()`

* Used the upper triangle of the matrix to avoid duplicates:

  * `upper = corr_matrix.where(np.triu(..., k=1).astype(bool))`

* Dropped features where any correlation exceeded the threshold:

  * `to_drop_corr = [col for col in upper.columns if any(upper[col] > 0.9)]`
  * `X_train_corr = X_train_var.drop(columns=to_drop_corr)`
  * `X_test_corr  = X_test_var.drop(columns=to_drop_corr)`

*Result:* Correlation filtering removed only a minimal set of redundant features (in our run, it primarily removed one aggregated feature), indicating that earlier feature engineering already controlled multicollinearity well.

---

### 10.4 Mutual Information (Non-Linear Relevance Check)

* Computed mutual information between numeric predictors and the encoded 3-class target:

  * `X_train_mi = X_train_corr.drop(columns=exclude_from_fs, errors='ignore')`
  * `X_train_mi = X_train_mi.select_dtypes(include=['number'])`
  * `mutual_info_classif(X_train_mi, y_train_encoded, random_state=42)`

* Printed the top MI-ranked features for interpretability.

*Interpretation:* MI confirmed that the most informative signals are consistent with deep EDA findings (capacity, location proxies, and property-type-related predictors).

---

### 10.5 Model-Based Feature Selection (L1-Regularized Logistic Regression)

* Used an L1-style sparse model as a **validation step** (not as the final model family):

  * `LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=1.0, C=0.1, max_iter=5000)`

* Fit on training predictors (excluding protected + datetime columns):

  * `X_train_l1 = X_train_corr.drop(columns=exclude_from_l1, errors='ignore')`
  * `l1_selector.fit(X_train_l1, y_train_encoded)`

* Selected features with non-zero coefficients:

  * `selected_l1_features = X_train_l1.columns[coef_max > 0]`

*Result:* No coefficients were shrunk fully to zero, supporting the conclusion that the current feature set is not overly redundant.

---

### 10.6 Final Modeling Matrices

* Kept the post-selection dataset:

  * `final_features = X_train_corr.columns.tolist()`
  * `X_train_final = X_train_corr[final_features].copy()`
  * `X_test_final  = X_test_corr[final_features].copy()`

* Created the **actual model inputs** by removing identifiers + datetime columns:

  * `exclude_from_model = protected_cols + datetime_cols`
  * `X_train_model = X_train_corr.drop(columns=[c for c in exclude_from_model if c in X_train_corr.columns])`
  * `X_test_model  = X_test_corr.drop(columns=[c for c in exclude_from_model if c in X_test_corr.columns])`

**Outcome:** Feature selection confirmed the engineered predictors are stable and informative. `X_train_model` / `X_test_model` are the final inputs used for training classifiers.

---

## SECTION 11 — GROUP-BASED VALIDATION (LEAKAGE-SAFE)

**Goal:** Create a validation split that preserves **class balance** while ensuring **no listing ID overlap** between train and validation.

### 11.1 Setup

* Configuration:

  * `RANDOM_STATE = 42`
  * `n_splits = 5` (for group-safe stratified CV)

* Groups are listing IDs aligned with the model matrix index:

  * `groups_all = df_train.loc[X_train_model.index, "id"]`
  * Safety check: raise error if `id` is missing from `df_train`

---

### 11.2 Stratified Group Split

* Used **StratifiedGroupKFold** to enforce:

  * **Stratification** (similar class proportions across folds)
  * **Grouping** (no listing ID can appear in both sets)

* Code:

  * `sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)`
  * `train_idx, val_idx = next(sgkf.split(X_train_model, y_train_encoded, groups=groups_all))`

* Produced holdout validation sets:

  * `X_tr, X_val = X_train_model.iloc[train_idx], X_train_model.iloc[val_idx]`
  * `y_tr, y_val = y_train_encoded[train_idx], y_train_encoded[val_idx]`

**Outcome:** Validation is realistic and leakage-safe even with repeated listings across monthly snapshots.

---

## SECTION 12 — EVALUATION METRICS

**Goal:** Evaluate multi-class performance beyond accuracy, with interpretable per-class diagnostics.

### 12.1 Metric Configuration

* Number of classes and class names:

  * `N_CLASSES = 3`
  * `CLASS_NAMES = ["low", "medium", "high"]`

---

### 12.2 Metrics Implemented

* Core metrics:

  * `accuracy`
  * `balanced_accuracy`
  * `macro_f1` (primary for balanced class performance)
  * `weighted_f1`

* Per-class sensitivity (recall):

  * `recall_score(..., average=None)`

* Per-class specificity (computed from confusion matrix):

  * `TN / (TN + FP)` for each class
  * Macro specificity as the mean across classes

* Confusion matrix + classification report:

  * `confusion_matrix(...)`
  * `classification_report(..., digits=4)`

---

### 12.3 Probabilities + ROC-AUC (Optional)

* Probability extraction helper:

  * `get_proba(model, X, n_classes)` with class alignment

* ROC-AUC (only if `predict_proba` exists):

  * Macro AUC: `roc_auc_score(..., multi_class="ovr", average="macro")`
  * Micro AUC: `roc_auc_score(..., multi_class="ovr", average="micro")`

---

### 12.4 ROC Curves (Multi-Class OvR)

* Implemented utilities to compute:

  * per-class ROC curves + AUC
  * micro-average ROC
  * macro-average ROC

* Added a helper to overlay **macro ROC curves across multiple models** for comparison.

---

## SECTION 13 — TRAINING BASELINES + STABILITY CHECKS

**Goal:** Train multiple baseline classifiers and compare them using leakage-safe validation and group-safe cross-validation.

### 13.1 Baseline Models

* Defined a baseline suite including:

  * `DummyClassifier` (sanity check)
  * `RandomForestClassifier`
  * `XGBClassifier`
  * `LGBMClassifier`
  * `CatBoostClassifier`
  * `HistGradientBoostingClassifier`

(Where supported by the environment, boosting models are included as strong defaults for non-linear tabular structure.)

---

### 13.2 Single Holdout Validation (From Section 11)

* For each model:

  * Fit on `X_tr, y_tr`
  * Predict on `X_val`
  * Evaluate using Section 12 metrics (Accuracy, Balanced Acc, Macro F1, Weighted F1, AUC, Sensitivity/Specificity)

* Stored results in a comparison table:

  * `holdout_df = DataFrame(holdout_results).sort_values("Macro F1", ascending=False)`

**Purpose:** Fast, high-signal sanity check before running full CV.

---

### 13.3 Group-Safe 5-Fold Cross-Validation (Mean ± Std)

* Ran cross-validation using the same group-aware splitter:

  * `for fold, (tr_idx, va_idx) in enumerate(cv_splitter.split(X_train_model, y_train_encoded, groups=groups_all), 1): ...`

* Captured fold-by-fold performance and summarized stability:

  * `Acc (mean±std)`
  * `Balanced Acc (mean±std)`
  * `Macro F1 (mean±std)` (primary)
  * `Weighted F1 (mean±std)`
  * `AUC (mean±std)`

**Interpretation:** Mean performance estimates generalization; standard deviation reflects robustness and potential sensitivity to data splits.

---

### 13.4 Baseline Selection Output

* Models are ranked primarily by **Macro F1** (balanced across the three price tiers).
* The best-performing baseline is retained as the reference model for downstream tuning / final reporting.



