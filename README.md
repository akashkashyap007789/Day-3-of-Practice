#Day 3 — One-Hot Encoding (Pandas & Scikit-Learn)

This project focuses on transforming categorical variables into machine-readable numerical formats using multiple One-Hot Encoding techniques. The goal is to understand how encoding works, why it's needed, and how different methods affect data preparation for machine learning models.

📘 Overview

The dataset (cars.csv) contains information such as:

brand

km_driven

fuel

owner

On Day 3, the objective was to:

Explore categorical columns

Apply One-Hot Encoding using different methods

Compare Pandas and Scikit-Learn implementations

Handle high-cardinality categories using “top categories” logic

This is a core step in feature engineering before training ML models.

🚀 What This Notebook Covers
1. One-Hot Encoding using Pandas

Applied pd.get_dummies() on fuel and owner.

Generated dummy columns for each category.

Useful for quick EDA and simple models.

2. K-1 Encoding (Avoiding Dummy Variable Trap)

Used drop_first=True with get_dummies.

Eliminates one column per feature to avoid multicollinearity.

Keeps the information while reducing redundancy.

3. One-Hot Encoding using Scikit-Learn

Steps performed:

Train–Test Split
Split dataset into training and test sets using:

train_test_split(df.iloc[:,0:4], df.iloc[:,-1], test_size=0.2, random_state=2)


Encoding with OneHotEncoder
Encoded fuel and owner with:

OneHotEncoder(drop='first', sparse_output=False, dtype=np.int32)


Combining Encoded Features
Merged encoded categorical arrays with numerical columns using np.hstack().

This method is essential for pipelines and model training workflows.

4. Handling High-Cardinality Features

Checked how many unique categories brand has.

Identified rare brands using a frequency threshold (e.g., ≤100).

Replaced rare categories with "uncommon".

Encoded the simplified column using pd.get_dummies().

This prevents excessive sparse features and improves model stability.


🧠 Key Takeaways

Encoding is mandatory before feeding categorical features into ML models.

Pandas is simple and fast, but Scikit-Learn is better for production workflows.

Dropping the first category avoids collinearity issues.

📅 Next Step

Day 4: Practice ColumnTransformer — apply multiple transformers together (numeric + categorical), create a clean preprocessing pipeline, and prepare data for model training.
High-cardinality categories must be grouped or simplified to avoid huge feature spaces.

Understanding encoding is essential for any ML project pipeline.
