## ✅ Your Turn – Bivariate Analysis on Sales Dataset

### Step 1: Data Loading and Inspection

We begin by loading the `sales.csv` file and inspecting the structure and missing values.

```python
import pandas as pd
df_sales = pd.read_csv("sales.csv")
df_sales.head()
print(df_sales.info())
print(df_sales.isnull().sum())
```

---

### Step 2: Data Preprocessing

We remove rows with missing data and convert categorical integer columns to string for proper analysis.

```python
df_sales_clean = df_sales.dropna().copy()
df_sales_clean['Store_Type'] = df_sales_clean['Store_Type'].astype(str)
df_sales_clean['City_Type'] = df_sales_clean['City_Type'].astype(str)
```

---

### Step 3: Cramér’s V Correlation Heatmap

We use the `dfcorrs` library to generate a correlation heatmap that includes numerical and categorical variables.

```python
from dfcorrs.cramersvcorr import Cramers
cram = Cramers()
cram.corr(df_sales_clean, plot_htmp=True)
```

---

### Step 4: Pearson Correlation (Numeric Variables)

This heatmap shows the linear relationships between numeric features.

```python
import seaborn as sns
import matplotlib.pyplot as plt

numeric_cols = df_sales_clean.select_dtypes(include='number').columns
sns.heatmap(df_sales_clean[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Pearson Correlation - Sales Dataset")
plt.show()
```

---

### Step 5: Boxplot – Sales by Product Quality

We visualize how sales vary across different levels of product quality.

```python
df_box = df_sales_clean[['Sales', 'Product_Quality']]
sns.boxplot(data=df_box, x='Product_Quality', y='Sales')
plt.title("Sales by Product Quality")
plt.xlabel("Product Quality")
plt.ylabel("Sales")
plt.show()
```

---

This completes the bivariate analysis including preprocessing, correlation, and visualization techniques on the `sales` dataset.
