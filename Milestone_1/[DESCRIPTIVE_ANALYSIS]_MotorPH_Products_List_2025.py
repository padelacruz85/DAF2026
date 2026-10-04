# -*- coding: utf-8 -*-
"""MotorPH Products List 2025 - Descriptive Analysis

Python script version of [DESCRIPTIVE_ANALYSIS]_MotorPH_Products_List_2025.ipynb (Google Colab format).
"""

# %% [markdown]
# # **MotorPH Products List 2025 – Descriptive Analysis**
#
# Converted from `[DATA_PREPROCESSING]_MotorPH_Products_List_2025.ipynb`: the cleaning pipeline now feeds a descriptive analysis of the MotorPH product list.

# %%
# Connecting to google drive (Colab). Skipped automatically outside Colab.
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# %%
#Import libraries
import os
import numpy as np                 #arithmetic and statistical computations
import pandas as pd                #data wrangling and manipulation
import matplotlib.pyplot as plt    #Basic Data Viz
import seaborn as sns              #Intermediate Statistical Data Viz
from scipy import stats            #intermediate statistical computation

sns.set_theme(style="whitegrid")
pd.options.display.float_format = "{:,.2f}".format

# %%
#Load Data Set
DRIVE_DIR = "/content/drive/MyDrive/4thYR_Term1/MO-IT106 - Data Analytics Fundamentals/Week 05/Dataset Preprocessing/"
GITHUB    = "https://raw.githubusercontent.com/padelacruz85/DAF2026/main/Raw_Data/"
PROCESSED = "[PROCESSED_DATA]_MotorPH_Products_List_2025.csv"
RAW_FILE  = "[RAW_DATA]_MotorPH_Products_List_2025.csv"
SALES     = "[RAW_DATA]_MotorPH_Sales Data-3rd Quarter-Year 2025.csv"

def locate(fname):
    """Google Drive first, then local folders, then the GitHub repo (raw files only)."""
    for p in [DRIVE_DIR + fname, fname, "Raw_Data/" + fname, "../Raw_Data/" + fname]:
        if os.path.exists(p):
            return p
    return GITHUB + fname.replace("[", "%5B").replace("]", "%5D").replace(" ", "%20")

PRICE, CC = "Unit Price (PHP)", "Engine Displacement (cc)"

# %% [markdown]
# # **0. PREPARE THE DATA**
#
# Uses the cleaned file produced by the preprocessing notebook. If it is not found, the same cleaning steps are re-applied to the raw file, so this notebook always runs on clean data.

# %%
def preprocess(raw, sales):
    """Same cleaning steps as the [DATA_PREPROCESSING] notebook."""
    d = raw.drop_duplicates().drop_duplicates(subset="EntrName").copy()
    for col in ["EntrName", "EntrDetails"]:
        d[col] = d[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    # Price outlier: SYM Husky 150 listed at 7,290,011; Q3 sales show 72,900 (data-entry error)
    husky = sales.loc[sales["product"].str.strip() == "SYM Husky 150", "unitprice"].unique()
    d.loc[d["EntrName"].eq("SYM Husky 150"), "UnitPrice"] = int(husky[0])
    det = d["EntrDetails"].str.extract(
        r"^(?P<type>[^/]+?)\s*/\s*(?P<engine>[^,]+),\s*(?P<cooling>[^,]+),\s*(?P<cc>\d+)\s*cc,\s*(?P<trans>.+)$")
    hy = lambda s: "-".join(w.capitalize() for w in s.split("-"))
    return pd.DataFrame({
        "Product ID Number": np.arange(1, len(d) + 1),
        "Product Name":   d["EntrName"].astype("string").values,
        "Product Type":   det["type"].str.strip().str.title().values,
        "Engine Type":    det["engine"].str.strip().str.title().str.replace("Parallel-Twin", "Parallel Twin").values,
        "Cooling System": det["cooling"].str.strip().apply(lambda s: "/".join(hy(x) for x in s.split("/"))).values,
        CC:               det["cc"].astype("int64").values,
        "Transmission":   det["trans"].str.strip().str.replace("-speed", "-Speed", regex=False).values,
        "Manufacturing Year": d["Manufacturing Date"].astype("int64").values,
        "Acquisition Year":   d["Acquisiton"].astype("int64").values,
        PRICE:            d["UnitPrice"].astype("float64").round(2).values,
    })

proc_path = next((p for p in [DRIVE_DIR + PROCESSED, PROCESSED, "../" + PROCESSED, "Milestone_1/" + PROCESSED] if os.path.exists(p)), None)
if proc_path:
    df = pd.read_csv(proc_path)
    print("Loaded cleaned data from:", proc_path)
else:
    df = preprocess(pd.read_csv(locate(RAW_FILE)), pd.read_csv(locate(SALES)))
    print("Cleaned file not found - rebuilt from raw data.")

df["Brand"] = df["Product Name"].str.split().str[0]        # derived: manufacturer (first word of the name)
df["Years to Acquire"] = df["Acquisition Year"] - df["Manufacturing Year"]
print(df.head(10))

# %% [markdown]
# # **1. DATA SET OVERVIEW**

# %%
print("Rows, columns:", df.shape)
print(df.info())

# %%
# Quick quality confirmation
print("Missing values :", df.isna().sum().sum())
print("Duplicate rows :", df.duplicated().sum())
print("Unique brands  :", df["Brand"].nunique())
print("Product types  :", df["Product Type"].nunique())

# %% [markdown]
# # **2. NUMERICAL VARIABLES**

# %%
num_cols = [PRICE, CC, "Manufacturing Year", "Acquisition Year"]
desc = df[num_cols].describe().T
desc["median"] = df[num_cols].median()
desc["range"] = desc["max"] - desc["min"]
desc["IQR"] = desc["75%"] - desc["25%"]
desc["CV (%)"] = desc["std"] / desc["mean"] * 100
desc["skewness"] = df[num_cols].skew()
desc["kurtosis"] = df[num_cols].kurt()
print(desc[["count","mean","median","std","min","25%","75%","max","range","IQR","CV (%)","skewness","kurtosis"]])

# %%
# Mode of each numerical field
for col in num_cols:
    print(f"{col:28s} mode(s): {df[col].mode().tolist()}")

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
sns.histplot(df[PRICE], bins=12, kde=True, ax=axes[0,0], color="#2a6f97")
axes[0,0].axvline(df[PRICE].mean(),   color="red",   ls="--", label="Mean")
axes[0,0].axvline(df[PRICE].median(), color="green", ls="--", label="Median")
axes[0,0].set_title("Distribution of Unit Price"); axes[0,0].legend()
sns.boxplot(x=df[PRICE], ax=axes[0,1], color="#89c2d9"); axes[0,1].set_title("Unit Price – Boxplot")
sns.histplot(df[CC], bins=12, kde=True, ax=axes[1,0], color="#e76f51"); axes[1,0].set_title("Distribution of Engine Displacement (cc)")
sns.boxplot(x=df[CC], ax=axes[1,1], color="#f4a261"); axes[1,1].set_title("Engine Displacement – Boxplot")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Interpretation – numerical variables
# - **Unit price** averages about **₱257,872** but the median is only **₱204,950**. The mean sits well above the median and the skewness is positive (**1.83**), so the distribution is right-skewed: most models are mid-priced and a few premium bikes pull the average up.
# - Prices are widely spread (std ≈ ₱183,425, **CV ≈ 71%**), ranging from **₱48,000** (Rusi Flash 125) to **₱995,000** (BMW R nineT). The middle 50% of products falls between ₱126,425 and ₱330,250.
# - **Engine displacement** ranges from 124 cc to 1,170 cc with a median of **241 cc** (mean 332 cc), also right-skewed: the catalogue is dominated by small-to-mid displacement motorcycles.
# - **Manufacturing years** span 2020–2023 and **acquisition years** 2021–2024, with the median acquisition in 2023.

# %% [markdown]
# # **3. CATEGORICAL VARIABLES**

# %%
def freq(col):
    t = df[col].value_counts().to_frame("Count")
    t["Percent (%)"] = (t["Count"] / len(df) * 100).round(1)
    return t

for col in ["Product Type", "Engine Type", "Cooling System", "Transmission", "Brand"]:
    print(f"--- {col} ---")
    display(freq(col).head(10)) if "display" in globals() else print(freq(col).head(10))
    print()

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, col in zip(axes.flat, ["Product Type", "Engine Type", "Cooling System", "Transmission"]):
    order = df[col].value_counts().index
    sns.countplot(y=df[col], order=order, ax=ax, color="#2a6f97")
    ax.set_title(f"Products by {col}")
    for cont in ax.containers: ax.bar_label(cont, padding=2)
plt.tight_layout(); plt.show()

top = df["Brand"].value_counts().head(8)
top.sort_values().plot(kind="barh", figsize=(7,4), color="#e76f51", title="Top 8 brands by number of models")
plt.xlabel("Number of models"); plt.show()

# %% [markdown]
# ### Interpretation – categorical variables
# - **Product type:** Scooters are the largest group (**12 of 50, 24%**), followed by Naked Bikes (7). Sport, Adventure and Cruiser have 4 each; the other 12 types have only 1–3 models each.
# - **Engine type:** **Single-cylinder** engines dominate (**37 models, 74%**); Parallel Twins are next (9). V-Twin (2), Flat Twin (1) and Twin Cylinder (1) are niche.
# - **Cooling:** **Liquid-cooled** is the majority (**34, 68%**), then air-cooled (13); oil-cooled and air/oil-cooled are rare.
# - **Transmission:** **6-speed** manuals lead (**30, 60%**), followed by CVT (13, mostly scooters), 5-speed (5) and 4-speed (2).
# - **Brands:** The 50 models come from **23 brands**. Yamaha (8) has the most models, then Honda (5) and Suzuki (4).

# %% [markdown]
# # **4. TIME PERIOD ANALYSIS**

# %%
yr = df.groupby("Manufacturing Year")[PRICE].agg(["count","mean","median","min","max"])
display(yr) if "display" in globals() else print(yr)
print(df["Years to Acquire"].value_counts().rename("Count").to_frame())

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.countplot(x=df["Manufacturing Year"], ax=axes[0], color="#2a6f97"); axes[0].set_title("Models by Manufacturing Year")
sns.countplot(x=df["Acquisition Year"],  ax=axes[1], color="#e76f51"); axes[1].set_title("Models by Acquisition Year")
for ax in axes:
    for cont in ax.containers: ax.bar_label(cont)
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Interpretation – time period
# - The product line is **recent**: 21 of 50 models (42%) were manufactured in 2023 and 37 (74%) in 2022–2023. Only 5 date from 2020.
# - Acquisition follows manufacturing closely: **49 of 50 models were acquired 1 year after manufacture**; the exception is the Honda CBR500R (made 2021, acquired 2023, a 2-year gap).
# - Newer model years are priced higher on average (₱164,200 for 2021 versus ₱321,262 for 2023), but the correlation with price is weak (r ≈ 0.29), so model year is not a strong price driver on its own.

# %% [markdown]
# # **5. PRICE BY CATEGORY**

# %%
for col in ["Product Type", "Engine Type", "Cooling System", "Transmission"]:
    t = (df.groupby(col)[PRICE].agg(["count","mean","median","min","max"])
           .sort_values("mean", ascending=False))
    print(f"--- Unit price by {col} ---")
    display(t) if "display" in globals() else print(t)
    print()

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
order = df.groupby("Product Type")[PRICE].median().sort_values(ascending=False).index
sns.boxplot(data=df, y="Product Type", x=PRICE, order=order, ax=axes[0], color="#89c2d9")
axes[0].set_title("Unit Price by Product Type")
order = df.groupby("Engine Type")[PRICE].median().sort_values(ascending=False).index
sns.boxplot(data=df, y="Engine Type", x=PRICE, order=order, ax=axes[1], color="#f4a261")
axes[1].set_title("Unit Price by Engine Type")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Interpretation – price by category
# - **Heritage** (avg ₱842,000) and **Cruiser** (₱474,000) are the most expensive types. **Commuter** (₱57,950), **Standard** (₱68,900) and **Street** (₱99,900) are the cheapest.
# - **Scooters**, the largest group, are mid-to-low priced (median ₱158,400), though they range from ₱79,900 (Suzuki Avenis 125) to ₱398,000 (Vespa GTS Super 300).
# - Price rises with engine complexity: single-cylinder bikes average **₱178,719**, parallel twins **₱418,889** and V-twins **₱594,000**. The single flat-twin (BMW R nineT) is the highest-priced item at ₱995,000.
# - Liquid-cooled models average ₱270,509 and 6-speed models ₱320,417, compared with ₱178,338 for CVT and ₱57,950 for 4-speed. Type groups with only 1–2 models give weak averages, so read them as indications only.

# %% [markdown]
# # **6. RELATIONSHIPS BETWEEN NUMERICAL VARIABLES**

# %%
corr = df[num_cols].corr()
display(corr) if "display" in globals() else print(corr)
r, p = stats.pearsonr(df[CC], df[PRICE]); rho, p2 = stats.spearmanr(df[CC], df[PRICE])
print(f"\nPrice vs Displacement: Pearson r = {r:.2f} (p = {p:.1e}); Spearman rho = {rho:.2f} (p = {p2:.1e})")

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=axes[0]); axes[0].set_title("Correlation Matrix")
sns.regplot(data=df, x=CC, y=PRICE, ax=axes[1], scatter_kws={"alpha": .7}, line_kws={"color": "red"})
axes[1].set_title("Unit Price vs Engine Displacement")
plt.tight_layout(); plt.show()

# %%
# Median price per displacement band
bands = pd.cut(df[CC], [0,150,160,250,400,700,2000], labels=["≤150","151–160","161–250","251–400","401–700",">700"])
band_tbl = df.groupby(bands, observed=True)[PRICE].agg(["count","median"])
display(band_tbl) if "display" in globals() else print(band_tbl)

# %% [markdown]
# ### Interpretation – relationships
# - **Engine displacement is the strongest driver of price** (Pearson **r ≈ 0.93**): bigger engines carry much higher prices. Median price climbs from about ₱80,900 for bikes up to 150 cc to ₱650,000 for those above 700 cc.
# - Manufacturing and acquisition year are almost perfectly correlated (r ≈ 0.99), as expected from the 1-year gap, and each has only a weak positive link to price (r ≈ 0.3).
# - Correlation shows association, not cause, and with 50 products these figures are descriptive only.

# %% [markdown]
# # **7. OUTLIERS AND EXTREMES**

# %%
q1, q3 = df[PRICE].quantile([.25, .75]); upper = q3 + 1.5*(q3 - q1)
print("IQR upper fence:", f"{upper:,.0f}")
print("\nPrice outliers (IQR rule):")
display(df.loc[df[PRICE] > upper, ["Product Name", "Product Type", CC, PRICE]]) if "display" in globals() else print(df.loc[df[PRICE] > upper, ["Product Name", PRICE]])
print("\nTop 5 highest priced:");  print(df.nlargest(5, PRICE)[["Product Name", PRICE]].to_string(index=False))
print("\nTop 5 lowest priced:");   print(df.nsmallest(5, PRICE)[["Product Name", PRICE]].to_string(index=False))
print("\nPrice brackets:")
print(pd.cut(df[PRICE], [0,100000,200000,350000,np.inf], labels=["≤100K","100K–200K","200K–350K",">350K"]).value_counts().sort_index())

# %% [markdown]
# ### Interpretation – outliers
# - Three models exceed the IQR upper fence (about ₱636,000): **BMW R nineT (₱995,000)**, **Moto Guzzi V7 Stone (₱689,000)** and **Honda Rebel 1100 (₱650,000)**. These are genuine premium/heritage models, not data errors, so they were kept.
# - The lowest-priced items are Rusi Flash 125 (₱48,000), Bajaj CT125 (₱67,900) and Keeway RKS 150 Sport (₱68,900).
# - Price brackets are balanced in the middle: 9 models at ₱100K or below, 16 at ₱100K–200K, 16 at ₱200K–350K and 9 above ₱350K.

# %% [markdown]
# # **8. SUMMARY OF KEY FINDINGS**
# 1. The catalogue has **50 products from 23 brands**, with no missing or duplicate records.
# 2. **Prices are right-skewed**: mean ₱257,872 vs median ₱204,950, with a range of ₱48,000–₱995,000.
# 3. The typical product is a **single-cylinder (74%), liquid-cooled (68%), 6-speed (60%)** motorcycle; **scooters** are the largest type (24%).
# 4. **Engine displacement drives price** (r ≈ 0.93); heritage and cruiser types and multi-cylinder engines are the most expensive.
# 5. The line-up is **recent**: 74% of models were made in 2022–2023, and nearly all were acquired one year after manufacture.
# 6. Three premium models are statistical price outliers but are valid data.
