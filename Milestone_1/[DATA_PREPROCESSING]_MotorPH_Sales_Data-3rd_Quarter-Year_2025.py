# -*- coding: utf-8 -*-
"""MotorPH Sales Data (3rd Quarter 2025) - Data Preprocessing

Python script version of the Colab notebook.
Input : [RAW_DATA]_MotorPH_Sales Data-3rd Quarter-Year 2025.csv + cleaned product list
Output: [CLEANED_DATA]_MotorPH_Sales_Data-3rd_Quarter-Year_2025.csv
"""

# %% [markdown]
# # **MotorPH Sales Data (3rd Quarter 2025) – Data Preprocessing**
#
# Source: `[RAW_DATA]_MotorPH_Sales Data-3rd Quarter-Year 2025.csv`  
# Reference: `[CLEANED_DATA]_MotorPH_Products_List_2025.csv` (cleaned product list)  
# Output: `[CLEANED_DATA]_MotorPH_Sales_Data-3rd_Quarter-Year_2025.csv`

# %%
# Connecting to google drive (Colab). Outside Colab this step is skipped automatically.
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
print("Running in Colab:", IN_COLAB)

# %%
# Import libraries
import os, re
import numpy as np                 # arithmetic and statistical computations
import pandas as pd                # data wrangling and manipulation
import matplotlib.pyplot as plt    # Basic Data Viz
import seaborn as sns              # Intermediate Statistical Data Viz

# %%
# Load Data Sets
# Folder in Google Drive that holds the CSV files (edit if yours is different)
DRIVE_DIR = "/content/drive/MyDrive/4thYR_Term1/MO-IT106 - Data Analytics Fundamentals/Week 05/Dataset Preprocessing/"
GITHUB    = "https://raw.githubusercontent.com/padelacruz85/DAF2026/main/"
SALES_FILE = "[RAW_DATA]_MotorPH_Sales Data-3rd Quarter-Year 2025.csv"
PROD_FILES = ["[CLEANED_DATA]_MotorPH_Products_List_2025.csv", "[PROCESSED_DATA]_MotorPH_Products_List_2025.csv"]

def locate(fname, folder):
    # Google Drive first, then local folders, then the GitHub repo.
    for p in [DRIVE_DIR + fname, fname, f"{folder}/{fname}", f"../{folder}/{fname}"]:
        if os.path.exists(p):
            return p
    return GITHUB + folder + "/" + fname.replace("[", "%5B").replace("]", "%5D").replace(" ", "%20")

sales_path = locate(SALES_FILE, "Raw_Data")
df_sales = pd.read_csv(sales_path)
df_raw = df_sales.copy()          # untouched backup for before/after comparison
print("Sales data loaded from:", sales_path)

# Cleaned product list = reference catalogue (product names, IDs, list prices)
prod_path = next((locate(f, "Milestone_1") for f in PROD_FILES
                  if os.path.exists(DRIVE_DIR + f) or os.path.exists(f) or os.path.exists("Milestone_1/" + f)
                  or os.path.exists("../Milestone_1/" + f)), locate(PROD_FILES[0], "Milestone_1"))
df_prod = pd.read_csv(prod_path)
print("Product list loaded from:", prod_path, df_prod.shape)

# %% [markdown]
# # **1. INSPECTING THE DATA SET**

# %%
print(df_sales.head(10))

# %%
print(df_sales.tail(10))

# %%
print("Rows, columns:", df_sales.shape)
print(df_sales.info())

# %%
print(df_sales.describe(include='all').T)

# %% [markdown]
# # **2. DETECTING ISSUES**

# %%
# 2.1 Missing values (NaN, empty strings, whitespace-only)
print("NaN per column:\n", df_sales.isna().sum(), "\n")
txt = df_sales.select_dtypes(include=["object", "string"])
print("Blank strings per column:\n", txt.apply(lambda s: s.str.strip().eq("")).sum())
print("\nRows with at least one missing value:", df_sales.isna().any(axis=1).sum())
print(df_sales[df_sales.isna().any(axis=1)].head(15))

# %%
# 2.2 Duplicates
print("Full-row duplicates:", df_sales.duplicated().sum())
print("Duplicates ignoring date (same product/qty/price/client/payment on different days):",
      df_sales.duplicated(subset=[c for c in df_sales.columns if c != "date"]).sum(), "(coincidences, not true duplicates)")

# %%
# 2.3 Data types and date formats
print(df_sales.dtypes, "\n")
# Convert every date to a pattern (9 = digit, a = letter) to reveal mixed formats
pattern = df_sales["date"].dropna().str.replace(r"\d", "9", regex=True).str.replace(r"[A-Za-z]", "a", regex=True)
fmt_counts = pattern.value_counts()
print(fmt_counts)
print("\nNon-standard date values:")
standard = pd.to_datetime(df_sales["date"], format="%m/%d/%Y", errors="coerce")
print(df_sales.loc[standard.isna() & df_sales["date"].notna(), "date"].value_counts())

# %%
# 2.4 Categorical fields
for col in ["client_type", "payment"]:
    print(f"--- {col} ---")
    print(df_sales[col].value_counts(dropna=False), "\n")

# %%
# 2.5 Product names vs. the cleaned product list
valid_products = set(df_prod["Product Name"])
unmatched = df_sales.loc[~df_sales["product"].isin(valid_products), "product"]
print("Distinct product names in sales:", df_sales["product"].nunique(), "| in product list:", len(valid_products))
print("Rows with a name not in the product list:", len(unmatched))
print(unmatched.value_counts())

# %%
# 2.6 Consistency of unit price and total
print("total != unitprice x quantity :", (df_sales["total"] != df_sales["unitprice"] * df_sales["quantity"]).sum(), "rows")

list_price = df_sales["product"].map(df_prod.set_index("Product Name")["Unit Price (PHP)"])
chk = df_sales.assign(list_price=list_price).dropna(subset=["list_price"])
print("unitprice != catalogue price  :", (chk["unitprice"] != chk["list_price"]).sum(), "rows")
print("total != catalogue price x qty:", (chk["total"] != chk["list_price"] * chk["quantity"]).sum(), "rows")
bad = chk[chk["unitprice"] != chk["list_price"]].copy()
bad["price_ratio"] = (bad["unitprice"] / bad["list_price"]).round(2)
print("\nRatio of recorded price to catalogue price:\n", bad["price_ratio"].value_counts())
print(bad[["product", "unitprice", "list_price", "quantity", "total", "price_ratio"]].head(8))

# %%
# 2.7 Ranges (quantity, price, total)
print(df_sales[["unitprice", "quantity", "total"]].describe().round(1))
print("\nZero/negative values:", (df_sales[["unitprice", "quantity", "total"]] <= 0).sum().to_dict())
fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
sns.boxplot(x=df_sales["unitprice"], ax=axes[0], color="#89c2d9"); axes[0].set_title("unitprice – raw (note the extreme values)")
sns.boxplot(x=df_sales["quantity"],  ax=axes[1], color="#f4a261"); axes[1].set_title("quantity – raw")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Findings
# | # | Issue | Evidence |
# |---|-------|----------|
# | 1 | **Dates stored as text, in mixed formats** (`7/22/2025`, `07-25-25`, `Aug-10-2025`, `13/01/2025`) | pattern table in 2.3 |
# | 2 | **Invalid dates** that cannot exist (`2025/07/32`, `2025-08-00`, `08-40-2025`) and **2 blank dates** | 2.1, 2.3 |
# | 3 | **10 blank `client_type`** and **10 blank `payment`** values | 2.1 |
# | 4 | **13 distinct product names are corrupted** (last character replaced by `x`, e.g. `Honda ADV 16x`, `Yamaha MT-1x`) – 15 rows that do not match the product list | 2.5 |
# | 5 | **20 rows have a unit price exactly 3x the catalogue price** (19 are visible in 2.6; the 20th has a corrupted product name), while `total` still equals catalogue price x quantity – the unit price is wrong, the total is right | 2.6 |
# | 6 | No full-row duplicates; no zero/negative quantity, price or total | 2.2, 2.7 |
# | 7 | Column names are lower-case/unclear for reporting (`unitprice`, `total`, `payment`) | header row |

# %% [markdown]
# # **3. PREPROCESSING**

# %%
# 3.1 Work on a copy; trim whitespace and normalise spacing in all text fields
df = df_raw.copy()
for col in ["date", "client_type", "product", "payment"]:
    df[col] = df[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
print("Rows before cleaning:", len(df))

# %%
# 3.2 Standardise category labels and handle missing categories
# Client type / payment cannot be inferred from other fields, so missing entries are kept as "Unknown"
# (this keeps the sales amounts in the analysis instead of discarding the whole transaction).
df["client_type"] = df["client_type"].str.title().fillna("Unknown")
df["payment"]     = df["payment"].str.title().fillna("Unknown")
print(df["client_type"].value_counts(), "\n")
print(df["payment"].value_counts())

# %%
# 3.3 Repair corrupted product names
# A corrupted name differs from exactly ONE catalogue name by exactly ONE character; only unambiguous matches are fixed.
catalogue = list(df_prod["Product Name"])

def repair_name(name):
    if name in valid_products:
        return name
    cands = [p for p in catalogue if len(p) == len(name) and sum(a != b for a, b in zip(p, name)) == 1]
    return cands[0] if len(cands) == 1 else pd.NA

df["product_fixed"] = df["product"].map(repair_name)
mapping = (df.loc[df["product"] != df["product_fixed"], ["product", "product_fixed"]]
             .value_counts().rename("rows").reset_index())
print("Names that could not be repaired:", df["product_fixed"].isna().sum())
print(mapping)

# %%
df["product"] = df["product_fixed"]
df = df.drop(columns="product_fixed")
print("All product names now in the product list:", df["product"].isin(valid_products).all())

# %%
# 3.4 Standardise dates: try the known formats in order, then validate
DATE_FORMATS = ["%m/%d/%Y",   # standard in this file
                "%m-%d-%Y", "%m-%d-%y", "%b-%d-%Y", "%Y-%m-%d", "%Y/%m/%d",
                "%d/%m/%Y"]   # day-first is tried last so it can only rescue values the others cannot read

def parse_date(v):
    if pd.isna(v):
        return pd.NaT
    for f in DATE_FORMATS:
        try:
            return pd.to_datetime(v, format=f)
        except ValueError:
            continue
    return pd.NaT

df["date_parsed"] = df["date"].map(parse_date)

# Valid window = the period covered by the correctly formatted records
std_dates = pd.to_datetime(df_raw["date"], format="%m/%d/%Y", errors="coerce")
win_start, win_end = std_dates.min(), std_dates.max()
print("Sales period in the file:", win_start.date(), "to", win_end.date())

in_window = df["date_parsed"].between(win_start, win_end)
rescued = df.loc[(std_dates.isna().values) & in_window.values & df["date"].notna(), ["date", "date_parsed"]]
print("\nDates recovered from non-standard formats:")
print(rescued)

# %%
# Records whose date is missing, impossible, or outside the sales period cannot be placed in time -> removed
invalid = df[~in_window]
print("Rows removed (missing / invalid / out-of-period date):", len(invalid))
print(invalid[["date", "client_type", "product", "quantity", "total"]])

# %%
df = df[in_window].copy()
df["date"] = df["date_parsed"].dt.normalize()
df = df.drop(columns="date_parsed")
print("Rows remaining:", len(df), "| date range:", df["date"].min().date(), "to", df["date"].max().date())

# %%
# 3.5 Correct unit price
# In 20 rows the recorded unit price is exactly 3x the catalogue price while the total matches catalogue price x quantity,
# so the unit price (not the total) is the wrong value. Restore it from the product list.
cat_price = df["product"].map(df_prod.set_index("Product Name")["Unit Price (PHP)"])
fixed = df.loc[df["unitprice"] != cat_price, ["product", "unitprice", "quantity", "total"]].assign(corrected_price=cat_price)
print("Unit prices corrected:", len(fixed))
df["unitprice"] = cat_price
print("unitprice x quantity == total for every row:", (df["unitprice"] * df["quantity"] == df["total"]).all())
print(fixed.head(10))

# %%
# 3.6 Remove duplicates (safeguard – none expected after cleaning)
before = len(df)
df = df.drop_duplicates()
print("Duplicate rows removed:", before - len(df))

# %%
# 3.7 Build the structured table: rename columns, correct types, sequential ID, column order
product_id = df["product"].map(df_prod.set_index("Product Name")["Product ID Number"])

df_clean = pd.DataFrame({
    "Date":              df["date"],
    "Client Type":       df["client_type"].astype("string"),
    "Product ID Number": product_id.astype("int64"),
    "Product Name":      df["product"].astype("string"),
    "Unit Price (PHP)":  df["unitprice"].astype("float64").round(2),
    "Quantity":          df["quantity"].astype("int64"),
    "Total Sales (PHP)": df["total"].astype("float64").round(2),
    "Payment Method":    df["payment"].astype("string"),
})

# Chronological order (stable sort keeps the original order within a day), then a sequential Transaction ID
df_clean = df_clean.sort_values("Date", kind="stable").reset_index(drop=True)
df_clean.insert(0, "Transaction ID", np.arange(1, len(df_clean) + 1))
print(df_clean.info())

# %% [markdown]
# # **4. VALIDATION (REVIEW PASS)**

# %%
checks = {
    "No missing values":                         df_clean.isna().sum().sum() == 0,
    "No duplicate rows":                         not df_clean.drop(columns="Transaction ID").duplicated().any(),
    "Transaction ID sequential 1..n":            (df_clean["Transaction ID"] == np.arange(1, len(df_clean) + 1)).all(),
    "Dates are valid datetimes":                 pd.api.types.is_datetime64_any_dtype(df_clean["Date"]),
    "All dates in 2025 sales period":            df_clean["Date"].between(win_start, win_end).all(),
    "Every product name is in product list":     df_clean["Product Name"].isin(valid_products).all(),
    "Product ID matches product name":           (df_clean["Product ID Number"] == df_clean["Product Name"].map(df_prod.set_index("Product Name")["Product ID Number"])).all(),
    "Unit price equals catalogue price":         (df_clean["Unit Price (PHP)"] == df_clean["Product Name"].map(df_prod.set_index("Product Name")["Unit Price (PHP)"])).all(),
    "Total = unit price x quantity":             (df_clean["Unit Price (PHP)"] * df_clean["Quantity"] == df_clean["Total Sales (PHP)"]).all(),
    "Quantity, price, total > 0":                (df_clean[["Quantity", "Unit Price (PHP)", "Total Sales (PHP)"]] > 0).all().all(),
    "No leading/trailing spaces":                all((df_clean[c] == df_clean[c].str.strip()).all() for c in df_clean.select_dtypes("string").columns),
}
for k, v in checks.items():
    print(f"{k:42s}: {v}")
print("\nClient Type :", df_clean["Client Type"].value_counts().to_dict())
print("Payment     :", df_clean["Payment Method"].value_counts().to_dict())

# %%
# Before / after summary
summary = pd.DataFrame({
    "Raw": [len(df_raw), df_raw.isna().sum().sum(), df_raw["product"].nunique(), (df_raw["unitprice"] * df_raw["quantity"] != df_raw["total"]).sum(), df_raw["total"].sum()],
    "Cleaned": [len(df_clean), df_clean.isna().sum().sum(), df_clean["Product Name"].nunique(), (df_clean["Unit Price (PHP)"] * df_clean["Quantity"] != df_clean["Total Sales (PHP)"]).sum(), df_clean["Total Sales (PHP)"].sum()],
}, index=["Rows", "Missing values", "Distinct products", "Rows where total != price x qty", "Total sales (PHP)"])
print(summary.style.format("{:,.0f}") if hasattr(summary, "style") else summary)

# %%
sns.boxplot(x=df_clean["Unit Price (PHP)"], color="#89c2d9")
plt.title("Unit Price – after cleaning"); plt.show()
print(df_clean.head(10))

# %% [markdown]
# # **5. EXPORT**

# %%
OUT_FILE = "[CLEANED_DATA]_MotorPH_Sales_Data-3rd_Quarter-Year_2025.csv"
df_clean.to_csv(OUT_FILE, index=False, encoding="utf-8")                       # saved in the Colab session
if IN_COLAB:
    df_clean.to_csv(DRIVE_DIR + OUT_FILE, index=False, encoding="utf-8")       # saved to Google Drive
    from google.colab import files
    files.download(OUT_FILE)                                                   # download to your computer
print("Saved:", OUT_FILE, df_clean.shape)
