# -*- coding: utf-8 -*-
"""MotorPH Products List 2025 - Data Preprocessing

Python script version of [DATA_PREPROCESSING]_MotorPH_Products_List_2025.ipynb (Google Colab format).
Input : [RAW_DATA]_MotorPH_Products_List_2025.csv
Output: [PROCESSED_DATA]_MotorPH_Products_List_2025.csv
"""

# %% [markdown]
# # **MotorPH Products List 2025 – Data Preprocessing**
#
# Source: `[RAW_DATA]_MotorPH_Products_List_2025.csv`  
# Output: `[PROCESSED_DATA]_MotorPH_Products_List_2025.csv`

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# %%
# Load Data Set
# Folder in Google Drive that holds the raw CSV (edit if yours is different)
DRIVE_DIR = "/content/drive/MyDrive/4thYR_Term1/MO-IT106 - Data Analytics Fundamentals/Week 05/Dataset Preprocessing/"
RAW_FILE  = "[RAW_DATA]_MotorPH_Products_List_2025.csv"
GITHUB    = "https://raw.githubusercontent.com/padelacruz85/DAF2026/main/Raw_Data/"

def locate(fname):
    """Google Drive first, then a local Raw_Data/ folder, then the GitHub repo."""
    for p in [DRIVE_DIR + fname, "Raw_Data/" + fname, "../Raw_Data/" + fname]:
        if os.path.exists(p):
            return p
    return GITHUB + fname.replace("[", "%5B").replace("]", "%5D").replace(" ", "%20")

import os
path = locate(RAW_FILE)
df_prodList = pd.read_csv(path)
df_raw = df_prodList.copy()      # untouched backup for before/after comparison
print("Loaded from:", path)

# %% [markdown]
# # **1. INSPECTING THE DATA SET**

# %%
print(df_prodList.head(10))

# %%
print(df_prodList.tail(10))

# %%
print("Rows, columns:", df_prodList.shape)
print(df_prodList.info())

# %%
print(df_prodList.describe(include='all').T)

# %% [markdown]
# # **2. DETECTING ISSUES**

# %%
# 2.1 Missing values (NaN, empty strings, whitespace-only)
print("NaN per column:\n", df_prodList.isna().sum(), "\n")
txt = df_prodList.select_dtypes(include=["object", "string"])
print("Blank strings per column:\n", (txt.apply(lambda s: s.str.strip().eq(""))).sum())

# %%
# 2.2 Duplicates (full-row, ID, and product name)
print("Full-row duplicates   :", df_prodList.duplicated().sum())
print("Duplicate EntrNo      :", df_prodList["EntrNo"].duplicated().sum())
print("Duplicate product name:", df_prodList["EntrName"].str.strip().str.lower().duplicated().sum())
print("EntrNo sequential 1..n:", (df_prodList["EntrNo"] == np.arange(1, len(df_prodList) + 1)).all())

# %%
# 2.3 Formatting: stray whitespace / double spaces in text fields
for col in ["EntrName", "EntrDetails"]:
    s = df_prodList[col]
    print(f"{col}: leading/trailing spaces = {(s != s.str.strip()).sum()}, double spaces = {s.str.contains('  ').sum()}")

# 2.4 Year validity
for col in ["Manufacturing Date", "Acquisiton"]:
    print(col, "->", df_prodList[col].min(), "to", df_prodList[col].max())
print("Acquired before manufactured:", (df_prodList["Acquisiton"] < df_prodList["Manufacturing Date"]).sum())

# %%
# 2.5 Price outliers (IQR rule + z-score)
q1, q3 = df_prodList["UnitPrice"].quantile([.25, .75])
upper = q3 + 1.5 * (q3 - q1)
print("IQR upper fence:", upper)
print(df_prodList[df_prodList["UnitPrice"] > upper])

# %%
sns.boxplot(x=df_prodList["UnitPrice"])
plt.title("UnitPrice – raw data (note the extreme value)")
plt.show()

# %% [markdown]
# ### Findings
# | # | Issue | Evidence |
# |---|-------|----------|
# | 1 | No missing values, no duplicates; `EntrNo` already runs 1–50 | checks above |
# | 2 | `EntrDetails` packs 5 attributes (type, engine, cooling, displacement, transmission) into one text field | e.g. `Sport / Parallel-twin, liquid-cooled, 471cc, 6-speed` |
# | 3 | Inconsistent text conventions (`Naked bike`, `Cafe racer`, `liquid-cooled`, `Parallel-twin` vs `Twin cylinder`) | lower-case second words, mixed hyphenation |
# | 4 | Years and price stored as plain integers (price should be a decimal amount; years should be validated) | `info()` |
# | 5 | Column names unclear / misspelled (`EntrNo`, `EntrName`, `Manufacturing Date` holds a year only, `Acquisiton`) | header row |
# | 6 | **Entry 50 (SYM Husky 150) priced at 7,290,011** – ~7x the next-highest underbone-class price and far outside the IQR fence | outlier check |

# %% [markdown]
# # **3. CROSS-CHECKING THE OUTLIER**
#
# The Q3 sales file records what MotorPH actually charged for this model, so we use it as evidence before changing the value.

# %%
df_sales = pd.read_csv(locate("[RAW_DATA]_MotorPH_Sales Data-3rd Quarter-Year 2025.csv"))
husky = df_sales.loc[df_sales["product"].str.strip() == "SYM Husky 150", "unitprice"].unique()
print("Unit price(s) for SYM Husky 150 in Q3 sales:", husky)

# %%
# The sales price (72,900) matches the listed value with the extra digits "11" appended
# (7290011 -> 72900), i.e. a data-entry error. Correct it using the sales evidence.
bad = df_prodList["EntrName"].eq("SYM Husky 150")
df_prodList.loc[bad, "UnitPrice"] = int(husky[0])
print(df_prodList.loc[bad, ["EntrNo", "EntrName", "UnitPrice"]])

# %% [markdown]
# # **4. PREPROCESSING**

# %%
# 4.1 Missing values & duplicates (none found – kept as safeguards so the pipeline is reusable)
df_prodList = df_prodList.dropna(how="all")
df_prodList = df_prodList.drop_duplicates()
df_prodList = df_prodList.drop_duplicates(subset="EntrName", keep="first")
print(df_prodList.shape)

# %%
# 4.2 Trim whitespace and normalise internal spacing
for col in ["EntrName", "EntrDetails"]:
    df_prodList[col] = df_prodList[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)

# %%
# 4.3 Split EntrDetails into separate attributes
details = df_prodList["EntrDetails"].str.extract(
    r"^(?P<type>[^/]+?)\s*/\s*(?P<engine>[^,]+),\s*(?P<cooling>[^,]+),\s*(?P<cc>\d+)\s*cc,\s*(?P<trans>.+)$"
)
print("Rows that failed to parse:", details.isna().any(axis=1).sum())
print(details.head())

# %%
# 4.4 Standardise labels (consistent capitalisation / hyphenation)
def title_keep_hyphen(s):
    return "-".join(w.capitalize() for w in s.split("-"))

prod_type = details["type"].str.strip().str.title()                      # Naked Bike, Cafe Racer, Dual-Sport
engine    = (details["engine"].str.strip().str.title()
             .str.replace("Parallel-Twin", "Parallel Twin")
             .str.replace("V-Twin", "V-Twin"))
cooling   = details["cooling"].str.strip().apply(lambda s: "/".join(title_keep_hyphen(x) for x in s.split("/")))
trans     = details["trans"].str.strip().str.replace("-speed", "-Speed", regex=False)  # 6-Speed, CVT

prod_type = prod_type.str.replace("Dual-Sport", "Dual-Sport")
print(sorted(prod_type.unique())); print(sorted(engine.unique()))
print(sorted(cooling.unique()));   print(sorted(trans.unique()))

# %%
# 4.5 Build the structured table with correct data types
df_clean = pd.DataFrame({
    "Product ID Number":   df_prodList["EntrNo"].astype("int64"),
    "Product Name":        df_prodList["EntrName"].astype("string"),
    "Product Type":        prod_type.astype("string"),
    "Engine Type":         engine.astype("string"),
    "Cooling System":      cooling.astype("string"),
    "Engine Displacement (cc)": details["cc"].astype("int64"),
    "Transmission":        trans.astype("string"),
    "Manufacturing Year":  df_prodList["Manufacturing Date"].astype("int64"),
    "Acquisition Year":    df_prodList["Acquisiton"].astype("int64"),
    "Unit Price (PHP)":    df_prodList["UnitPrice"].astype("float64").round(2),
}).reset_index(drop=True)

# Product ID: make sequential 1..n in listed order
df_clean["Product ID Number"] = np.arange(1, len(df_clean) + 1)
print(df_clean.info())

# %% [markdown]
# # **5. VALIDATION (REVIEW PASS)**

# %%
checks = {
    "No missing values":              df_clean.isna().sum().sum() == 0,
    "No duplicate rows":              not df_clean.duplicated().any(),
    "No duplicate product names":     not df_clean["Product Name"].duplicated().any(),
    "Product ID sequential 1..n":     (df_clean["Product ID Number"] == np.arange(1, len(df_clean)+1)).all(),
    "Manufacturing year valid (2000-2025)": df_clean["Manufacturing Year"].between(2000, 2025).all(),
    "Acquisition year valid (2000-2025)":   df_clean["Acquisition Year"].between(2000, 2025).all(),
    "Acquired on/after manufacture":  (df_clean["Acquisition Year"] >= df_clean["Manufacturing Year"]).all(),
    "Unit price > 0":                 (df_clean["Unit Price (PHP)"] > 0).all(),
    "No leading/trailing spaces":     all((df_clean[c] == df_clean[c].str.strip()).all()
                                          for c in df_clean.select_dtypes("string").columns),
    "Price outliers (IQR) remaining": (df_clean["Unit Price (PHP)"] > df_clean["Unit Price (PHP)"].quantile(.75)
                                        + 1.5*(df_clean["Unit Price (PHP)"].quantile(.75) - df_clean["Unit Price (PHP)"].quantile(.25))).sum(),
}
for k, v in checks.items():
    print(f"{k:42s}: {v}")

# %%
# Remaining high-priced units are genuine premium models (BMW R nineT, Honda Rebel 1100, ...), not errors
sns.boxplot(x=df_clean["Unit Price (PHP)"]); plt.title("UnitPrice – after cleaning"); plt.show()
print(df_clean.nlargest(5, "Unit Price (PHP)")[["Product Name", "Unit Price (PHP)"]])

# %% [markdown]
# # **6. EXPORT**

# %%
OUT_FILE = "[PROCESSED_DATA]_MotorPH_Products_List_2025.csv"
df_clean.to_csv(OUT_FILE, index=False, encoding="utf-8")        # saved in the Colab session
if IN_COLAB:
    df_clean.to_csv(DRIVE_DIR + OUT_FILE, index=False, encoding="utf-8")   # saved to Google Drive
    from google.colab import files
    files.download(OUT_FILE)                                      # download to your computer
print("Saved:", OUT_FILE, df_clean.shape)
print(df_clean.head(10))
