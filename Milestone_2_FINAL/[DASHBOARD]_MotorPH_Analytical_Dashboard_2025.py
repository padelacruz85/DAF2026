# %% [markdown]
# # **MotorPH Analytical Dashboard – Q3 2025 (June–August)**
#
# **Milestone 2 – MO-IT106 Data Analytics Fundamentals**
#
# This notebook builds the final MotorPH dashboard. It:
# 1. loads the cleaned Products and Sales datasets from Milestone 1 and **documents a data-quality review**,
# 2. computes the **key metrics** (units, revenue, products, categories, transactions),
# 3. builds **matplotlib charts**: sales trends over time, revenue and units comparisons, and category performance,
# 4. compiles everything into a four-section dashboard
#    (**MotorPH Overview · Sales Performance · Sales by Product Category · Inventory**), and
# 5. saves the outputs to the GitHub repository folder (`Milestone_2_FINAL`) and to Google Drive.

# %%
# Connect to Google Drive (Colab). Skipped automatically outside Colab.
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# %%
# Import libraries
import os
import subprocess
import numpy as np                      # numeric helpers
import pandas as pd                     # data wrangling and manipulation
import matplotlib
if not IN_COLAB and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")               # headless run (no display) - Colab/Jupyter keep their own backend
import matplotlib.pyplot as plt         # data visualization
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

pd.options.display.float_format = "{:,.2f}".format

# %%
# Paths and file names
DRIVE_DATA = "/content/drive/MyDrive/4thYR_Term1/MO-IT106 - Data Analytics Fundamentals/Week 08/DATA/"
DRIVE_OUT  = ("/content/drive/MyDrive/4thYR_Term1/MO-IT106 - Data Analytics Fundamentals/Week 08/"
              "[FINAL] MO-IT106 | Milestone 2 | Dela Cruz, P.P./")

REPO_URL   = "https://github.com/padelacruz85/DAF2026.git"
GITHUB_RAW = "https://raw.githubusercontent.com/padelacruz85/DAF2026/main/Milestone_2_FINAL/"
TARGET_FOLDER = "Milestone_2_FINAL"

if IN_COLAB:
    REPO_DIR = "/content/DAF2026"
    if not os.path.isdir(REPO_DIR):                       # clone so outputs can be saved (and pushed) inside the repo
        subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
else:
    cwd = Path.cwd()
    if cwd.name == TARGET_FOLDER:
        REPO_DIR = str(cwd.parent)
    else:
        REPO_DIR = str(cwd)
REPO_OUT = os.path.join(REPO_DIR, TARGET_FOLDER)
os.makedirs(REPO_OUT, exist_ok=True)

PRODUCTS = "[CLEANED_DATA]_MotorPH_Products_List_2025.csv"
SALES    = "[CLEANED_DATA]_MotorPH_Sales_Data-3rd_Quarter-Year_2025.csv"

def locate(fname):
    """Find a data file: Google Drive -> repo folder -> current folder -> GitHub raw URL."""
    for p in [DRIVE_DATA + fname, os.path.join(REPO_OUT, fname), fname]:
        if os.path.exists(p):
            return p
    return GITHUB_RAW + fname.replace("[", "%5B").replace("]", "%5D").replace(" ", "%20")

# Consistent colours for every chart
NAVY, LIGHT, ORANGE, GREY, GREEN, RED = "#1F4E79", "#8FB3D9", "#E07B39", "#6B7280", "#2E7D32", "#C62828"

# %% [markdown]
# # **1. LOAD THE DATA**

# %%
products = pd.read_csv(locate(PRODUCTS))
sales    = pd.read_csv(locate(SALES), parse_dates=["Date"])

print("Products:", products.shape, "| Sales:", sales.shape)
products.head()

# %% [markdown]
# # **2. DATA-QUALITY REVIEW (SALES DATASET)**
#
# The Sales file was already cleaned in Milestone 1. Before analysing it, it is re-checked here so the dashboard
# is built on data that is **verified**, not assumed. Each check reports `PASS` or `REVIEW` with its evidence
# (instead of a bare `assert` that stops the notebook without explanation).

# %%
def quality_review(products, sales):
    """Run the data-quality checks and return a table: Check | Result | Detail."""
    rows = []
    def add(check, ok, detail):
        rows.append({"Check": check, "Result": "PASS" if ok else "REVIEW", "Detail": detail})

    n = len(sales)
    miss = int(sales.isna().sum().sum())
    add("Missing values", miss == 0, f"{miss} blank cells across {sales.shape[1]} columns")

    dup_rows = int(sales.drop(columns="Transaction ID").duplicated().sum())
    add("Duplicate records", dup_rows == 0, f"{dup_rows} duplicate rows (ignoring Transaction ID)")
    add("Unique Transaction ID", sales["Transaction ID"].is_unique, f"{sales['Transaction ID'].nunique():,} IDs for {n:,} rows")

    types_ok = (pd.api.types.is_datetime64_any_dtype(sales["Date"])
                and pd.api.types.is_integer_dtype(sales["Quantity"])
                and pd.api.types.is_float_dtype(sales["Total Sales (PHP)"]))
    add("Data types", types_ok, "Date = datetime, Quantity = integer, prices/totals = float")

    d0, d1 = sales["Date"].min(), sales["Date"].max()
    in_window = d0 >= pd.Timestamp("2025-06-01") and d1 <= pd.Timestamp("2025-08-31")
    add("Dates inside Q3 window", in_window, f"{d0:%b %d, %Y} to {d1:%b %d, %Y}")

    nonpos = int((sales[["Unit Price (PHP)", "Quantity", "Total Sales (PHP)"]] <= 0).sum().sum())
    add("No zero / negative values", nonpos == 0, f"{nonpos} non-positive prices, quantities or totals")

    gap = (sales["Quantity"] * sales["Unit Price (PHP)"] - sales["Total Sales (PHP)"]).abs().max()
    add("Total = Unit Price x Quantity", gap < 0.01, f"largest difference PHP {gap:,.2f}")

    orphan = set(sales["Product ID Number"]) - set(products["Product ID Number"])
    add("Referential integrity (Sales -> Products)", not orphan, f"{len(orphan)} sales rows point to unknown product IDs")

    m = sales.merge(products[["Product ID Number", "Product Name", "Unit Price (PHP)"]],
                    on="Product ID Number", suffixes=("", "_cat"))
    bad_name  = int((m["Product Name"] != m["Product Name_cat"]).sum())
    bad_price = int((m["Unit Price (PHP)"] != m["Unit Price (PHP)_cat"]).sum())
    add("Product names match catalogue", bad_name == 0, f"{bad_name} mismatches")
    add("Unit prices match catalogue", bad_price == 0, f"{bad_price} mismatches")

    cats = products["Product Type"]
    add("Category labels clean", cats.notna().all() and (cats == cats.str.strip()).all(),
        f"{cats.nunique()} categories, no blanks or stray spaces")

    unk = int(((sales["Client Type"] == "Unknown") | (sales["Payment Method"] == "Unknown")).sum())
    unk_rev = sales.loc[(sales["Client Type"] == "Unknown") | (sales["Payment Method"] == "Unknown"), "Total Sales (PHP)"].sum()
    add("'Unknown' placeholders (kept)", True,
        f"{unk} rows ({unk / n:.1%}) = {unk_rev / sales['Total Sales (PHP)'].sum():.1%} of revenue; not charted")

    q1, q3 = sales["Quantity"].quantile([.25, .75])
    hi = int((sales["Quantity"] > q3 + 1.5 * (q3 - q1)).sum())
    add("Quantity outliers (IQR rule)", True, f"{hi} above IQR fence; largest order {sales['Quantity'].max()} units - kept")

    unsold = sorted(set(products["Product ID Number"]) - set(sales["Product ID Number"]))
    add("Products with no Q3 sales", not unsold, f"{len(unsold)} of {len(products)} products unsold")
    return pd.DataFrame(rows)

dq = quality_review(products, sales)
dq

# %%
# What was already fixed in Milestone 1 (raw -> cleaned). Documented here so the review is traceable end to end.
m1_fixes = pd.DataFrame([
    ("Dates stored as text in mixed formats",         "Parsed 7 formats into one datetime column"),
    ("Blank / impossible dates (e.g. 2025/07/32)",    "7 rows removed: 1,000 raw -> 993 clean rows"),
    ("Blank Client Type (10) and Payment (10)",       "Cannot be inferred -> labelled 'Unknown'"),
    ("13 corrupted product names (ends in 'x')",      "Matched back to the product list (15 rows)"),
    ("20 unit prices exactly 3x the catalogue price", "Restored from catalogue; totals were already right"),
    ("Unclear column names (unitprice, total ...)",   "Renamed for reporting; Product ID added"),
], columns=["Issue found in raw Sales data (Milestone 1)", "How it was resolved"])
m1_fixes

# %%
# Stop only if a CRITICAL check fails; informational items never block the dashboard.
critical = dq[dq["Check"].isin(["Unique Transaction ID", "Referential integrity (Sales -> Products)",
                                "Missing values", "Total = Unit Price x Quantity"])]
if (critical["Result"] != "PASS").any():
    raise ValueError("Critical data-quality check failed:\n" + critical[critical["Result"] != "PASS"].to_string())
print(f"Data-quality review: {(dq['Result'] == 'PASS').sum()} of {len(dq)} checks PASS - dashboard can be built.")

# %% [markdown]
# # **3. PREPARE THE DATA**
#
# - Add each product's **Category**, **Q3 units sold, revenue and transactions**.
# - Build the **category summary**, the **monthly summary**, the **daily series**, and a **category x month** table.

# %%
products["Product Type"] = products["Product Type"].str.strip()
products["Category"]     = products["Product Type"]

sales = sales.merge(products[["Product ID Number", "Category"]], on="Product ID Number", how="left")
sales["Month"] = sales["Date"].dt.to_period("M")

sales_by_product = (sales.groupby("Product ID Number")
                    .agg(**{"Q3 Units Sold": ("Quantity", "sum"),
                            "Q3 Revenue (PHP)": ("Total Sales (PHP)", "sum"),
                            "Q3 Transactions": ("Transaction ID", "count")})
                    .reset_index())
products_enriched = products.merge(sales_by_product, on="Product ID Number", how="left")
for c in ["Q3 Units Sold", "Q3 Revenue (PHP)", "Q3 Transactions"]:
    products_enriched[c] = products_enriched[c].fillna(0)
products_enriched[["Q3 Units Sold", "Q3 Transactions"]] = products_enriched[["Q3 Units Sold", "Q3 Transactions"]].astype(int)

# Category summary (data behind the category charts)
category_summary = (products_enriched.groupby("Category")
                    .agg(**{"Number of Products": ("Product ID Number", "count"),
                            "Q3 Units Sold": ("Q3 Units Sold", "sum"),
                            "Q3 Revenue (PHP)": ("Q3 Revenue (PHP)", "sum")})
                    .sort_values(["Number of Products", "Q3 Revenue (PHP)"], ascending=[False, False])
                    .reset_index())
category_summary["Share of Products (%)"]  = (category_summary["Number of Products"] / category_summary["Number of Products"].sum() * 100).round(1)
category_summary["Share of Revenue (%)"]   = (category_summary["Q3 Revenue (PHP)"] / category_summary["Q3 Revenue (PHP)"].sum() * 100).round(1)
category_summary["Revenue per Model (PHP)"] = (category_summary["Q3 Revenue (PHP)"] / category_summary["Number of Products"]).round(0)
category_summary["Revenue Rank"]       = category_summary["Q3 Revenue (PHP)"].rank(ascending=False).astype(int)
category_summary["Revenue/Model Rank"] = category_summary["Revenue per Model (PHP)"].rank(ascending=False).astype(int)

# Monthly summary
monthly = (sales.groupby("Month")
           .agg(**{"Units Sold": ("Quantity", "sum"), "Revenue (PHP)": ("Total Sales (PHP)", "sum"),
                   "Transactions": ("Transaction ID", "count")}))
monthly["Revenue Change vs Prior Month (%)"] = (monthly["Revenue (PHP)"].pct_change() * 100).round(1)
monthly.index = monthly.index.strftime("%b %Y")

# Daily series (a 7-day rolling mean avoids the partial-week problem of weekly bins: Jun 1 and Aug 31 are both Sundays)
daily = (sales.groupby("Date").agg(Revenue=("Total Sales (PHP)", "sum"), Units=("Quantity", "sum"))
         .asfreq("D", fill_value=0))
daily["Revenue 7-day avg"] = daily["Revenue"].rolling(7, min_periods=1).mean()

# Category x month revenue (temporal decomposition)
cat_month = sales.pivot_table(index="Category", columns="Month", values="Total Sales (PHP)", aggfunc="sum", fill_value=0)
cat_month.columns = cat_month.columns.strftime("%b")
cat_month = cat_month.loc[category_summary.sort_values("Q3 Revenue (PHP)", ascending=False)["Category"]]
cat_month_rank  = cat_month.rank(ascending=False).astype(int)
cat_month_share = cat_month.div(cat_month.sum(axis=1), axis=0) * 100      # % of the category's own Q3 revenue

monthly

# %% [markdown]
# # **4. KEY METRICS**

# %%
total_units = int(sales["Quantity"].sum())
total_rev   = float(sales["Total Sales (PHP)"].sum())
n_products  = int(products["Product ID Number"].nunique())
n_cats      = int(products["Category"].nunique())
n_txn       = int(len(sales))
avg_txn     = total_rev / n_txn
period      = f"{sales['Date'].min():%b %d} – {sales['Date'].max():%b %d, %Y}"
top_cat     = category_summary.sort_values("Q3 Revenue (PHP)", ascending=False).iloc[0]
best_model  = category_summary.sort_values("Revenue per Model (PHP)", ascending=False).iloc[0]

kpis = pd.DataFrame({
    "Metric": ["Total Units Sold", "Total Revenue (PHP)", "Number of Products", "Number of Categories",
               "Total Transactions", "Average Revenue per Transaction (PHP)", "Sales Period"],
    "Value":  [f"{total_units:,}", f"{total_rev:,.0f}", f"{n_products}", f"{n_cats}",
               f"{n_txn:,}", f"{avg_txn:,.0f}", period]})
kpis

# %%
# Does the category ranking change month to month, or is it stable? Is any category's revenue concentrated in one month?
even = 100 / cat_month.shape[1]
conc = cat_month_share.max(axis=1).round(1)
conc_flag = conc[conc >= 45].sort_values(ascending=False)         # an even split would be 33.3% per month

heritage_peak = cat_month_share.loc["Heritage"]
print("Heritage share of its own Q3 revenue by month (%):", heritage_peak.round(1).to_dict())
print("Heritage monthly rank:", cat_month_rank.loc["Heritage"].to_dict())
print(f"-> Heritage is {'CONCENTRATED in ' + heritage_peak.idxmax() if heritage_peak.max() >= 45 else 'spread evenly (no single-month spike)'}: "
      f"recommendation to promote Heritage {'needs a closer look' if heritage_peak.max() >= 45 else 'stands'}.")
print("\nCategories with >=45% of Q3 revenue in a single month (even split = 33%):")
print(conc_flag.to_string())

chg = (cat_month["Aug"] - cat_month["Jun"]) / cat_month["Jun"] * 100
print("\nBiggest Jun -> Aug movers (revenue %):")
print(pd.concat([chg.sort_values().head(3), chg.sort_values().tail(3)]).round(1).to_string())

# %% [markdown]
# # **5. CHART FUNCTIONS**
#
# Every chart is a function that can draw on its own figure (saved as a stand-alone PNG) **or** on a panel
# of the dashboard (`ax=`), so the dashboard and the stand-alone charts always match.

# %%
def _clean(ax, grid="x"):
    ax.grid(axis=grid, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

def plot_products_by_category(summary, ax=None, fs=10):
    """MILESTONE 2 DRAFT CHART - number of products per category, largest on top."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(11, 8))
    else:
        fig = ax.figure
    data  = summary.sort_values("Number of Products", ascending=True)
    total, top = data["Number of Products"].sum(), data["Number of Products"].max()
    colors = [NAVY if n == top else LIGHT for n in data["Number of Products"]]
    bars = ax.barh(data["Category"], data["Number of Products"], color=colors, edgecolor="white")
    for b, n in zip(bars, data["Number of Products"]):
        ax.text(b.get_width() + 0.15, b.get_y() + b.get_height() / 2, f"{n}  ({n / total:.0%})", va="center", fontsize=fs - 1)
    ax.set_title("Distribution of MotorPH Products by Category (2025 Product List)", fontsize=fs + 4 if own else fs + 2, fontweight="bold", pad=10)
    ax.set_xlabel("Number of Products", fontsize=fs + 1)
    ax.set_ylabel("Product Category", fontsize=fs + 1)
    ax.set_xlim(0, top + 2.6); ax.set_xticks(range(0, top + 2, 2))
    ax.tick_params(axis="y", labelsize=fs); _clean(ax)
    if own:
        fig.text(0.99, 0.01, f"Total: {total} products across {len(data)} categories", ha="right", va="bottom", fontsize=9, style="italic", color="dimgray")
        fig.tight_layout(rect=(0, 0.02, 1, 1))
    else:
        ax.text(0.99, 0.02, f"Total: {total} products, {len(data)} categories", transform=ax.transAxes, ha="right", fontsize=fs - 1, style="italic", color="dimgray")
    return fig

def plot_revenue_by_category(summary, ax=None, fs=10):
    """Q3 revenue per category (PHP millions), annotated with the number of models."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(11, 8))
    else:
        fig = ax.figure
    data = summary.sort_values("Q3 Revenue (PHP)", ascending=True)
    top_c  = data["Category"].iloc[-1]
    best_c = summary.sort_values("Revenue per Model (PHP)", ascending=False)["Category"].iloc[0]
    colors = [NAVY if c == top_c else ORANGE if c == best_c else LIGHT for c in data["Category"]]
    mil = data["Q3 Revenue (PHP)"] / 1e6
    bars = ax.barh(data["Category"], mil, color=colors, edgecolor="white")
    for b, m, sh, n in zip(bars, mil, data["Share of Revenue (%)"], data["Number of Products"]):
        ax.text(b.get_width() + mil.max() * 0.01, b.get_y() + b.get_height() / 2,
                f"PHP {m:,.1f}M ({sh:.1f}%) | {n} model{'s' if n > 1 else ''}", va="center", fontsize=fs - 1)
    ax.set_title("Q3 2025 Revenue by Product Category", fontsize=fs + 4 if own else fs + 2, fontweight="bold", pad=10)
    ax.set_xlabel("Q3 Revenue (PHP millions)", fontsize=fs + 1)
    ax.set_ylabel("Product Category", fontsize=fs + 1)
    ax.set_xlim(0, mil.max() * 1.45); ax.tick_params(axis="y", labelsize=fs); _clean(ax)
    note = f"Dark blue = highest revenue ({top_c}); orange = highest revenue per model ({best_c})"
    if own:
        fig.text(0.99, 0.01, note, ha="right", va="bottom", fontsize=9, style="italic", color="dimgray"); fig.tight_layout(rect=(0, 0.02, 1, 1))
    else:
        ax.text(0.99, 0.02, note, transform=ax.transAxes, ha="right", fontsize=fs - 2, style="italic", color="dimgray")
    return fig

def plot_share_comparison(summary, ax=None, fs=10):
    """Share of products vs share of revenue, per category."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(11, 8))
    else:
        fig = ax.figure
    data = summary.sort_values("Share of Revenue (%)", ascending=True)
    y, h = np.arange(len(data)), 0.4
    ax.barh(y + h / 2, data["Share of Products (%)"], h, label="Share of products (%)", color=LIGHT)
    ax.barh(y - h / 2, data["Share of Revenue (%)"],  h, label="Share of Q3 revenue (%)", color=NAVY)
    ax.set_yticks(y); ax.set_yticklabels(data["Category"], fontsize=fs)
    ax.set_title("Share of Products vs Share of Revenue by Category", fontsize=fs + 4 if own else fs + 2, fontweight="bold", pad=10)
    ax.set_xlabel("Share of total (%)", fontsize=fs + 1); ax.set_ylabel("Product Category", fontsize=fs + 1)
    ax.legend(loc="lower right", frameon=False, fontsize=fs); _clean(ax)
    if own:
        fig.tight_layout()
    return fig

def plot_units_by_category(summary, ax=None, fs=10):
    """Units sold per category (volume comparison)."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(11, 8))
    else:
        fig = ax.figure
    data = summary.sort_values("Q3 Units Sold", ascending=True)
    top_c = data["Category"].iloc[-1]
    bars = ax.barh(data["Category"], data["Q3 Units Sold"], color=[NAVY if c == top_c else LIGHT for c in data["Category"]], edgecolor="white")
    tot = data["Q3 Units Sold"].sum()
    for b, u in zip(bars, data["Q3 Units Sold"]):
        ax.text(b.get_width() + data["Q3 Units Sold"].max() * 0.01, b.get_y() + b.get_height() / 2, f"{u:,} ({u / tot:.1%})", va="center", fontsize=fs - 1)
    ax.set_title("Q3 2025 Units Sold by Product Category", fontsize=fs + 4 if own else fs + 2, fontweight="bold", pad=10)
    ax.set_xlabel("Units sold", fontsize=fs + 1); ax.set_ylabel("Product Category", fontsize=fs + 1)
    ax.set_xlim(0, data["Q3 Units Sold"].max() * 1.22); ax.tick_params(axis="y", labelsize=fs); _clean(ax)
    if own:
        fig.tight_layout()
    return fig

def plot_daily_trend(daily, ax=None, fs=10):
    """Daily revenue with a 7-day rolling average, plus month boundaries."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(11, 5))
    else:
        fig = ax.figure
    ax.bar(daily.index, daily["Revenue"] / 1e6, color=LIGHT, width=0.9, label="Daily revenue")
    ax.plot(daily.index, daily["Revenue 7-day avg"] / 1e6, color=NAVY, lw=2.4, label="7-day average")
    for d in pd.to_datetime(["2025-07-01", "2025-08-01"]):
        ax.axvline(d, color=GREY, ls=":", lw=1)
    ax.set_title("Daily Revenue, June–August 2025", fontsize=fs + 3, fontweight="bold", pad=8)
    ax.set_ylabel("Revenue (PHP millions)", fontsize=fs); ax.set_xlabel("Date", fontsize=fs)
    ax.legend(frameon=False, fontsize=fs - 1, loc="upper right", ncol=2)
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 15]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(labelsize=fs - 1); _clean(ax, "y")
    ax.margins(x=0.01)
    if own:
        fig.tight_layout()
    return fig

def plot_monthly_summary(monthly, ax=None, fs=10):
    """Two stacked panels: monthly revenue (top) and monthly units sold (bottom)."""
    if ax is None:
        fig = plt.figure(figsize=(7, 6.5))
        gs = fig.add_gridspec(2, 1, hspace=0.5)
    else:
        fig, sps = ax.figure, ax.get_subplotspec()
        ax.remove()
        gs = sps.subgridspec(2, 1, hspace=0.55)
    a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    x = np.arange(len(monthly))
    rev, units = monthly["Revenue (PHP)"] / 1e6, monthly["Units Sold"]
    bars = a1.bar(x, rev, color=NAVY, width=0.55)
    for bar, r, ch in zip(bars, rev, monthly["Revenue Change vs Prior Month (%)"]):
        a1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + rev.max() * 0.02,
                f"PHP {r:,.0f}M" + ("" if pd.isna(ch) else f"\n({ch:+.1f}% MoM)"), ha="center", va="bottom", fontsize=fs - 1)
    a1.set_ylim(0, rev.max() * 1.35); a1.set_ylabel("Revenue (PHP M)", fontsize=fs)
    a1.set_title("Monthly Revenue and Units Sold", fontsize=fs + 3, fontweight="bold", pad=8)
    bars = a2.bar(x, units, color=ORANGE, width=0.55)
    for bar, u, ch in zip(bars, units, units.pct_change() * 100):
        a2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + units.max() * 0.02,
                f"{u:,} units" + ("" if pd.isna(ch) else f"\n({ch:+.1f}% MoM)"), ha="center", va="bottom", fontsize=fs - 1)
    a2.set_ylim(0, units.max() * 1.35); a2.set_ylabel("Units sold", fontsize=fs)
    for a in (a1, a2):
        a.set_xticks(x); a.set_xticklabels(monthly.index, fontsize=fs); a.tick_params(axis="y", labelsize=fs - 1); _clean(a, "y")
    return fig

def plot_category_monthly_lines(cat_month, summary, ax=None, fs=10, n=6):
    """Monthly revenue of the top-n categories (shows which categories are rising or falling)."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure
    top = cat_month.head(n)
    palette = [NAVY, ORANGE, "#2E7D32", "#7B1FA2", "#C62828", "#00838F"]
    # spread the end labels so they never overlap
    ends = sorted([(row.iloc[-1] / 1e6, cat) for cat, row in top.iterrows()])
    gap, ypos = (top.max().max() / 1e6) * 0.06, []
    for v, cat in ends:
        ypos.append(max(v, ypos[-1] + gap) if ypos else v)
    label_y = {cat: y for (v, cat), y in zip(ends, ypos)}
    for (cat, row), col in zip(top.iterrows(), palette):
        ax.plot(row.index, row / 1e6, marker="o", lw=2.2, color=col, label=cat)
        ax.text(len(row) - 1 + 0.07, label_y[cat], f"{cat} ({(row.iloc[-1] / row.iloc[0] - 1) * 100:+.0f}%)", color=col, fontsize=fs - 1, va="center")
    ax.set_xlim(-0.1, len(top.columns) - 1 + 0.75)
    ax.set_title(f"Monthly Revenue of the Top {n} Categories (label = Jun→Aug change)", fontsize=fs + 3, fontweight="bold", pad=8)
    ax.set_ylabel("Revenue (PHP millions)", fontsize=fs); ax.set_xlabel("Month (2025)", fontsize=fs)
    ax.tick_params(labelsize=fs); _clean(ax, "y")
    if own:
        fig.tight_layout()
    return fig

def plot_category_month_heatmap(cat_month, rank, share, ax=None, fs=8):
    """Heatmap: revenue (PHP M) and monthly rank per category; shading = share of the category's own Q3 revenue."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 8))
    else:
        fig = ax.figure
    ax.imshow(share.values, cmap="Blues", vmin=0, vmax=70, aspect="auto")
    for i in range(cat_month.shape[0]):
        for j in range(cat_month.shape[1]):
            v = share.values[i, j]
            ax.text(j, i, f"{cat_month.values[i, j] / 1e6:,.0f}  (#{rank.values[i, j]})", ha="center", va="center",
                    fontsize=fs, color="white" if v > 42 else "black")
    ax.set_xticks(range(cat_month.shape[1])); ax.set_xticklabels(cat_month.columns, fontsize=fs + 1)
    ax.set_yticks(range(cat_month.shape[0])); ax.set_yticklabels(cat_month.index, fontsize=fs + 1)
    ax.xaxis.tick_top()
    ax.set_title("Category Revenue by Month: PHP M (rank)", fontsize=fs + 4, fontweight="bold", pad=22)
    ax.text(0.5, -0.025, "Shading = share of the category's own Q3 revenue (darker = concentrated in that month)",
            transform=ax.transAxes, ha="center", va="top", fontsize=fs - 1, style="italic", color="dimgray")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    if own:
        fig.tight_layout()
    return fig

# %% [markdown]
# # **6. STAND-ALONE CHARTS**
# (Section 3 of the dashboard reuses the Milestone 2 draft chart exactly as submitted.)

# %%
fig_cat   = plot_products_by_category(category_summary);  plt.show()

# %%
fig_rev   = plot_revenue_by_category(category_summary);   plt.show()

# %%
fig_share = plot_share_comparison(category_summary);      plt.show()

# %%
fig_units = plot_units_by_category(category_summary);     plt.show()

# %%
fig_daily = plot_daily_trend(daily);                      plt.show()

# %%
fig_month = plot_monthly_summary(monthly);                plt.show()

# %%
fig_lines = plot_category_monthly_lines(cat_month, category_summary);  plt.show()

# %%
fig_heat  = plot_category_month_heatmap(cat_month, cat_month_rank, cat_month_share);  plt.show()

# %% [markdown]
# # **7. COMPILE THE DASHBOARD**
#
# | Section | Content |
# |---|---|
# | **1. MotorPH Overview** | Key metrics, key takeaways, data-quality snapshot |
# | **2. Sales Performance** | Daily trend, monthly revenue and units, category trends, category x month heatmap |
# | **3. Sales by Product Category** | Milestone 2 draft chart, revenue, units, share of products vs revenue |
# | **4. Inventory** | Structured list of all products with category and price (plus Q3 units sold) |

# %%
PAGE_TOTAL = 4
def frame(fig, number, title, subtitle):
    """Common header band and footer so all pages look consistent."""
    fig.patch.set_facecolor("white")
    band = fig.add_axes([0, 0.93, 1, 0.07]); band.axis("off"); band.set_facecolor(NAVY)
    band.add_patch(plt.Rectangle((0, 0), 1, 1, color=NAVY, transform=band.transAxes))
    band.text(0.02, 0.62, f"MotorPH Analytical Dashboard  |  Q3 2025 ({period})", color="white", fontsize=17, fontweight="bold", va="center")
    band.text(0.02, 0.18, f"Section {number}: {title}  -  {subtitle}", color="#D6E4F0", fontsize=11, va="center")
    fig.text(0.02, 0.008, "Source: MotorPH cleaned Products List 2025 and Q3 2025 Sales Data (Milestone 1)   |   MO-IT106 Milestone 2   |   Revenue is not profit",
             fontsize=8, color=GREY, style="italic")
    fig.text(0.98, 0.008, f"Page {number} of {PAGE_TOTAL}", fontsize=8, color=GREY, ha="right")

def draw_table(ax, df, col_widths, fs=8, header_color=NAVY, result_col=None, scale_y=1.5):
    ax.axis("off")
    t = ax.table(cellText=df.values, colLabels=df.columns, colWidths=col_widths, loc="upper center", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(fs); t.scale(1, scale_y)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#D9D9D9")
        if r == 0:
            cell.set_facecolor(header_color); cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F3F7FB" if r % 2 == 0 else "white")
            if result_col is not None and c == result_col:
                ok = cell.get_text().get_text() == "PASS"
                cell.set_text_props(color=GREEN if ok else RED, fontweight="bold")
    return t

# ---------- Page 1: MotorPH Overview ----------
def page_overview():
    fig = plt.figure(figsize=(16, 9))
    frame(fig, 1, "MotorPH Overview", "high-level metrics for products and sales")
    cards = [("Total Units Sold", f"{total_units:,}", "units, Jun–Aug 2025"),
             ("Total Revenue", f"PHP {total_rev / 1e9:,.2f}B", f"PHP {total_rev:,.0f}"),
             ("Number of Products", f"{n_products}", "models in the 2025 catalogue"),
             ("Product Categories", f"{n_cats}", f"largest: {category_summary.iloc[0]['Category']} ({int(category_summary.iloc[0]['Number of Products'])} models)"),
             ("Transactions", f"{n_txn:,}", "valid sales records"),
             ("Avg Revenue / Transaction", f"PHP {avg_txn / 1e6:,.2f}M", f"PHP {avg_txn:,.0f}")]
    w, gap, x0 = 0.147, 0.0155, 0.02
    for i, (lab, val, sub) in enumerate(cards):
        ax = fig.add_axes([x0 + i * (w + gap), 0.69, w, 0.19]); ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#F3F7FB", edgecolor=LIGHT, lw=1.5))
        ax.add_patch(plt.Rectangle((0, 0.93), 1, 0.07, transform=ax.transAxes, facecolor=NAVY if i % 2 == 0 else ORANGE))
        ax.text(0.5, 0.74, lab, ha="center", fontsize=10.5, color=GREY, fontweight="bold")
        ax.text(0.5, 0.40, val, ha="center", fontsize=21, color=NAVY, fontweight="bold")
        ax.text(0.5, 0.13, sub, ha="center", fontsize=8.5, color=GREY)

    # Key takeaways (computed, not typed)
    ax = fig.add_axes([0.02, 0.07, 0.42, 0.55]); ax.axis("off")
    ax.text(0, 1.0, "Key takeaways", fontsize=14, fontweight="bold", color=NAVY, va="top")
    jun, aug = monthly["Revenue (PHP)"].iloc[0], monthly["Revenue (PHP)"].iloc[-1]
    cru = cat_month.loc["Cruiser"]; dsp = cat_month.loc["Dual-Sport"]
    lines = [
        f"1. Revenue is easing: {monthly.index[0]} PHP {jun / 1e6:,.0f}M to {monthly.index[-1]} PHP {aug / 1e6:,.0f}M ({(aug / jun - 1) * 100:+.1f}%); units fell {(monthly['Units Sold'].iloc[-1] / monthly['Units Sold'].iloc[0] - 1) * 100:+.1f}%.",
        f"2. {top_cat['Category']} leads revenue (PHP {top_cat['Q3 Revenue (PHP)'] / 1e6:,.0f}M, {top_cat['Share of Revenue (%)']}%) with {int(top_cat['Number of Products'])} models, but earns only PHP {top_cat['Revenue per Model (PHP)'] / 1e6:,.1f}M per model.",
        f"3. {best_model['Category']} is the most efficient: PHP {best_model['Revenue per Model (PHP)'] / 1e6:,.1f}M per model from only {int(best_model['Number of Products'])} models.",
        f"4. Heritage is steady (rank {cat_month_rank.loc['Heritage'].min()}–{cat_month_rank.loc['Heritage'].max()} every month), so its strength is not a one-month spike.",
        f"5. Cruiser is slipping (PHP {cru['Jun'] / 1e6:,.0f}M to {cru['Aug'] / 1e6:,.0f}M, {(cru['Aug'] / cru['Jun'] - 1) * 100:+.0f}%) while Dual-Sport is rising ({(dsp['Aug'] / dsp['Jun'] - 1) * 100:+.0f}%).",
        f"6. Top 3 categories hold {category_summary.sort_values('Q3 Revenue (PHP)', ascending=False).head(3)['Share of Revenue (%)'].sum():.1f}% of revenue; "
        f"{int((category_summary['Number of Products'] == 1).sum())} categories have a single model.",
    ]
    import textwrap
    y = 0.90
    for ln in lines:
        wrapped = textwrap.fill(ln, 74)
        ax.text(0, y, wrapped, fontsize=11, va="top", color="#222222", linespacing=1.35)
        y -= 0.062 * (wrapped.count("\n") + 1) + 0.04

    # Data-quality snapshot
    ax = fig.add_axes([0.47, 0.07, 0.51, 0.55]); ax.axis("off")
    ax.text(0, 1.0, f"Data-quality review of the Sales data ({(dq['Result'] == 'PASS').sum()} of {len(dq)} checks pass)",
            fontsize=14, fontweight="bold", color=NAVY, va="top", transform=ax.transAxes)
    snap = dq.copy()
    snap["Detail"] = snap["Detail"].str.slice(0, 62)
    sub = fig.add_axes([0.47, 0.07, 0.51, 0.50])
    draw_table(sub, snap[["Check", "Result", "Detail"]], [0.34, 0.09, 0.57], fs=7.6, result_col=1, scale_y=1.42)
    return fig

# ---------- Page 2: Sales Performance ----------
def page_sales_performance():
    fig = plt.figure(figsize=(16, 10))
    frame(fig, 2, "Sales Performance", "trends over time and comparisons")
    gs = fig.add_gridspec(2, 3, left=0.05, right=0.985, top=0.89, bottom=0.085, hspace=0.36, wspace=0.34, height_ratios=[1, 1.15])
    plot_daily_trend(daily, ax=fig.add_subplot(gs[0, 0:2]), fs=10)
    plot_monthly_summary(monthly, ax=fig.add_subplot(gs[0, 2]), fs=10)
    plot_category_monthly_lines(cat_month, category_summary, ax=fig.add_subplot(gs[1, 0:2]), fs=10)
    plot_category_month_heatmap(cat_month, cat_month_rank, cat_month_share, ax=fig.add_subplot(gs[1, 2]), fs=7.5)
    return fig

# ---------- Page 3: Sales by Product Category ----------
def page_category():
    fig = plt.figure(figsize=(16, 10))
    frame(fig, 3, "Sales by Product Category", "includes the Milestone 2 draft chart")
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.985, top=0.89, bottom=0.085, hspace=0.30, wspace=0.34)
    plot_products_by_category(category_summary, ax=fig.add_subplot(gs[0, 0]), fs=8.5)
    plot_revenue_by_category(category_summary, ax=fig.add_subplot(gs[0, 1]), fs=8.5)
    plot_units_by_category(category_summary, ax=fig.add_subplot(gs[1, 0]), fs=8.5)
    plot_share_comparison(category_summary, ax=fig.add_subplot(gs[1, 1]), fs=8.5)
    return fig

# ---------- Page 4: Inventory ----------
inventory = (products_enriched
             .merge(category_summary[["Category", "Revenue Rank"]], on="Category")
             .sort_values(["Revenue Rank", "Unit Price (PHP)"], ascending=[True, False])
             [["Product ID Number", "Product Name", "Category", "Unit Price (PHP)", "Q3 Units Sold"]]
             .rename(columns={"Product ID Number": "ID"}))

def page_inventory():
    fig = plt.figure(figsize=(16, 10))
    frame(fig, 4, "Inventory", f"all {len(inventory)} products by category (ordered by category revenue, then price)")
    view = inventory.copy()
    view["Unit Price (PHP)"] = view["Unit Price (PHP)"].map("{:,.0f}".format)
    view["Q3 Units Sold"]    = view["Q3 Units Sold"].map("{:,}".format)
    half = int(np.ceil(len(view) / 2))
    for k, part in enumerate([view.iloc[:half], view.iloc[half:]]):
        ax = fig.add_axes([0.02 + k * 0.495, 0.04, 0.475, 0.86])
        draw_table(ax, part, [0.07, 0.38, 0.21, 0.19, 0.15], fs=8.5, scale_y=1.78)
    return fig

pages = [page_overview(), page_sales_performance(), page_category(), page_inventory()]
plt.show()

# %% [markdown]
# # **8. SAVE THE OUTPUTS (REPOSITORY + GOOGLE DRIVE)**

# %%
OUTPUT_FILES = {                                   # every file written by this notebook (also used for the summary printout)
    "[DASHBOARD]_MotorPH_Dashboard_2025.pdf": "PDF",
    "[DASHBOARD]_Page1_MotorPH_Overview.png": pages[0],
    "[DASHBOARD]_Page2_Sales_Performance.png": pages[1],
    "[DASHBOARD]_Page3_Sales_by_Product_Category.png": pages[2],
    "[DASHBOARD]_Page4_Inventory.png": pages[3],
    "[VISUALIZATION]_MotorPH_Products_by_Category_2025.png": fig_cat,
    "[VISUALIZATION]_MotorPH_Revenue_by_Category_2025.png": fig_rev,
    "[VISUALIZATION]_MotorPH_Product_vs_Revenue_Share_2025.png": fig_share,
    "[VISUALIZATION]_MotorPH_Units_by_Category_2025.png": fig_units,
    "[VISUALIZATION]_MotorPH_Daily_Revenue_Trend_2025.png": fig_daily,
    "[VISUALIZATION]_MotorPH_Monthly_Revenue_Units_2025.png": fig_month,
    "[VISUALIZATION]_MotorPH_Category_Monthly_Trend_2025.png": fig_lines,
    "[VISUALIZATION]_MotorPH_Category_Month_Heatmap_2025.png": fig_heat,
}
TABLE_FILES = {
    "[EDITED_DATA]_MotorPH_Products_List_2025.csv": products_enriched,
    "[SUMMARY]_MotorPH_Products_by_Category_2025.csv": category_summary,
    "[SUMMARY]_MotorPH_Monthly_Sales_2025.csv": monthly.reset_index(),
    "[SUMMARY]_MotorPH_Category_by_Month_Revenue_2025.csv": cat_month.reset_index(),
    "[SUMMARY]_MotorPH_Key_Metrics_2025.csv": kpis,
    "[DATA_QUALITY]_MotorPH_Sales_Review_2025.csv": dq,
    "[DATA_QUALITY]_MotorPH_Milestone1_Fixes_2025.csv": m1_fixes,
    "[INVENTORY]_MotorPH_Product_List_2025.csv": inventory,
}

targets = [REPO_OUT]
if os.path.isdir(os.path.dirname(DRIVE_OUT.rstrip("/"))):          # Google Drive mounted / available
    os.makedirs(DRIVE_OUT, exist_ok=True)
    targets.append(DRIVE_OUT)
else:
    print("Google Drive path not found (not running in Colab) - saving to the repository folder only.")

for folder in targets:
    for name, obj in OUTPUT_FILES.items():
        path = os.path.join(folder, name)
        if obj == "PDF":
            with PdfPages(path) as pdf:
                for pg in pages:
                    pdf.savefig(pg)
        else:
            if name.startswith("[DASHBOARD]"):
                obj.savefig(path, dpi=200)                      # fixed page size keeps the header band full width
            else:
                obj.savefig(path, dpi=300, bbox_inches="tight")
    for name, df in TABLE_FILES.items():
        df.to_csv(os.path.join(folder, name), index=False)
    print(f"Saved {len(OUTPUT_FILES) + len(TABLE_FILES)} files to: {folder}")

# %% [markdown]
# # **9. OPTIONAL: PUSH TO GITHUB (Colab only)**
# Needs Colab secrets `GITHUB_TOKEN`, `UserName`, `UserEmail` (key icon in the sidebar, with "Notebook access" on).
# Set `PUSH_TO_GITHUB = False` to skip. Outside Colab this cell does nothing, so the notebook runs anywhere.

# %%
PUSH_TO_GITHUB = True

if PUSH_TO_GITHUB and IN_COLAB:
    from google.colab import userdata          # imported here (not at the top) so the notebook also runs outside Colab
    try:
        token, user_name, user_email = (userdata.get(k) for k in ("GITHUB_TOKEN", "UserName", "UserEmail"))
    except Exception as e:
        raise RuntimeError("Could not read GITHUB_TOKEN / UserName / UserEmail. Add them under the key icon "
                           "in the left sidebar and enable 'Notebook access'.") from e

    repo = Path(REPO_DIR)
    if not (repo / ".git").exists():
        raise RuntimeError(f"{repo} is not a git repository - clone the repo first or point REPO_DIR to the clone.")
    if not (repo / TARGET_FOLDER).is_dir():
        raise FileNotFoundError(f"'{TARGET_FOLDER}' not found in {repo}. Contents: {sorted(p.name for p in repo.iterdir())}")

    remote = REPO_URL.replace("https://", f"https://x-access-token:{token}@", 1)

    def run(*args, check=True):
        """Run a git command in the repo; the token is masked in any output."""
        result = subprocess.run(args, cwd=repo, capture_output=True, text=True)
        output = (result.stdout + result.stderr).replace(token, "***").strip()
        if output:
            print(output)
        if check and result.returncode != 0:
            raise RuntimeError(f"'git {args[1]}' failed with exit code {result.returncode}")
        return result

    run("git", "config", "user.name", user_name)
    run("git", "config", "user.email", user_email)
    run("git", "add", TARGET_FOLDER)
    if run("git", "diff", "--cached", "--quiet", check=False).returncode == 0:      # 0 = nothing staged
        print("Nothing new to commit.")
    else:
        run("git", "commit", "-m", "Milestone 2 FINAL: analytical dashboard, sales trends, key metrics, data-quality review")
    run("git", "pull", "--rebase", remote, "main")
    run("git", "push", remote, "HEAD:main")
    print("Pushed to GitHub.")
else:
    print("GitHub push skipped (not in Colab, or PUSH_TO_GITHUB = False).")
