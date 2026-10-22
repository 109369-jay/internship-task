
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# 1. Load dataset
df = pd.read_excel("Online Retail.xlsx")

# 2. Clean data
df.columns = df.columns.str.strip()
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)

df = df.dropna(
    subset=["InvoiceDate", "Quantity", "UnitPrice"]
)
df = df.drop_duplicates()

# 3. Remove cancellations and invalid records
is_cancellation = (
    df["InvoiceNo"].astype(str).str.startswith("C")
)

sales = df[
    (~is_cancellation)
    & (df["Quantity"] > 0)
    & (df["UnitPrice"] >= 0)
].copy()

# 4. Calculate daily order counts
daily_orders = (
    sales.set_index("InvoiceDate")
    .resample("D")["InvoiceNo"]
    .nunique()
)

# 5. Plot daily orders
daily_orders.plot(
    figsize=(10, 4),
    title="Daily Order Count"
)

plt.xlabel("Date")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.savefig("daily_orders.png")
plt.show()

# 6. Simple forecast baseline
results = daily_orders.to_frame("actual")
results["forecast"] = results["actual"].shift(7)

valid = results.dropna()

mae = mean_absolute_error(
    valid["actual"],
    valid["forecast"]
) if not valid.empty else float("nan")

print("Total cleaned records:", len(sales))
print("Average daily orders:", round(daily_orders.mean(), 2))
print("Baseline forecast MAE:", round(mae, 2))
print("Chart saved as daily_orders.png")
