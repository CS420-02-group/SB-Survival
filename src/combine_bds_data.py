import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))  # path to src/
data_dir = os.path.join(script_dir, "..", "data")

output_dir = os.path.join("..", "processed")
os.makedirs(output_dir, exist_ok=True)

# Load and clean bd.data.1.AllItems.txt
data = pd.read_csv(os.path.join(data_dir, "bd.data.1.AllItems.txt"), sep=r"\s+", engine="python")
data.columns = data.columns.str.strip().str.lower()
data["series_id"] = data["series_id"].astype(str).str.strip()

# Load and clean bd.series.txt
series = pd.read_csv(os.path.join(data_dir, "bd.series.txt"), sep="\t", engine="python", encoding="utf-8-sig")
series.columns = series.columns.str.strip().str.lower()
for col in ["series_id", "dataclass_code", "dataelement_code", "industry_code", "sizeclass_code"]:
    series[col] = series[col].astype(str).str.strip()

# Load and clean mapping files
dataclass = pd.read_csv(os.path.join(data_dir, "bd.dataclass.txt"), sep="\t", engine="python")
dataclass.columns = dataclass.columns.str.strip().str.lower()
dataclass["dataclass_code"] = dataclass["dataclass_code"].astype(str).str.strip()

dataelement = pd.read_csv(os.path.join(data_dir, "bd.dataelement.txt"), sep="\t", engine="python")
dataelement.columns = dataelement.columns.str.strip().str.lower()
dataelement["dataelement_code"] = dataelement["dataelement_code"].astype(str).str.strip()

industry = pd.read_csv(os.path.join(data_dir, "bd.industry.txt"), sep="\t", engine="python")
industry.columns = industry.columns.str.strip().str.lower()
industry["industry_code"] = industry["industry_code"].astype(str).str.strip()

sizeclass = pd.read_csv(os.path.join(data_dir, "bd.sizeclass.txt"), sep="\t", engine="python")
sizeclass.columns = sizeclass.columns.str.strip().str.lower()
sizeclass["sizeclass_code"] = sizeclass["sizeclass_code"].astype(str).str.strip()

# Merge all data
df = data.merge(series, on="series_id", how="left")
df = df.merge(dataclass, on="dataclass_code", how="left")
df = df.merge(dataelement, on="dataelement_code", how="left")
df = df.merge(industry, on="industry_code", how="left")
df = df.merge(sizeclass, on="sizeclass_code", how="left")

# Filter: Only include data from 2005 and onward
df = df[df["year"] >= 2005]

# Show sample before filtering
print("Unique dataclass_name values:", df["dataclass_name"].dropna().unique())
print("Sample rows before filtering by dataclass_name:")
print(df[["year", "dataclass_name", "dataelement_name", "industry_name", "sizeclass_name"]].head(10))

# Filter to include only establishment births and deaths
df = df[df["dataclass_name"].isin(["Establishment Births", "Establishment Deaths"])]

# Save to processed CSV
output_file = os.path.join(output_dir, "filtered_bds_data.csv")
df.to_csv(output_file, index=False)
print(f"✅ Processed data saved to: {output_file}")
