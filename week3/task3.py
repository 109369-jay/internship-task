import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned logistics dataset
df = pd.read_csv("logistics_advanced_dataset.csv")

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

# 1. Boxplot: Transit Time across Transport Modes
fig, ax = plt.subplots(figsize=(7, 4.2))
sns.boxplot(data=df, x='transport_mode', y='transit_time_hrs', palette='Set2', ax=ax, showmeans=True,
            meanprops={"marker":"o","markerfacecolor":"red", "markeredgecolor":"red", "markersize":"5"})
ax.set_title("Figure 1: Transit Time Distribution Across Transport Modes", fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel("Transport Mode", fontsize=10, fontweight='bold')
ax.set_ylabel("Transit Time (Hours)", fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig("fig1_transit_time_distribution.png", dpi=200)
plt.close()

# 2. Scatter Plot: Cargo Weight vs Transport Cost by Mode
fig, ax = plt.subplots(figsize=(7, 4.2))
sns.scatterplot(data=df, x='weight_kg', y='transport_cost_usd', hue='transport_mode', style='transport_mode',
                palette='tab10', alpha=0.75, s=45, ax=ax)
ax.set_title("Figure 2: Impact of Cargo Weight on Transport Cost by Modal Choice", fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel("Cargo Weight (kg)", fontsize=10, fontweight='bold')
ax.set_ylabel("Transport Cost (USD)", fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig("fig2_cost_vs_weight.png", dpi=200)
plt.close()

# 3. Correlation Heatmap
fig, ax = plt.subplots(figsize=(6.5, 5))
num_cols = ['shipment_volume_cbm', 'weight_kg', 'distance_km', 'handling_time_hrs', 'transit_time_hrs', 'transport_cost_usd', 'delay_hrs']
corr_matrix = df[num_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
ax.set_title("Figure 3: Pearson Correlation Matrix of Operational Variables", fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig("fig3_correlation_heatmap.png", dpi=200)
plt.close()

# 4. Carrier OTIF % Bar Chart
carrier_otif = df.groupby('carrier')['delivery_status'].apply(lambda x: (x == 'On-Time').mean() * 100).reset_index()
carrier_otif.columns = ['carrier', 'on_time_pct']
carrier_otif = carrier_otif.sort_values(by='on_time_pct', ascending=False)
fig, ax = plt.subplots(figsize=(7, 4))
bars = sns.barplot(data=carrier_otif, x='carrier', y='on_time_pct', palette='viridis', ax=ax)
ax.set_title("Figure 4: On-Time Fulfillment Rate (OTIF %) by Carrier", fontsize=12, fontweight='bold', pad=12)
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig("fig4_carrier_otif.png", dpi=200)
plt.close()

# 5. Bottleneck Analysis Bar Chart
route_perf = df.groupby('route')[['handling_time_hrs', 'transit_time_hrs']].mean().reset_index()
fig, ax = plt.subplots(figsize=(7.5, 4.2))
x = np.arange(len(route_perf))
width = 0.35
ax.bar(x - width/2, route_perf['handling_time_hrs'], width, label='Terminal Handling Time (hrs)', color='#f59e0b')
ax.bar(x + width/2, route_perf['transit_time_hrs'], width, label='Total Transit Time (hrs)', color='#2563eb')
ax.set_xticks(x)
ax.set_xticklabels(route_perf['route'], rotation=15, ha='right')
ax.set_title("Figure 5: Operational Bottlenecks - Handling vs Total Transit Time by Route", fontsize=12, fontweight='bold', pad=12)
ax.legend()
plt.tight_layout()
plt.savefig("fig5_route_bottleneck.png", dpi=200)
plt.close()

