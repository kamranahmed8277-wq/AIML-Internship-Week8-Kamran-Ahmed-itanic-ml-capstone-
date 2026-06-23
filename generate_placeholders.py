import matplotlib.pyplot as plt
import numpy as np
import os

# Create the reports folder if it doesn't exist
os.makedirs("reports", exist_ok=True)

print(" Generating placeholder images for Streamlit dashboard...")

# -------- 1. EDA Overview (for Page 3) --------
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Plot 1: Age distribution
axes[0, 0].hist(np.random.randn(1000), bins=30, color='skyblue', edgecolor='black')
axes[0, 0].set_title("Age Distribution")

# Plot 2: Pclass counts
axes[0, 1].bar(["1st", "2nd", "3rd"], [216, 184, 491], color=['gold', 'silver', '#cd7f32'])
axes[0, 1].set_title("Passenger Class")

# Plot 3: Fare vs Age scatter
axes[1, 0].scatter(np.random.randn(200)*10 + 30, np.random.randn(200)*50 + 50, alpha=0.5)
axes[1, 0].set_title("Fare vs Age")
axes[1, 0].set_xlabel("Age")
axes[1, 0].set_ylabel("Fare")

# Plot 4: Survival pie chart
axes[1, 1].pie([549, 342], labels=["Died", "Survived"], autopct='%1.1f%%',
               colors=['#ff6b6b', '#51cf66'], startangle=90)
axes[1, 1].set_title("Survival Rate")

plt.tight_layout()
plt.savefig("reports/eda_overview.png", dpi=150)
plt.close()
print("    reports/eda_overview.png created")

# -------- 2. SHAP Beeswarm (for Page 4) --------
plt.figure(figsize=(8, 6))
# Simulate SHAP values
np.random.seed(42)
for i, feature in enumerate(["Sex", "Pclass", "Age", "Fare", "Title"]):
    values = np.random.randn(50) * 0.2 + (0.5 - i * 0.1)
    plt.scatter(values, [i] * 50 + np.random.randn(50) * 0.1, alpha=0.6, s=30)

plt.yticks(range(5), ["Sex", "Pclass", "Age", "Fare", "Title"])
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel("SHAP value (impact on model output)")
plt.title("SHAP Beeswarm Plot (Placeholder)")
plt.tight_layout()
plt.savefig("reports/shap_beeswarm.png", dpi=150)
plt.close()
print("    reports/shap_beeswarm.png created")

# -------- 3. SHAP Bar Chart (for Page 4) --------
plt.figure(figsize=(6, 4))
features = ["Sex", "Pclass", "Fare", "Age", "Title"]
importance = [0.42, 0.28, 0.15, 0.10, 0.05]
bars = plt.barh(features, importance, color='#339af0')
plt.xlabel("Mean |SHAP value| (average impact)")
plt.title("Feature Importance (SHAP Bar Plot)")
for bar, val in zip(bars, importance):
    plt.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.2f}', va='center')
plt.tight_layout()
plt.savefig("reports/shap_bar.png", dpi=150)
plt.close()
print("    reports/shap_bar.png created")

print("\n All placeholder images generated successfully!")
print(" Location: reports/ folder")