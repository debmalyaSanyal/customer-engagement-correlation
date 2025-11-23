import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 1. Generate synthetic data
# -----------------------------
np.random.seed(42)
n_customers = 250

# Core engagement metrics (synthetic but realistic)
purchase_frequency = np.random.poisson(lam=3, size=n_customers)  # visits per month
avg_session_duration = np.random.normal(loc=8, scale=3, size=n_customers)  # minutes
avg_session_duration = np.clip(avg_session_duration, 1, 30)

email_open_rate = np.random.normal(
    loc=0.35 + 0.02 * purchase_frequency,  # more frequent buyers open more emails
    scale=0.07,
    size=n_customers
)
email_open_rate = np.clip(email_open_rate, 0, 1)

click_through_rate = np.random.normal(
    loc=0.08 + 0.8 * email_open_rate,  # CTR strongly linked to open rate
    scale=0.05,
    size=n_customers
)
click_through_rate = np.clip(click_through_rate, 0, 0.8)

loyalty_score = np.random.normal(
    loc=50 + 5 * purchase_frequency + 10 * click_through_rate,  # higher for engaged users
    scale=10,
    size=n_customers
)
loyalty_score = np.clip(loyalty_score, 0, 100)

social_engagement_index = np.random.normal(
    loc=20 + 3 * purchase_frequency,
    scale=10,
    size=n_customers
)
social_engagement_index = np.clip(social_engagement_index, 0, None)

customer_lifetime_value = np.random.normal(
    loc=200 + 40 * purchase_frequency + 100 * (loyalty_score / 100),
    scale=100,
    size=n_customers
)
customer_lifetime_value = np.clip(customer_lifetime_value, 0, None)

# Build DataFrame
df = pd.DataFrame({
    "Purchase_Frequency": purchase_frequency,
    "Avg_Session_Duration_min": avg_session_duration,
    "Email_Open_Rate": email_open_rate,
    "Click_Through_Rate": click_through_rate,
    "Loyalty_Score": loyalty_score,
    "Social_Engagement_Index": social_engagement_index,
    "Customer_Lifetime_Value": customer_lifetime_value
})

# -----------------------------
# 2. Compute correlation matrix
# -----------------------------
corr_matrix = df.corr(method="pearson")

# -----------------------------
# 3. Seaborn styling
# -----------------------------
sns.set_style("whitegrid")
sns.set_context("talk")  # presentation-ready font sizes

# -----------------------------
# 4. Create heatmap
# -----------------------------
plt.figure(figsize=(8, 8))  # 8 inches * 64 dpi = 512 pixels

ax = sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    vmin=-1, vmax=1,
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": 0.8}
)

plt.title("Customer Engagement Metrics Correlation Matrix", pad=20)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

# -----------------------------
# 5. Save as 512x512 PNG
# -----------------------------
plt.savefig("chart.png", dpi=64, bbox_inches="tight")
plt.close()
