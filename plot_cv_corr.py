import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

cv_df = pd.read_csv(r"C:\Users\sunpu\lal-auto-essay-scoring2\cv_vs_lb.csv")

sns.set_style("darkgrid")
ax = sns.scatterplot(cv_df, x="cv", y="lb", size="fold_num", hue="type", legend="full")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

for idx, txt in enumerate(cv_df["exp"]):
    plt.text(cv_df["cv"][idx], cv_df["lb"][idx], txt)

# plt.grid()
plt.savefig("cv_vs_lb.png")
plt.show()
