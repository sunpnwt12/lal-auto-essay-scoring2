import matplotlib.pyplot as plt
import pandas as pd

cv_record_df = pd.read_csv(r"C:\Users\sunpu\lal-auto-essay-scoring2\cv_vs_lb.csv")
cv_array = cv_record_df["cv"]
lb_array = cv_record_df["lb"]
diff_array = cv_record_df["diff"]
exp_array = cv_record_df["exp"]


fig = plt.figure()
ax = plt.axes(projection="3d")

ax.scatter3D(cv_array, diff_array, lb_array)
ax.set_xlabel("cv")
ax.set_ylabel("diff")
ax.set_zlabel("lb")

plt.show()
