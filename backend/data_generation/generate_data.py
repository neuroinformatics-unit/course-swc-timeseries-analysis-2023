"""
Script to create data for timeseries analysis course from raw data. This will
load the pre-drug and post-drug files and adjust the baseline so they
can be concatenated together.

cell5-pre-drug, remove stimulus, take first 60 seconds, save as CSV
cell3-cch, remove stimulus, take first 60 seconds

NOTES:

annoying warning in older pycharm: https://youtrack.jetbrains.com/issue/PY-56361/FutureWarning-pandas.Series.iteritems-in-pydevdutils.py
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

repo_path = Path(r"C:\Users\Joe\work\git-repos\swc-timeseries-analysis-course-2023\backend\data_generation")
pre_drug = pd.read_csv(repo_path / "cell5-predrug-cut.csv")
post_drug = pd.read_csv(repo_path / "cell3-cch-cut.csv", )

# Load and check data
pre_drug_im = pre_drug["Im"].to_numpy()
pre_drug_time = pre_drug["Time(s)"].to_numpy()

post_drug_im = post_drug["Im"].to_numpy()
post_drug_time = post_drug["Time(s)"].to_numpy()

assert np.array_equal(pre_drug_time, post_drug_time)

plt.plot(pre_drug_time, pre_drug_im)
plt.show()

plt.plot(post_drug_time, post_drug_im)
plt.show()

# adjust baseline of post-drug cell to match pre-drug
diff = np.median(pre_drug_im[-1000:]) - np.median(post_drug_im[:1000])
adj_baseline_post_drug_im = post_drug_im + diff

plt.plot(post_drug_time,
         adj_baseline_post_drug_im)
plt.show()

# Concatenate and plot, create time new
new_data = np.r_[pre_drug_im, adj_baseline_post_drug_im]

ts = pre_drug_time[1]

assert np.array_equal(pre_drug_time.astype(np.float32),  np.linspace(0, 60-ts, pre_drug_time.size).astype(np.float32))  #  sanity check the time creation method

new_time = np.linspace(0, 120-ts, new_data.size).astype(np.float32)

plt.plot(new_time, new_data)
plt.show()

# get the time to split data, save as csv
split_time = pre_drug_time[-1] + ts
print(split_time)

new_dataframe = pd.DataFrame(np.c_[new_time, new_data], columns=["time_s", "current_pa"]).astype(np.float32)
new_dataframe.to_csv(repo_path / "sub-001_drug-cch_origdata.csv", index=None)



