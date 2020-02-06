from mvn.utils.vis import *
import pickle

results_file = "results/results.pkl"

with open(results_file, "rb") as f:
    data = pickle.load(f)
    keypoints3d = data["keypoints_3d"]
    indexes = data["indexes"]

print(len(keypoints3d))

'''
n_cols, n_rows = 64, 64
size = 1

fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows,
                         figsize=(n_cols * size, n_rows * size))

draw_3d_pose(keypoints3d[0], axes)
plt.show()
'''
