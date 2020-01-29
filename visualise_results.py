from mvn.utils.vis import *

results_file = "results/results.pkl"

with open(fileName, "rb") as f:
    data = pickle.load(f)
    keypoints3d = data["keypoints_3d"]
    indexes = data["indexes"]

draw_3d_pose(keypoints3d, axes3d)
plt.show()