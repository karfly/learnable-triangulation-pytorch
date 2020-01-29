from mvn.utils.vis import *

results_file = "results.pkl"

with open(fileName, "rb") as f:
    data = pickle.load(f)
    keypoints3d = data["keypoints_3d"]
    indexes = data["indexes"]

