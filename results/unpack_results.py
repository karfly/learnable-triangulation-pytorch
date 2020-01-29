import pickle
import numpy as np
import sys

fileName = "results.pkl"

with open(fileName, "rb") as f:
    data = pickle.load(f);

fileOutput = "keypoints.txt"
fileOutput2 = "indexes.txt"

np.set_printoptions(threshold=sys.maxsize, precision=None, suppress=True)

with open(fileOutput, "w+") as f:
    f.write(str(data["keypoints_3d"]))

with open(fileOutput2, "w+") as f:
    f.write(str(data["indexes"]))