CMU Panoptic Dataset preprocessing scripts
=======

These scripts help preprocess CMU Panoptic dataset so that it can be used with `class Human36MMultiViewDataset`.

Here is how we do it (brace yourselves):

0. Make sure you have a lot (around 200+ GB) of free disk space. Otherwise, be prepared to always carefully delete intermediate files (e.g. after you extract movies, delete the archives).

1. Allocate a folder for the dataset. Make it accessible as `$THIS_REPOSITORY/data/cmupanoptic/`, i.e. either

    * store your data `$SOMEWHERE_ELSE` and make a soft symbolic link:
    ```bash
    mkdir $THIS_REPOSITORY/data
    ln -s $SOMEWHERE_ELSE $THIS_REPOSITORY/data/cmupanoptic
    ```
    * or just store the dataset along with the code at `$THIS_REPOSITORY/data/cmupanoptic/`.

1. Clone the [panoptic toolbox](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox). Follow their manual to download, extract and unpack CMU Panoptic dataset into image files (follow up to step 4 of the manual). 

There are many poses and scenes that you can use, for example, [`171204_pose1`](http://domedb.perception.cs.cmu.edu/171204_pose1.html) under the ["Range of Motion"](http://domedb.perception.cs.cmu.edu/range_of_motion.html) scene and [`170407_haggling_a2`](http://domedb.perception.cs.cmu.edu/170407_haggling_a2.html) under the ["Haggling"](http://domedb.perception.cs.cmu.edu/haggling.html) scene. It does not matter which set of scenes, poses or even images that you choose to extract; the scripts will handle that. It is advised that you only download the HD videos/images though.

However, you must copy them over with the right structure (in this case `SCENE_NAME` is `171026_pose1`):
```
FORMATTING/ORGANISATION OF FOLDERS & FILES

Images:
    $DIR_ROOT/[POSE_NAME]/hdImgs/[VIEW_ID]/[VIEW_ID]_[FRAME_ID].jpg
    (e.g.) ./171026_pose1/hdImgs/00_24/00_24_00012200.jpg

Pose Data:
    $DIR_ROOT/[POSE_NAME]/hdPose3d_stage1_coco19/body3DScene_[FRAME_ID].jpg
    (e.g.) ./171026_pose1/hdPose3d_stage1_coco19/body3DScene_00012200.json

    JSON data has this notable format:
    "bodies":[
        {
            "id": [PERSON_ID],
            "joints": [ ARRAY OF JOINT COORDINATES IN COCO 19 FORMAT]
        },
        {
            ...
        }
    ] 

Camera Calibration Data:
    $DIR_ROOT/[POSE_NAME]/calibration_[POSE_NAME].json
```

Move the folders to `$THIS_REPOSITORY/data/cmupanoptic`.

2. We need the BBOXES detections of each person in the scene, by camera. The script will handle this also, but you need to download the files from the original paper's [Google Drive](https://drive.google.com/drive/folders/1Nf2XPjHR4rw7-nESrrcoI8rMmdJmuxqX). See the original GitHub issue [#19](https://github.com/karfly/learnable-triangulation-pytorch/issues/19#issuecomment-545993330) for more details.

Save the bboxes in their original directory structure at `$THIS_REPOSITORY/data/pretrained/cmu`

3. Run `generate-labels-npy.py` to convert the data from native json format to an npy format which the `CMUPanopticDataset(Dataset)` class can use. The first argument is the directory to the CMU Panoptic Data; the second argument is to the pretrained BBOXES for the person detections:

```bash
python3 $THIS_REPOSITORY/data/cmupanoptic $THIS_REPOSITORY/data/pretrained/cmu
```


There will be an output file `cmu-multiview-labels-{BBOXES_SOURCE}bboxes.npy` in the `$THIS_REPOSITORY/data/cmupanoptic` folder
 
1. Optionally, you can test if everything went well by viewing frames with skeletons and bounding boxes on a GUI machine:

    ```bash
    python3 view-dataset.py $THIS_REPOSITORY/data/cmupanoptic $THIS_REPOSITORY/data/cmupanoptic/extra/human36m-multiview-labels-GTbboxes.npy [<start-sample-number> [<samples-per-step>]]`
    ```

    You can test different settings by changing dataset constructor parameters in `view-dataset.py`.
