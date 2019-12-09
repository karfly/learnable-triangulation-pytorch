Human3.6M preprocessing scripts
=======

These scripts help preprocess Human3.6M dataset so that it can be used with `class Human36MMultiViewDataset`.

Here is how we do it (brace yourselves):

0. Make sure you have a lot (around 200 GiB?) of free disk space. Otherwise, be prepared to always carefully delete intermediate files (e.g. after you extract movies, delete the archives).

1. Allocate a folder for the dataset. Make it accessible as `$THIS_REPOSITORY/data/human36m/`, i.e. either

    * store your data `$SOMEWHERE_ELSE` and make a soft symbolic link:
    ```bash
    mkdir $THIS_REPOSITORY/data
    ln -s $SOMEWHERE_ELSE $THIS_REPOSITORY/data/human36m
    ```
    * or just store the dataset along with the code at `$THIS_REPOSITORY/data/human36m/`.

1. Clone [this toolbox](https://github.com/anibali/h36m-fetch). Follow their manual to download, extract and unpack Human3.6M into image files. Move the result to `$THIS_REPOSITORY/data/human36m`.

    After that, you should have images unpacked as e.g. `$THIS_REPOSITORY/data/human36m/processed/S1/Phoning-1/imageSequence/54138969/img_000001.jpg`.

2. Additionally, if you want to use ground truth bounding boxes for training, download them as well (the website calls them *"Segments BBoxes MAT"*) and unpack them like so: `"$THIS_REPOSITORY/data/human36m/processed/S1/MySegmentsMat/ground_truth_bb/Phoning 1.58860488.mat"`.

3. Convert those bounding boxes into sane format. This will create `$THIS_REPOSITORY/data/human36m/extra/bboxes-Human36M-GT.npy`:

    ```bash
    cd $THIS_REPOSITORY/mvn/datasets/human36m_preprocessing
    # in our environment, this took around 6 minutes with 40 processes
    python3 collect-bboxes.py $THIS_REPOSITORY/data/human36m <number-of-parallel-processes>
    ```

4. Existing 3D keypoint positions and camera intrinsics are difficult to decipher, so initially we used the converted ones [from Julieta Martinez](https://github.com/una-dinosauria/3d-pose-baseline/):

    ```bash
    mkdir -p $THIS_REPOSITORY/data/human36m/extra/una-dinosauria-data
    cd $THIS_REPOSITORY/data/human36m/extra/una-dinosauria-data
    wget https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip
    unzip h36m.zip && rm h36m.zip
    cd -
    ```

5. Wrap the 3D keypoint positions, bounding boxes and camera intrinsics together. This will create `$THIS_REPOSITORY/data/human36m/extra/human36m-multiview-labels-GTbboxes.npy`:

    ```bash
    python3 generate-labels-npy-multiview.py $THIS_REPOSITORY/data/human36m $THIS_REPOSITORY/data/human36m/extra/una-dinosauria-data/h36m $THIS_REPOSITORY/data/human36m/extra/bboxes-Human36M-GT.npy
    ```

    You should see only one warning saying `camera 54138969 isn't present in S11/Directions-2`. That's fine.

    Now you can train and evaluate models by setting these config values (already set by default in the example configs):

    ```yaml
    dataset:
      {train,val}:
        h36m_root: "data/human36m/processed/"
        labels_path: "data/human36m/extra/human36m-multiview-labels-GTbboxes.npy"
    ```

6. To use `undistort_images: true`, undistort the images beforehand. This will put undistorted images to e.g. `$THIS_REPOSITORY/data/human36m/processed/S1/Phoning-1/imageSequence-undistorted/54138969/img_000001.jpg`:

    ```bash
    # in our environment, this took around 90 minutes with 50 processes
    python3 undistort-h36m.py $THIS_REPOSITORY/data/human36m $THIS_REPOSITORY/data/human36m/extra/human36m-multiview-labels-GTbboxes.npy <number-of-parallel-processes>`
    ```

    *TODO: move undistortion to the dataloader. We can do it on the fly during training.*

7. Optionally, you can test if everything went well by viewing frames with skeletons and bounding boxes on a GUI machine:

    ```bash
    python3 view-dataset.py $THIS_REPOSITORY/data/human36m $THIS_REPOSITORY/data/human36m/extra/human36m-multiview-labels-GTbboxes.npy [<start-sample-number> [<samples-per-step>]]`
    ```

    You can test different settings by changing dataset constructor parameters in `view-dataset.py`.
