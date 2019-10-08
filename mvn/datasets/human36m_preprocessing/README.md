Human3.6M preprocessing scripts
=======

These scripts help preprocess Human3.6M dataset so that it can be used in `Human36MMultiViewDataset` from [this system](https://github.sec.samsung.net/RRU8-VIOLET/multi-view-net/).

Here is how we do it (brace yourselves):

0. Make sure you have a lot (around 200 GiB?) of free disk space. Otherwise, be prepared to always carefully delete intermediate files (e.g. after you extract movies, delete the archives).

1. Clone [this repository](https://github.com/anibali/h36m-fetch). Follow their manual to download, extract and unpack Human3.6M into image files. After that, you should have images unpacked as e.g. `<dataset-root>/S1/Phoning-1/imageSequence/54138969/img_000001.jpg`.

2. Additionally, download ground truth bounding boxes and unpack them like so: `"<dataset-root>/S1/MySegmentsMat/ground_truth_bb/Phoning 1.58860488.mat"`.

3. Convert those bounding boxes into sane format: `python3 collect-bboxes.py <dataset-root> <number-of-parallel-processes>` (in our environment, this took around 6 minutes with 40 processes). This will produce `bboxes-Human36M-GT.npy` in your working directory.

4. Existing 3D keypoint positions and camera intrinsics are difficult to decipher, so initially we used the converted ones [from Julieta Martinez](https://github.com/una-dinosauria/3d-pose-baseline/):

    ```bash
    mkdir una-dinosauria-data
    cd una-dinosauria-data
    wget https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip
    unzip h36m.zip && rm h36m.zip
    cd ..
    ```

5. Wrap the 3D keypoint positions, bounding boxes and camera intrinsics together: `python3 generate-labels-npy-multiview.py <dataset-root> una-dinosauria-data/h36m bboxes-Human36M-GT.npy`. You should see only one warning saying `camera 54138969 isn't present in S11/Directions-2`. That's fine.

    The script should have created `human36m-multiview-labels-GTbboxes.npy` in your working directory. Now you can instantiate `Human36MMultiViewDataset` from `mvn/datasets/human36m.py` by feeding `labels_path=<path/to/human36m-multiview-labels-GTbboxes.npy>` and `h36m_root=<dataset-root>` to the constructor!

6. To use `undistort_images=True` in `Human36MMultiViewDataset`'s constructor, undistort the images beforehand:

    * Make sure you have write access in paths like `<dataset-root>/S1/Phoning-1/`
    * `pip3 install --user tqdm`
    * `python3 undistort-h36m.py <dataset-root> human36m-multiview-labels-GTbboxes.npy <number-of-parallel-processes>`

    *TODO: move undistortion to the dataloader. We can do it on the fly during training.*

7. Optionally, you can test if everything went well by viewing frames with skeletons and bounding boxes on a GUI machine:

    `python3 view-dataset.py <dataset-root> human36m-multiview-labels-GTbboxes.npy [<start-sample-number> [<samples-per-step>]]`

    You can test different settings by changing dataset constructor parameters in `view-dataset.py`.
