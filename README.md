# Automatic Video Deduplicator using PyTorch

## Project Overview
Sometimes I have a directory with many videos that contains a lot of duplicated videos (with different names) among them. I decided to use machine learning and deep learning, instead of doing a manual search, to find and delete these duplicated videos. 

Here are some of the steps I will undertake to achieve this goal:
1. Write a custom PyTorch dataset to load the videos in a directory and their paths as a dataset.
2. Use Deep CNN architectures to embed (encode) the videos as feature vectors.
3. Group the videos with of same length.
4. Cluster the videos of each group using DBSCAN algorithm to find duplicated/similar videos.
5. (Optional) Delete the duplicate videos.


## Preparing the environment
**Note**: I have developed this project on __Linux__. It can surely be run on Windows and Mac with some little changes.

1. Clone the repository, and navigate to the downloaded folder.
```
git clone https://github.com/iamirmasoud/video_deduplicator.git
cd video_deduplicator
```

2. Create (and activate) a new environment, named `deduplicator_env` with Python 3.7. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	```shell
	conda create -n deduplicator_env python=3.7
	source activate deduplicator_env
	```
	
	At this point your command line should look something like: `(deduplicator_env) <User>:video_deduplicator <user>$`. The `(deduplicator_env)` indicates that your environment has been activated, and you can proceed with further package installations.

3. Before you can experiment with the code, you'll have to make sure that you have all the libraries and dependencies required to support this project. You will mainly need Python3.7+, PyTorch, and its torchvision. You can install dependencies using:
```
pip install -r requirements.txt
```

4. Navigate back to the repo. (Also, your source environment should still be activated at this point.)
```shell
cd video_deduplicator
```

## Project Structure

### Configs
You need to configure the project before you can run the code. You can do this by opening the `configs.py` file and editing the values. Here are some important configs.

- `videos_dir` - directory of videos to be deduplicated. 
- `extensions` - tuple of video file extensions to search when creating the dataset.
- `batch_size` - the batch size for the data loader when encoding videos. 
- `type` - model type to extract the feature from. To get features from the 3d model instead, just change this argument 2d per 3d.
- `epsilon` - epsilon parameter for DBSCAN (similarity threshold to put duplicate videos in a cluster).
- `delete_files` - whether to delete files from disk or not.

### Running step by step
I have created python scripts for each of the steps of the project. They should be run in sequential order. Here are the steps:
1. Extracting features and storing them for all videos (`1_feature_extraction.py`)
2. Perform clustering and deduplication process using the extracted features (`2_deduplication.py`)

### Running the main process

If you are not interested in the details of each step, you can just run the following python script to run the whole deduplication process:

```shell 
python main.py 
```

You can override the `videos_dir` config value by passing an argument to the script as below.

```shell 
python main.py <videos_directory> 
```

*Note*: The script is intended to be run on ONE single GPU only. If multiple GPUs are available, please make sure that only one free GPU is set visible by the script with the CUDA_VISIBLE_DEVICES variable environment for example.

## Video Feature Extractor

### Fast video feature extractor

Most of the time, extracting CNN features from a video is cumbersome. In fact, this usually requires dumping video frames into the disk, loading the dumped frames one by one, pre-processing them, and using a CNN to extract features on chunks of videos. This process is not efficient because of the dumping of frames on disk which is slow and can use a lot of inodes when working with a large dataset of videos. 
I used an easy to use and efficient code for extracting video features using deep CNN (2D or 3D). It has been originally designed to extract video features for the large scale video dataset [HowTo100M](https://www.di.ens.fr/willow/research/howto100m/) in an efficient manner.


### Deep models
So far, only one 2D and one 3D model can be used.

- The 2D model is the PyTorch model zoo ResNet-152 pretrained on ImageNet. The 2D features are extracted at 1 feature per second at the resolution of 224.
- The 3D model is a [ResNexT-101 16 frames](https://github.com/kenshohara/3D-ResNets-PyTorch) pretrained on Kinetics. The 3D features are extracted at 1.5 features per second at a resolution of 112.

### Downloading pretrained models
This will download the pretrained 3D ResNext-101 model I used from: https://github.com/kenshohara/3D-ResNets-PyTorch 

```sh
mkdir model
cd model
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/models/resnext101.pth
```


## Acknowledgments
The video feature extraction part of this project is reused from [this Github](https://github.com/antoine77340/video_feature_extractor).

