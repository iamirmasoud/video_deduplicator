# Automatic Video Deduplicator using PyTorch

## Project Overview
Sometimes I have a directory with many videos that contains a lot of duplicated videos (with different names) among them. I decided to use machine learning and deep learning, instead of doing a manual search, to find and delete these duplicated videos. 

Here are some of the steps I will undertake to achieve this goal:
1. Write a custom PyTorch dataset to load the videos in a directory and their paths as a dataset.
2. Use Deep CNN architectures to embed (encode) the videos as feature vectors.
3. Group the videos with the same length.
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

6. Before you can experiment with the code, you'll have to make sure that you have all the libraries and dependencies required to support this project. You will mainly need Python3.7+, PyTorch and its torchvision, OpenCV, and Matplotlib. You can install  dependencies using:
```
pip install -r requirements.txt
```

7. Navigate back to the repo. (Also, your source environment should still be activated at this point.)
```shell
cd video_deduplicator
```

## Project Structure

### Configs
You need to configure the project before you can run the code. You can do this by opening the `configs.py` file and editing the values:

- `videos_dir` - directory of videos to be deduplicated. 
- `extensions` - tuple of video file extensions to search when creating the dataset.
- `batch_size` - the batch size for the data loader when encoding videos. 
- `embedding_size` - the dimensionality of the video embeddings. 
- `epsilon` - epsilon parameter for DBSCAN (similarity threshold to put duplicate videos in a cluster).
- `delete_files` - whether to delete files from disk or not.

### Jupyter Notebooks
I have created jupyter notebook files for each of the steps of the project. They provide details of each step and should be run in sequential order.

### Running the main process

If you are not interested in the details of each step, you can just run the following python script to run the whole deduplication process:

```shell 
python main.py 
```

You can override the `videos_dir` config value by passing an argument to the script as below.

```shell 
python main.py <videos_directory> 
```


## Architecture
![Encoder Model](assets/encoder.png?raw=true) [Video source](https://arxiv.org/pdf/1411.4555.pdf)

The `EncoderCNN` class in `model.py` encodes the critical information contained in a regular picture file into a "feature vector" of a specific size. The CNN encoder is a pre-trained ResNet on VideoNet, which is a VGG convolutional neural network with skip connections. It has been proven to work really well on tasks like video recognition because the residual connections help model the residual differences before and after the convolution with the help of the identity block. A good pre-trained network on VideoNet is already good at extracting both useful low-level and high-level features for video tasks, so it naturally serves as a feature encoder. Since I am not doing the traditional video classification task, I drop the last fully connected layer and replace it without a new trainable fully connected layer.



# Fast and Easy to use video feature extractor

This repo aims at providing an easy to use and efficient code for extracting
video features using deep CNN (2D or 3D).

It has been originally designed to extract video features for the large scale video dataset HowTo100M (https://www.di.ens.fr/willow/research/howto100m/) in an efficient manner.


Most of the time, extracting CNN features from video is cumbersome.
In fact, this usually requires dumping video frames into the disk, loading the dumped frames one
by one, pre processing them and use a CNN to extract features on chunks of videos.
This process is not efficient because of the dumping of frames on disk which is
slow and can use a lot of inodes when working with large dataset of videos.

To avoid having to do that, this repo provides a simple python script for that task: Just provide a list of raw videos and the script will take care of on the fly video decoding (with ffmpeg) and feature extraction using state-of-the-art models. While being fast, it also happen to be very convenient.

This script is also optimized for multi processing GPU feature extraction.


# Requirements
- Python 3
- PyTorch (>= 1.0)
- ffmpeg-python (https://github.com/kkroening/ffmpeg-python)

# How To Use ?

First of all you need to generate a csv containing the list of videos you
want to process. For instance, if you have video1.mp4 and video2.webm to process,
you will need to generate a csv of this form:

```
video_path,feature_path
absolute_path_video1.mp4,absolute_path_of_video1_features.npy
absolute_path_video2.webm,absolute_path_of_video2_features.npy
```

And then just simply run:

```sh
python extract.py --csv=input.csv --type=2d --batch_size=64 --num_decoding_thread=4
```
This command will extract 2d video feature for video1.mp4 (resp. video2.webm) at path_of_video1_features.npy (resp. path_of_video2_features.npy) in
a form of a numpy array.
To get feature from the 3d model instead, just change type argument 2d per 3d.
The parameter --num_decoding_thread will set how many parallel cpu thread are used for the decoding of the videos.

Please note that the script is intended to be run on ONE single GPU only.
if multiple gpu are available, please make sure that only one free GPU is set visible
by the script with the CUDA_VISIBLE_DEVICES variable environnement for example.

# Can I use multiple GPU to speed up feature extraction ?

Yes ! just run the same script with same input csv on another GPU (that can be from a different machine, provided that the disk to output the features is shared between the machines). The script will create a new feature extraction process that will only focus on processing the videos that have not been processed yet, without overlapping with the other extraction process already running.

# What models are implemented ?
So far, only one 2D and one 3D models can be used.

- The 2D model is the pytorch model zoo ResNet-152 pretrained on ImageNet. The 2D features are extracted at 1 feature per second at the resolution of 224.
- The 3D model is a ResNexT-101 16 frames (https://github.com/kenshohara/3D-ResNets-PyTorch) pretrained on Kinetics. The 3D features are extracted at 1.5 feature per second at the resolution of 112.

# Downloading pretrained models
This will download the pretrained 3D ResNext-101 model we used from: https://github.com/kenshohara/3D-ResNets-PyTorch 

```sh
mkdir model
cd model
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/models/resnext101.pth
```



### Acknowledgements
The video feature extraction part of this project is reused from [this Github](https://github.com/antoine77340/video_feature_extractor).

