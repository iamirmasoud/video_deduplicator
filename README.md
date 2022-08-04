# Automatic Video Deduplicator using PyTorch

## Project Overview
Sometimes I have a directory with many videos that contains a lot of duplicated videos (with different names) among them. I decided to use machine learning and deep learning, instead of doing a manual search, to find and delete these duplicated videos. 

Here are some of the steps I will undertake to achieve this goal:
1. Write a custom PyTorch dataset to load the videos in a directory and their paths as a dataset.
2. Use a Deep CNN architecture as an Encoder to embed the videos as feature vectors.
3. Cluster the videos using DBSCAN algorithm to find duplicate/similar videos.
4. (Optional) Delete the duplicate videos.


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

8. Open the directory of notebooks, using the below command. You'll see all of the project files appear in your local environment; open the first notebook and follow the instructions.
```shell
jupyter notebook
```

9. Once you open any of the project notebooks, make sure you are in the correct `deduplicator_env` environment by clicking `Kernel > Change Kernel > deduplicator_env`.

## Project Structure

### Configs
You need to configure the project before you can run the code. You can do this by opening the `configs.py` file and editing the values:

- `videos_dir` - directory of videos to be deduplicated. 
- `extensions` - tuple of video file extensions to search when creating the dataset.
- `batch_size` - the batch size for the data loader when encoding videos. 
- `embedding_size` - the dimensionality of the video embeddings. 
- `video_resize` - size of resizing transform for CNN input.
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



