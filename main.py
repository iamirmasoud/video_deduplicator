import logging
import math
import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import configs
from dbscan_features import dbscan_features
from model import get_model
from preprocessing import Preprocessing
from videos_dataset import VideosDataset

logger = logging.getLogger(__name__)
logging.basicConfig(
    level="INFO",
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
videos_dir = configs.videos_dir

if len(sys.argv) > 1:
    videos_dir = sys.argv[1]

if not os.path.isdir(videos_dir):
    logger.error(f"Directory '{videos_dir}' does not exist.")
    raise NotADirectoryError

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {str(device).upper()}")

dataset = VideosDataset(
    directory=videos_dir,
    extensions=configs.extensions,
    frame_rate=1 if configs.type == "2d" else 24,
    size=224 if configs.type == "2d" else 112,
    center_crop=(configs.type == "3d"),
)
logger.info(f"Found {len(dataset)} video files in directory '{videos_dir}'.")


loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
)
preprocess = Preprocessing(configs.type)
model = get_model(model_type=configs.type, model_path=configs.model_path)

model.to(device)
model.eval()
features = {}
with torch.no_grad():
    for k, data in enumerate(tqdm(loader)):
        video, video_path = data[0], data[1][0]
        video = video.squeeze()
        if len(video.shape) == 4:
            logger.debug(f'Computing features of video "{video_path}"')
            video = preprocess(video)
            n_chunk = len(video)
            video_features = torch.cuda.FloatTensor(n_chunk, 2048).fill_(0)
            n_iter = int(math.ceil(n_chunk / float(configs.batch_size)))
            for i in range(n_iter):
                min_ind = i * configs.batch_size
                max_ind = (i + 1) * configs.batch_size
                video_batch = video[min_ind:max_ind].cuda()
                batch_features = model(video_batch)
                if configs.l2_normalize:
                    batch_features = F.normalize(batch_features, dim=1)
                video_features[min_ind:max_ind] = batch_features
            video_features = video_features.cpu().numpy()
            if configs.half_precision:
                video_features = video_features.astype("float16")
            features[video_path] = video_features
        else:
            logger.warning(f'Skipping video "{video_path}"')

logger.info("Features for all videos have been extracted.")

# group videos by their length
lengths_df = pd.DataFrame(
    {k: v.shape[0] for k, v in features.items()}.items(), columns=["path", "len"]
)
grouped_by_length = lengths_df.groupby("len")["path"]

# in each group (videos of the same length), perform dbscan clustering to find duplicates
files_to_remove = set()
for length, group in tqdm(grouped_by_length):
    paths = group.tolist()
    if len(paths) > 1:
        logger.info(f"Deduplicating ({len(paths)}) items with length {length}:")
        # flatten two dimensional series of videos to one dimensional time series for DBSCAN
        len_group_features = [features[path].flatten() for path in paths]

        features_df = pd.DataFrame(len_group_features, index=paths)
        to_remove = dbscan_features(features_df, epsilon=configs.epsilon)
        if to_remove:
            files_to_remove.update(to_remove)

logger.info(f"Found {len(files_to_remove)} duplicated files to remove!")


if configs.delete_files:
    for file_path in files_to_remove:
        try:
            os.remove(file_path)
        except OSError as e:
            print("Cannot delete file '%s': %s" % (file_path, e.strerror))
    logger.info(f"Successfully deleted {len(files_to_remove)} duplicate files.")
