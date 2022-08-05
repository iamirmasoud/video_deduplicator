import logging
import os
import pickle

import pandas as pd
from tqdm import tqdm

import configs
from dbscan_features import dbscan_features

logger = logging.getLogger(__name__)
logging.basicConfig(
    level="INFO",
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

with open("features.pkl", "rb") as handle:
    features = pickle.load(handle)


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
