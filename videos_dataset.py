import os
import pathlib

import ffmpeg
import numpy as np
import torch as th
from torch.utils.data import Dataset


class VideosDataset(Dataset):
    def list_files(
        self,
    ):
        """
        List all files in a directory with the given extensions.
        """
        files = [
            str(file)
            for ext in self.extensions
            for file in pathlib.Path(self.directory).rglob(f"*.{ext}")
        ]
        return files

    def __init__(
        self,
        directory,
        extensions=("mp4", "MP4", "avi", "AVI", "mov", "MOV"),
        frame_rate=1,
        size=112,
        center_crop=False,
    ):
        self.directory = directory
        self.extensions = extensions
        self.center_crop = center_crop
        self.size = size
        self.frame_rate = frame_rate
        self.video_paths = self.list_files()

    def __getitem__(self, index):
        video_path = self.video_paths[index]
        if os.path.isfile(video_path):
            try:
                h, w = self._get_video_dim(video_path)
            except Exception as e:
                print(f"ffprobe failed at: {video_path}")
                return th.zeros(1), video_path
            height, width = self._get_output_dim(h, w)
            cmd = (
                ffmpeg.input(video_path)
                .filter("fps", fps=self.frame_rate)
                .filter("scale", width, height)
            )
            if self.center_crop:
                x = int((width - self.size) / 2.0)
                y = int((height - self.size) / 2.0)
                cmd = cmd.crop(x, y, self.size, self.size)
            out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
                capture_stdout=True, quiet=True
            )
            if self.center_crop and isinstance(self.size, int):
                height, width = self.size, self.size
            video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
            video = th.from_numpy(video.astype("float32"))
            video = video.permute(0, 3, 1, 2)
        else:
            video = th.zeros(1)

        return video, video_path

    def __len__(self):
        return len(self.video_paths)

    def _get_video_dim(self, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        return height, width

    def _get_output_dim(self, h, w):
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif h >= w:
            return int(h * self.size / w), self.size
        else:
            return self.size, int(w * self.size / h)
