import pathlib

from PIL import Image
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
        transform=None,
    ):
        self.directory = directory
        self.extensions = extensions
        self.transform = transform
        self.video_paths = self.list_files()

    def __getitem__(self, index):
        video_path = self.video_paths[index]
        video = Image.open(video_path).convert("RGB")
        if self.transform is not None:
            video = self.transform(video)
        return video, video_path

    def __len__(self):
        return len(self.video_paths)
