videos_dir = "videos"
extensions = ("mp4", "MP4", "avi", "AVI", "mov", "MOV")
type = "2d"
batch_size = 64
half_precision = 1
l2_normalize = 1
model_path = "model/resnext101.pth"
epsilon = 0.001
# Warning: Setting the following config to True will result in removing the detected duplicate videos from disk.
# If you are not sure to do this, you can check and then delete the detected videos manually one by one.
delete_files = True
