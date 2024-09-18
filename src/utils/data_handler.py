from torch.utils.data import Dataset

import glob

# Dataset classes

############################################################################
#                              Youtube-8m dataset                          #
############################################################################


class Youtube8mDataset(Dataset):
    def __init__(self, video_folder, annotation_files):
        self.video_folder = video_folder
        self.annotation_files = annotation_files

        self.video_files = glob.glob(f"{self.video_folder}/*.mp4")

    def __len__(self):
        len(self.video_files)

    def __getitem__(self, index):
        pass


############################################################################
#                              COIN dataset                                #
############################################################################


class COINDataset(Dataset):
    def __init__(self, video_folder, annotation_files):
        self.video_folder = video_folder
        self.annotation_files = annotation_files

        self.video_files = glob.glob(f"{self.video_folder}/*.mp4")

    def __len__(self):
        len(self.video_files)

    def __getitem__(self, index):
        pass


############################################################################
#                              Breakfest dataset                           #
############################################################################


class BreakfestDataset(Dataset):
    def __init__(self, video_folder, annotation_files):
        self.video_folder = video_folder
        self.annotation_files = annotation_files

        self.video_files = glob.glob(f"{self.video_folder}/*.mp4")

    def __len__(self):
        len(self.video_files)

    def __getitem__(self, index):
        pass


if __name__ == "__main__":
    test_dataset = COINDataset("data/coin/videos", "data/coin/COIN.json")
