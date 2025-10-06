import json
import os


def download_coin_datasets(json_path, output_path):
    """Download videos from COIN dataset using youtube-dl"""

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    data = json.load(open(json_path, "r"))["database"]
    youtube_ids = list(data.keys())

    for youtube_id in data:
        info = data[youtube_id]
        type = info["recipe_type"]
        url = info["video_url"]
        vid_loc = output_path + "/" + str(type)
        if not os.path.exists(vid_loc):
            os.mkdir(vid_loc)

        if any([youtube_id in f for f in os.listdir(vid_loc)]):
            print("Already downloaded: " + youtube_id)
            continue

        os.system(
            "yt-dlp -o "
            + vid_loc
            + "/"
            + youtube_id
            + ".mp4"
            + " --cookies-from-browser firefox "
            + url
        )


def download_youtube8m_dataset(videos_output_path, labels_output_path, num_files=5):
    """
    Download a subset of YouTube-8M raw video files and labels.
    Args:
        videos_output_path (str): Directory to save video files.
        labels_output_path (str): Directory to save label files.
        num_files (int): Number of files to download for each (for demo purposes).
    """
    import urllib.request

    # URLs for YouTube-8M sample data (change as needed for full dataset)
    video_urls = [
        f"https://storage.googleapis.com/youtube8m-ml/2/frame/train/train-{i:05d}-of-00010.tfrecord"
        for i in range(num_files)
    ]
    label_urls = [
        f"https://storage.googleapis.com/youtube8m-ml/2/annotations/train/train-{i:05d}-of-00010.csv"
        for i in range(num_files)
    ]

    os.makedirs(videos_output_path, exist_ok=True)
    os.makedirs(labels_output_path, exist_ok=True)

    print("Downloading YouTube-8M video files...")
    for url in video_urls:
        filename = os.path.join(videos_output_path, os.path.basename(url))
        if not os.path.exists(filename):
            print(f"Downloading {filename} ...")
            urllib.request.urlretrieve(url, filename)
        else:
            print(f"Already downloaded: {filename}")

    print("Downloading YouTube-8M label files...")
    for url in label_urls:
        filename = os.path.join(labels_output_path, os.path.basename(url))
        if not os.path.exists(filename):
            print(f"Downloading {filename} ...")
            urllib.request.urlretrieve(url, filename)
        else:
            print(f"Already downloaded: {filename}")


if __name__ == "__main__":
    output_path = "/media/gustavo-furlanetto/KINGSTON/datasets/coin/videos"
    json_path = "/media/gustavo-furlanetto/KINGSTON/datasets/coin/COIN.json"

    download_coin_datasets(json_path, output_path)
