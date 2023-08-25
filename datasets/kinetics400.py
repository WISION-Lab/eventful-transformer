import csv
import shutil
from pathlib import Path
from sys import stderr

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from tqdm import tqdm

from utils.misc import decode_video, download_file, seeded_shuffle

CLASSES = [
    "abseiling",
    "air drumming",
    "answering questions",
    "applauding",
    "applying cream",
    "archery",
    "arm wrestling",
    "arranging flowers",
    "assembling computer",
    "auctioning",
    "baby waking up",
    "baking cookies",
    "balloon blowing",
    "bandaging",
    "barbequing",
    "bartending",
    "beatboxing",
    "bee keeping",
    "belly dancing",
    "bench pressing",
    "bending back",
    "bending metal",
    "biking through snow",
    "blasting sand",
    "blowing glass",
    "blowing leaves",
    "blowing nose",
    "blowing out candles",
    "bobsledding",
    "bookbinding",
    "bouncing on trampoline",
    "bowling",
    "braiding hair",
    "breading or breadcrumbing",
    "breakdancing",
    "brush painting",
    "brushing hair",
    "brushing teeth",
    "building cabinet",
    "building shed",
    "bungee jumping",
    "busking",
    "canoeing or kayaking",
    "capoeira",
    "carrying baby",
    "cartwheeling",
    "carving pumpkin",
    "catching fish",
    "catching or throwing baseball",
    "catching or throwing frisbee",
    "catching or throwing softball",
    "celebrating",
    "changing oil",
    "changing wheel",
    "checking tires",
    "cheerleading",
    "chopping wood",
    "clapping",
    "clay pottery making",
    "clean and jerk",
    "cleaning floor",
    "cleaning gutters",
    "cleaning pool",
    "cleaning shoes",
    "cleaning toilet",
    "cleaning windows",
    "climbing a rope",
    "climbing ladder",
    "climbing tree",
    "contact juggling",
    "cooking chicken",
    "cooking egg",
    "cooking on campfire",
    "cooking sausages",
    "counting money",
    "country line dancing",
    "cracking neck",
    "crawling baby",
    "crossing river",
    "crying",
    "curling hair",
    "cutting nails",
    "cutting pineapple",
    "cutting watermelon",
    "dancing ballet",
    "dancing charleston",
    "dancing gangnam style",
    "dancing macarena",
    "deadlifting",
    "decorating the christmas tree",
    "digging",
    "dining",
    "disc golfing",
    "diving cliff",
    "dodgeball",
    "doing aerobics",
    "doing laundry",
    "doing nails",
    "drawing",
    "dribbling basketball",
    "drinking",
    "drinking beer",
    "drinking shots",
    "driving car",
    "driving tractor",
    "drop kicking",
    "drumming fingers",
    "dunking basketball",
    "dying hair",
    "eating burger",
    "eating cake",
    "eating carrots",
    "eating chips",
    "eating doughnuts",
    "eating hotdog",
    "eating ice cream",
    "eating spaghetti",
    "eating watermelon",
    "egg hunting",
    "exercising arm",
    "exercising with an exercise ball",
    "extinguishing fire",
    "faceplanting",
    "feeding birds",
    "feeding fish",
    "feeding goats",
    "filling eyebrows",
    "finger snapping",
    "fixing hair",
    "flipping pancake",
    "flying kite",
    "folding clothes",
    "folding napkins",
    "folding paper",
    "front raises",
    "frying vegetables",
    "garbage collecting",
    "gargling",
    "getting a haircut",
    "getting a tattoo",
    "giving or receiving award",
    "golf chipping",
    "golf driving",
    "golf putting",
    "grinding meat",
    "grooming dog",
    "grooming horse",
    "gymnastics tumbling",
    "hammer throw",
    "headbanging",
    "headbutting",
    "high jump",
    "high kick",
    "hitting baseball",
    "hockey stop",
    "holding snake",
    "hopscotch",
    "hoverboarding",
    "hugging",
    "hula hooping",
    "hurdling",
    "hurling (sport)",
    "ice climbing",
    "ice fishing",
    "ice skating",
    "ironing",
    "javelin throw",
    "jetskiing",
    "jogging",
    "juggling balls",
    "juggling fire",
    "juggling soccer ball",
    "jumping into pool",
    "jumpstyle dancing",
    "kicking field goal",
    "kicking soccer ball",
    "kissing",
    "kitesurfing",
    "knitting",
    "krumping",
    "laughing",
    "laying bricks",
    "long jump",
    "lunge",
    "making a cake",
    "making a sandwich",
    "making bed",
    "making jewelry",
    "making pizza",
    "making snowman",
    "making sushi",
    "making tea",
    "marching",
    "massaging back",
    "massaging feet",
    "massaging legs",
    "massaging person's head",
    "milking cow",
    "mopping floor",
    "motorcycling",
    "moving furniture",
    "mowing lawn",
    "news anchoring",
    "opening bottle",
    "opening present",
    "paragliding",
    "parasailing",
    "parkour",
    "passing American football (in game)",
    "passing American football (not in game)",
    "peeling apples",
    "peeling potatoes",
    "petting animal (not cat)",
    "petting cat",
    "picking fruit",
    "planting trees",
    "plastering",
    "playing accordion",
    "playing badminton",
    "playing bagpipes",
    "playing basketball",
    "playing bass guitar",
    "playing cards",
    "playing cello",
    "playing chess",
    "playing clarinet",
    "playing controller",
    "playing cricket",
    "playing cymbals",
    "playing didgeridoo",
    "playing drums",
    "playing flute",
    "playing guitar",
    "playing harmonica",
    "playing harp",
    "playing ice hockey",
    "playing keyboard",
    "playing kickball",
    "playing monopoly",
    "playing organ",
    "playing paintball",
    "playing piano",
    "playing poker",
    "playing recorder",
    "playing saxophone",
    "playing squash or racquetball",
    "playing tennis",
    "playing trombone",
    "playing trumpet",
    "playing ukulele",
    "playing violin",
    "playing volleyball",
    "playing xylophone",
    "pole vault",
    "presenting weather forecast",
    "pull ups",
    "pumping fist",
    "pumping gas",
    "punching bag",
    "punching person (boxing)",
    "push up",
    "pushing car",
    "pushing cart",
    "pushing wheelchair",
    "reading book",
    "reading newspaper",
    "recording music",
    "riding a bike",
    "riding camel",
    "riding elephant",
    "riding mechanical bull",
    "riding mountain bike",
    "riding mule",
    "riding or walking with horse",
    "riding scooter",
    "riding unicycle",
    "ripping paper",
    "robot dancing",
    "rock climbing",
    "rock scissors paper",
    "roller skating",
    "running on treadmill",
    "sailing",
    "salsa dancing",
    "sanding floor",
    "scrambling eggs",
    "scuba diving",
    "setting table",
    "shaking hands",
    "shaking head",
    "sharpening knives",
    "sharpening pencil",
    "shaving head",
    "shaving legs",
    "shearing sheep",
    "shining shoes",
    "shooting basketball",
    "shooting goal (soccer)",
    "shot put",
    "shoveling snow",
    "shredding paper",
    "shuffling cards",
    "side kick",
    "sign language interpreting",
    "singing",
    "situp",
    "skateboarding",
    "ski jumping",
    "skiing (not slalom or crosscountry)",
    "skiing crosscountry",
    "skiing slalom",
    "skipping rope",
    "skydiving",
    "slacklining",
    "slapping",
    "sled dog racing",
    "smoking",
    "smoking hookah",
    "snatch weight lifting",
    "sneezing",
    "sniffing",
    "snorkeling",
    "snowboarding",
    "snowkiting",
    "snowmobiling",
    "somersaulting",
    "spinning poi",
    "spray painting",
    "spraying",
    "springboard diving",
    "squat",
    "sticking tongue out",
    "stomping grapes",
    "stretching arm",
    "stretching leg",
    "strumming guitar",
    "surfing crowd",
    "surfing water",
    "sweeping floor",
    "swimming backstroke",
    "swimming breast stroke",
    "swimming butterfly stroke",
    "swing dancing",
    "swinging legs",
    "swinging on something",
    "sword fighting",
    "tai chi",
    "taking a shower",
    "tango dancing",
    "tap dancing",
    "tapping guitar",
    "tapping pen",
    "tasting beer",
    "tasting food",
    "testifying",
    "texting",
    "throwing axe",
    "throwing ball",
    "throwing discus",
    "tickling",
    "tobogganing",
    "tossing coin",
    "tossing salad",
    "training dog",
    "trapezing",
    "trimming or shaving beard",
    "trimming trees",
    "triple jump",
    "tying bow tie",
    "tying knot (not on a tie)",
    "tying tie",
    "unboxing",
    "unloading truck",
    "using computer",
    "using remote controller (not gaming)",
    "using segway",
    "vault",
    "waiting in line",
    "walking the dog",
    "washing dishes",
    "washing feet",
    "washing hair",
    "washing hands",
    "water skiing",
    "water sliding",
    "watering plants",
    "waxing back",
    "waxing chest",
    "waxing eyebrows",
    "waxing legs",
    "weaving basket",
    "welding",
    "whistling",
    "windsurfing",
    "wrapping present",
    "wrestling",
    "writing",
    "yawning",
    "yoga",
    "zumba",
]

CLASS_IDS = {name: i for i, name in enumerate(CLASSES)}

SPLITS = ["train", "test", "val"]

# https://github.com/cvdfoundation/kinetics-dataset/blob/main/k400_downloader.sh
LABEL_DOWNLOADS = {
    split: f"https://s3.amazonaws.com/kinetics/400/annotations/{split}.csv"
    for split in SPLITS
}
VIDEO_DOWNLOADS = {
    split: f"https://s3.amazonaws.com/kinetics/400/{split}/k400_{split}_path.txt"
    for split in SPLITS
}


class Kinetics400(Dataset):
    """
    A loader for the Kinetics-400 dataset.
    """

    def __init__(
        self,
        location,
        split="val",
        decode_size=None,
        decode_fps=None,
        max_tars=None,
        shuffle=True,
        shuffle_seed=42,
        video_transform=None,
    ):
        """
        Initializes the loader. On the first call, this constructor will
        do some one-time setup (including downloading data).

        :param location: Directory where the dataset should be stored
        :param split: Either "train", "test", or "val"
        :param decode_size: The short-edge length for decoded frames
        :param decode_fps: The fps for decoded frames
        :param max_tars: Set a cap on the number of tar files to
        download for this split. Each tar contains about 1k videos.
        :param shuffle: Whether to shuffle videos
        :param shuffle_seed: The seed to use if shuffling
        :param video_transform: A callable to be applied to each video
        as it is loaded
        """
        assert split in SPLITS
        self.video_transform = video_transform

        base_split = split
        if max_tars is not None:
            split = f"{split}_{max_tars}"

        # Make sure the dataset has been set up.
        Path(location, split).mkdir(parents=True, exist_ok=True)
        if not self.is_downloaded(location, split):
            self.clean_downloaded(location, split)
            self.download(location, base_split, split, max_tars)
        if not self.is_unpacked(location, split):
            self.clean_unpacked(location, split)
            self.unpack(location, split)
        if not self.is_decoded(location, split, decode_size, decode_fps):
            self.clean_decoded(location, split, decode_size, decode_fps)
            self.decode(location, split, decode_size, decode_fps)

        # Load information about each video in the dataset.
        self.frames_path = Path(location, split, f"frames_{decode_size}_{decode_fps}")
        self.videos_info = self._get_videos_info(
            location, split, decode_size, decode_fps
        )

        # Optionally shuffle the videos (by default they are sorted).
        if shuffle:
            seeded_shuffle(self.videos_info, shuffle_seed)

    def __getitem__(self, index):
        """
        Loads and returns an item from the dataset.

        :param index: The index of the item to load
        :return: A (video, label) tuple, where "video" is a tensor and
        "label" is the class label.
        """
        video_info = self.videos_info[index]
        video_path = self.frames_path / video_info["video_id"]
        video = torch.stack(
            [read_image(str(video_path / frame)) for frame in video_info["frames"]]
        )
        if self.video_transform is not None:
            video = self.video_transform(video)
        return video, video_info["label"]

    def __len__(self):
        """
        Returns the number of items in the dataset.
        """
        return len(self.videos_info)

    @staticmethod
    def clean_decoded(location, split, decode_size, decode_fps):
        """
        Deletes one-time setup data (decoded frames).

        :param location: The location of the dataset (see __init__)
        :param split: The split name
        :param decode_size: The short-edge length for decoded frames
        :param decode_fps: The fps for decoded frames
        """
        base_path = Path(location, split)
        (base_path / f"decoded_{decode_size}_{decode_fps}").unlink(missing_ok=True)
        folder_path = base_path / f"frames_{decode_size}_{decode_fps}"
        if folder_path.is_dir():
            shutil.rmtree(folder_path)

    @staticmethod
    def clean_downloaded(location, split):
        """
        Deletes downloaded data (e.g., tar files).

        :param location: The location of the dataset (see __init__)
        :param split: The split name
        """
        base_path = Path(location, split)
        (base_path / "downloaded").unlink(missing_ok=True)
        (base_path / "labels.csv").unlink(missing_ok=True)
        folder_path = base_path / "downloads"
        if folder_path.is_dir():
            shutil.rmtree(folder_path)

    @staticmethod
    def clean_unpacked(location, split):
        """
        Deletes one-time setup data (unpacked tars).

        :param location: The location of the dataset (see __init__)
        :param split: The split name
        """
        base_path = Path(location, split)
        (base_path / "unpacked").unlink(missing_ok=True)
        folder_path = base_path / "videos"
        if folder_path.is_dir():
            shutil.rmtree(folder_path)

    @staticmethod
    def decode(location, split, decode_size, decode_fps):
        """
        Performs one-time setup (frame decoding).

        :param location: The location of the dataset (see __init__)
        :param split: The split name
        :param decode_size: The short-edge length for decoded frames
        :param decode_fps: The fps for decoded frames
        """
        base_path = Path(location, split)
        frames_path = base_path / f"frames_{decode_size}_{decode_fps}"
        frames_path.mkdir(exist_ok=True)

        # Decode videos into images.
        print("Decoding videos...", file=stderr, flush=True)
        video_list = list((base_path / "videos").glob("*.mp4"))
        for video_path in tqdm(video_list, total=len(video_list), ncols=0):
            ffmpeg_output_args = ["-qscale:v", "2"]
            if decode_size is not None:
                # These options resize the short side to decode_size.
                ffmpeg_output_args += [
                    "-filter:v",
                    f"scale={decode_size}:{decode_size}:force_original_aspect_ratio=increase",
                ]
            if decode_fps is not None:
                # Use the "framerate" or "minterpolate" filters for
                # higher-quality FPS adjustments.
                # https://ffmpeg.org/ffmpeg-filters.html
                ffmpeg_output_args += ["-r", f"{decode_fps}"]
            decode_path = frames_path / video_path.stem
            return_code = decode_video(
                video_path,
                decode_path,
                name_format="%3d",
                image_format="jpg",
                ffmpeg_output_args=ffmpeg_output_args,
            )
            if return_code != 0:
                print(
                    f"Decoding failed for video {video_path.stem}.",
                    file=stderr,
                    flush=True,
                )
                shutil.rmtree(decode_path)

        # Create an empty indicator file.
        print("Decoding complete.", file=stderr, flush=True)
        (base_path / f"decoded_{decode_size}_{decode_fps}").touch()

    @staticmethod
    def download(location, base_split, split, max_tars):
        """
        Performs one-time setup (downloading data).

        :param location: The location of the dataset (see __init__)
        :param base_split: The main split ("train", "test", or "val")
        :param split: The qualified split name (e.g., train_40 for
        max_tars=40)
        :param max_tars: Set a cap on the number of tar files to
        download for this split. Each tar contains about 1k videos.
        """
        base_path = Path(location, split)
        downloads_path = base_path / "downloads"
        downloads_path.mkdir(exist_ok=True)

        # Download the class labels.
        download_file(LABEL_DOWNLOADS[base_split], base_path / "labels.csv")

        # Download the video archive files.
        download_file(VIDEO_DOWNLOADS[base_split], downloads_path / "download_list.txt")
        n = 0
        with open(downloads_path / "download_list.txt", "r") as download_list:
            for url in download_list:
                if (max_tars is not None) and (n >= max_tars):
                    break
                url = url.strip()
                filename = url.split("/")[-1]
                download_file(url, downloads_path / filename)
                n += 1

        # Create an empty indicator file.
        print("Downloads complete.", file=stderr, flush=True)
        (base_path / "downloaded").touch()

    @staticmethod
    def is_decoded(location, split, decode_size, decode_fps):
        """
        Returns true if one-time setup (frame decoding) has been
        completed.

        :param location: The location of the dataset (see __init__)
        :param split: The split name
        :param decode_size: The short-edge length for decoded frames
        :param decode_fps: The fps for decoded frames
        """
        return Path(location, split, f"decoded_{decode_size}_{decode_fps}").is_file()

    @staticmethod
    def is_downloaded(location, split):
        """
        Returns true if one-time setup (data download) has been
        completed.

        :param location: The location of the dataset (see __init__)
        :param split: The split name
        """
        return Path(location, split, "downloaded").is_file()

    @staticmethod
    def is_unpacked(location, split):
        """
        Returns true if one-time setup (tar unpacking) has been
        completed.

        :param location: The location of the dataset (see __init__)
        :param split: The split name
        """
        return Path(location, split, "unpacked").is_file()

    @staticmethod
    def unpack(location, split):
        """
        Performs one-time setup (unpacking tars).

        :param location: The location of the dataset (see __init__)
        :param split: The split name
        """
        base_path = Path(location, split)
        downloads_path = base_path / "downloads"
        videos_path = base_path / "videos"
        videos_path.mkdir(exist_ok=True)

        # Unpack the video archive files.
        with open(downloads_path / "download_list.txt", "r") as download_list:
            for url in download_list:
                url = url.strip()
                filename = url.split("/")[-1]
                filepath = downloads_path / url.split("/")[-1]
                if filepath.exists():
                    print(f"Unpacking {filename}...", file=stderr, flush=True)
                    shutil.unpack_archive(filepath, videos_path)

        # Create an empty indicator file.
        print("Unpacking complete.", file=stderr, flush=True)
        (base_path / "unpacked").touch()

    @staticmethod
    def _get_videos_info(location, split, decode_size, decode_fps):
        videos_info = []
        frames_path = Path(location, split, f"frames_{decode_size}_{decode_fps}")
        with open(Path(location, split, "labels.csv"), "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # Skip header line
            for line in csv_reader:
                video_id = f"{line[1]}_{int(line[2]):06d}_{int(line[3]):06d}"
                label = CLASS_IDS[line[0]]
                video_path = frames_path / video_id
                if not video_path.is_dir():
                    continue
                frames = [path.name for path in video_path.glob("*.jpg")]
                frames.sort()
                videos_info.append(
                    {"video_id": video_id, "label": label, "frames": frames}
                )
        videos_info.sort(key=lambda x: x["video_id"])
        return videos_info
