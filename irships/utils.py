import contextlib
import cv2
import numpy as np
import os
import pandas as pd
import random
import yaml


__version__ = "1.0"


class Namespace(object):

    def __init__(self, *args, **kwargs):
        self.add(*args, **kwargs)

    def __iter__(self):
        return self.__dict__.__iter__()

    def __str__(self, quote=""):
        return "\n".join(self.__print(self.__dict__, quote=quote))

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def items(self):
        return self.__dict__.items()

    def add(self, *args, **kwargs):
        """
        :param args: str denoting path the yaml file to be loaded or dict of items to be added
        :param kwargs: items to be added
        :return: None
        """
        for arg in args:
            if isinstance(arg, str):
                with open(arg, "r") as f:
                    arg = yaml.safe_load(f)
            if isinstance(arg, dict):
                self.__dict__ = {**self.__dict__, **arg}
            else:
                raise TypeError("Expected %s got %s" % (dict, type(arg)))
        self.__dict__ = {**self.__dict__, **self.__convert(kwargs)}

    def save(self, filename, quote=""):
        with open(filename, "w") as f:
            f.write(self.__str__(quote=quote))

    @staticmethod
    def __convert(dictionary):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                dictionary[k] = Namespace(v)
        return dictionary

    def __print(self, dictionary, tab="  ", level=0, list_lim=10, quote=""):
        """
        :param dictionary: [dict] the object to be printed
        :param tab: [str] string to be used as tab
        :param level: [int] the degree of indentation
        :param list_lim: [int] number of items in list before using multiple lines
        :param quote: [str] the character to use as quote
        :return: list of strings outlining the contents of self
        """
        lines = []
        for key, item in dictionary.items():
            item = self.__format_item(item, quote)
            # If item is a dictionary, write the key, then call self.
            if isinstance(item, Namespace):
                lines.append("%s%s:" % ((level * tab), key))
                lines += self.__print(item, level=level+1, quote=quote)
            # If item is nd.array, write the key and array shape
            elif isinstance(item, np.ndarray):
                lines.append("%s%s: %s numpy.ndarray of type %s" % ((level * tab), key, item.shape, item.dtype))
            # If item is a list, if the list contains a list/tuple/Namespace or is long, write in separate lines
            elif isinstance(item, list):
                if any([any([isinstance(i, t) for t in (list, tuple, Namespace)]) for i in item]) or len(item) > list_lim:
                    lines.append("%s%s:" % ((level * tab), key))
                    for i in item:
                        if isinstance(i, Namespace):
                            lines += [
                                "%s  %s" % ((level+1) * tab, line) if j else "%s- %s" % ((level+1) * tab, line)
                                for j, line in enumerate(self.__print(i, quote=quote))
                            ]
                        else:
                            lines.append("%s- %s" % ((level+1) * tab, i))
                else:
                    lines.append("%s%s: %s" % ((level * tab), key, item))
            else:
                lines.append("%s%s: %s" % ((level * tab), key, item))
        return lines

    def __format_item(self, item, quote):
        if isinstance(item, list) or isinstance(item, tuple):
            return [self.__format_item(i, quote) for i in item]
        else:
            return "%s%s%s" % (quote, item, quote) if isinstance(item, str) else item


class Dataloader(object):

    def __init__(self,
                 root,
                 metadata="metadata.csv",
                 sea_intensity_range=(5, 30),
                 sky_intensity_range=(2, 30),
                 clutter_intensity_range=None,
                 clutter_height_range=(0.2, 0.7),
                 clutter_probability=0.5,
                 augment_sea="augment/sea",
                 augment_sky="augment/sky",
                 augment_clutter="augment/clutter",
                 cache_sea_images=True,
                 cache_sky_images=True,
                 cache_clutter_images=True,
                 blur=3
                 ):
        print("IRShips Dataloader, version %s" % __version__)
        print("Initialising dataset...")

        # Main data variables
        self._metadata = None  # Private placeholder for dataframe
        self._metadata_file = None  # private placeholder to store metadata path
        self.indices = None  # Placeholder for indexing dataframe
        self.root = root
        self.key = Namespace("/".join((root, "key.yaml")))
        self.load_metadata(metadata=metadata)

        # Set data augmentation options
        self.sea_intensity_range = sea_intensity_range
        self.cloud_intensity_range = sky_intensity_range
        self.clutter_probability = clutter_probability
        self.clutter_height_range = clutter_height_range
        if clutter_intensity_range is None:
            self.clutter_intensity_range = {
                "ice": (0, 5),
                "structure": (20, 70),
                "landscape": (15, 60)
            }
        else:
            self.clutter_intensity_range = clutter_intensity_range

        # Initialise variables to store real-world images in
        self.sea_images = None
        self.sky_images = None
        self.clutter_images = None

        # Caching options
        self._cache_sea_images = cache_sea_images
        self._cache_sky_images = cache_sky_images
        self._cache_clutter_images = cache_clutter_images

        # Set data augmentation images
        self.blur = blur
        self.augment_sea = augment_sea
        self.augment_sky = augment_sky
        self.augment_clutter = augment_clutter
        print("Done")

    def __getitem__(self, index: int):
        """
        :param index: int the index of an instance to retrive
        :return: tuple contaioning a Namespace of image data and a Namespace of label data
        """
        index = self.indices[index]

        # Make image and label items
        image = Namespace(
            {k: v for k, v in self.metadata.iloc[index].items()},
            dataset="IRShips",
            type_id=self.key["Type"].index(self.metadata.iloc[index].ship_type),
            class_id=self.key["Class"].index(self.metadata.iloc[index].ship_class),
            image=cv2.imread("/".join((self.root, "images", self.metadata.iloc[index].filename)), cv2.IMREAD_UNCHANGED),
            is_input=True,
        )
        label = Namespace(
            {k: v for k, v in self.metadata.iloc[index].items()},
            dataset="IRShips",
            type_id=self.key["Type"].index(self.metadata.iloc[index].ship_type),
            class_id=self.key["Class"].index(self.metadata.iloc[index].ship_class),
            image=cv2.imread("/".join((self.root, "labels", self.metadata.iloc[index].labelname)), cv2.IMREAD_UNCHANGED),
            is_input=False,
        )

        # Find sea level of horizontal images
        image["sea_level"] = get_sea_level(image.image) if image.pitch == 0 else 0

        # Add sea, clouds and clutter
        if self.augment_sea:
            image = self.__add_sea(image, label)
        if image.pitch == 0:
            if self.augment_sky:
                image = self.__add_sky(image, label)
            if self.augment_clutter:
                image = self.__add_clutter(image, label)
                
        # Apply Gaussian blur
        if isinstance(self.blur, int) and self.blur > 0:
            image.image = cv2.GaussianBlur(image.image, (self.blur, self.blur), 0)

        # Return the data, is_loaded and is_transformed
        return image, label

    def __len__(self):
        return len(self.indices)

    def load_metadata(self, metadata=None):
        print("Loading metadata")
        if metadata is not None:
            self._metadata_file = metadata
        self.metadata = pd.read_csv("/".join((self.root, self._metadata_file)))

    def shuffle(self):
        random.shuffle(self.indices)

    def sort(self):
        self.indices.sort()

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, new_value):
        """
        :param new_value: pd.dataframe of IRShips metadata
        :return: None
        """
        self._metadata = new_value
        self.indices = list(range(len(self._metadata)))
        print("Indexed %i instances" % len(self.indices))

    @property
    def augment_sea(self):
        return self._augment_sea

    @augment_sea.setter
    def augment_sea(self, new_value):
        """
        :param new_value: string : path to sea augmentation images
        :return: None
        """
        self.sea_images = None if new_value is None else self.__load_sea_images(new_value)
        self._augment_sea = new_value

    @property
    def augment_sky(self):
        return self._augment_sky

    @augment_sky.setter
    def augment_sky(self, new_value):
        self.sky_images = None if new_value is None else self.__load_sky_images(new_value)
        self._augment_sky = new_value

    @property
    def augment_clutter(self):
        return self._augment_clutter

    @augment_clutter.setter
    def augment_clutter(self, new_value):
        self.clutter_images = None if new_value is None else self.__load_clutter_images(new_value)
        self._augment_clutter = new_value

    @property
    def cache_sea_images(self):
        return self._cache_sea_images

    @cache_sea_images.setter
    def cache_sea_images(self, new_value):
        self._cache_sea_images = new_value  # Set variable
        if self.sea_images is not None:  # If augment sea images have already been loaded
            print("Caching sea images...") if self.cache_sea_images else print("Indexing sea images...")
            self.augment_sea = self._augment_sea  # Reload them
            print("Done")

    @property
    def cache_sky_images(self):
        return self._cache_sky_images

    @cache_sky_images.setter
    def cache_sky_images(self, new_value):
        self._cache_sky_images = new_value  # Set variable
        if self.sky_images is not None:  # If augment sky images have already been loaded
            print("Caching sky images...") if self.cache_sea_images else print("Indexing sky images...")
            self.augment_sky = self._augment_sky  # Reload them
            print("Done")

    @property
    def cache_clutter_images(self):
        return self._cache_clutter_images

    @cache_clutter_images.setter
    def cache_clutter_images(self, new_value):
        self._cache_clutter_images = new_value  # Set variable
        if self.clutter_images is not None:  # If augment clutter images have already been loaded
            print("Caching sea images...") if self.cache_clutter_images else print("Indexing sea images...")
            self.augment_clutter = self._augment_clutter  # Reload them
            print("Done")

    @staticmethod
    def __load_images(directory, cache_images, load_flag=-1):
        images = []
        for image_name in os.listdir(directory):
            image_path = "%s/%s" % (directory, image_name)
            if cache_images:
                image = cv2.imread(image_path, load_flag)
                if image is not None:
                    images.append(image)
                else:
                    raise FileNotFoundError("Unable to load image: %s" % image_path)
            else:
                if os.path.isfile(image_path):
                    images.append(image_path)
                else:
                    raise FileNotFoundError("Unable to find image: %s" % image_path)
        return images

    def __load_sea_images(self, directory):
        h = self.__load_images("%s/%s/horizontal" % (self.root, directory), self.cache_sea_images, cv2.IMREAD_GRAYSCALE)
        print("\tLoaded %i images for sea-state augmentation (horizontal perspective)" % len(h))
        e = self.__load_images("%s/%s/elevated" % (self.root, directory), self.cache_sea_images, cv2.IMREAD_GRAYSCALE)
        print("\tLoaded %i images for sea-state augmentation (elevated perspective)" % len(e))
        return {"horizontal": h, "elevated": e}

    def __load_sky_images(self, directory):
        sky_images = self.__load_images("%s/%s" % (self.root, directory), self.cache_sky_images, cv2.IMREAD_GRAYSCALE)
        print("\tLoaded %i images for sky-state augmentation" % len(sky_images))
        return sky_images

    def __load_clutter_images(self, directory):
        ice = self.__load_images("%s/%s/ice" % (self.root, directory), self.cache_clutter_images, cv2.IMREAD_GRAYSCALE)
        ice = [(i, "ice") for i in ice]
        print("\tLoaded %i images for clutter augmentation (ice)" % len(ice))
        struct = self.__load_images("%s/%s/structure" % (self.root, directory), self.cache_clutter_images, cv2.IMREAD_GRAYSCALE)
        struct = [(i, "structure") for i in struct]
        print("\tLoaded %i images for clutter augmentation (structure)" % len(struct))
        land = self.__load_images("%s/%s/landscape" % (self.root, directory), self.cache_clutter_images, cv2.IMREAD_GRAYSCALE)
        land = [(i, "landscape") for i in land]
        print("\tLoaded %i images for clutter augmentation (landscapes)" % len(land))
        return ice + land + struct

    def __add_sea(self, image, label):
        # Randomly select intensity range
        with contextlib.suppress(TypeError):
            intensity_range = np.random.randint(*self.sea_intensity_range)

        # Select/load a sea image
        sea_image = random.choice(self.sea_images["horizontal" if image.pitch == 0 else "elevated"])
        if not self.cache_sea_images:
            sea_image = cv2.imread(sea_image, cv2.IMREAD_GRAYSCALE)

        mask = label.image[image.sea_level:, ...].copy()  # Mask of foreground pixels to preserve
        image_lower = image.image[image.sea_level:, ...]  # Bottom half of image (to augment)

        # Randomly scale, crop and scale pixel intensities
        sea_image = process_sea(
            sea_image,
            shape=image_lower.shape[0:2],
            average_intensity=int(np.mean(image_lower)),
            intensity_range=intensity_range
        )

        image.image[image.sea_level:, ...] = superimpose(image_lower, sea_image, mask)
        return image

    def __add_sky(self, image, label):
        if image.sea_level:
            with contextlib.suppress(TypeError):
                intensity_range = np.random.randint(*self.cloud_intensity_range)

            mask = label.image[0: image.sea_level, ...].copy()  # Mask of foreground pixels to preserve
            image_top = image.image[0:image.sea_level, ...]  # Top half of image
            background_temp = int(np.mean(image_top))  # Average background temperature

            # Select/load a sky image
            sky_image = random.choice(self.sky_images)
            if not self.cache_sky_images:
                sky_image = cv2.imread(sky_image, cv2.IMREAD_GRAYSCALE)

            # Random scale, crop and scale pixel intensities
            clouds = process_sky(
                sky=sky_image,
                shape=image_top.shape[0:2],
                lower=background_temp,
                upper=background_temp + intensity_range
            )

            image.image[0:image.sea_level, ...] = superimpose(image_top, clouds, mask)
        return image

    def __add_clutter(self, image, label):
        if np.random.random() < self.clutter_probability and image.sea_level:

            # Set intensity
            intensity = self.clutter_intensity_range
            for key, item in self.clutter_intensity_range.items():
                with contextlib.suppress(TypeError):
                    intensity[key] = np.random.randint(*item)

            # Randomly select clutter height variable
            with contextlib.suppress(TypeError):
                clutter_height = get_random_value(*self.clutter_height_range)
            clutter_height = max(10, int(clutter_height * image.sea_level))

            orig_image = image.image.copy()

            clutter_image, clutter_type = random.choice(self.clutter_images)
            if not self.cache_clutter_images:
                clutter_image = cv2.imread(clutter_image, cv2.IMREAD_GRAYSCALE)

            intensity = self.clutter_intensity_range[clutter_type]
            with contextlib.suppress(TypeError):
                intensity = np.random.randint(*intensity)

            clutter = process_clutter(clutter_image, image.image.shape, intensity, clutter_height)
            y0 = image.sea_level - clutter.shape[0]
            # Make ship and clutter masks
            ship_mask = label.image[y0: image.sea_level:, ...].copy()
            clutter_mask = clutter.copy()
            clutter_mask[clutter > 0] = 1
            clutter_mask = 1 - clutter_mask
            # Superimpose clutter and ship
            image.image[y0: image.sea_level, ...] = superimpose(image.image[y0: image.sea_level, ...], clutter, clutter_mask)
            image.image[y0: image.sea_level, ...] = superimpose(orig_image[y0: image.sea_level, ...], clutter, ship_mask)
        return image


def get_sea_level(image, label=None, threshold=5):
    image = image[:, 0]  # Take first column
    image -= image[0]  # Subtract by the value of the first number
    image[image > 0] = 1  # Make values 1 if greater than 0
    i = np.flip(np.arange(image.shape[0]))  # Make a fsimilar array of descending values
    sea_level = np.argmax(image * i) + 1  # Take argmax of image * descending array
    if sea_level < threshold and label is not None:
        sea_level = find_extents(label, normalize=False)[3]
    return int(sea_level) if int(sea_level) > 1 else 0


def get_random_value(lower, upper):
    return lower + np.random.random() * (upper - lower)


def choose_image(directory):
    try:
        filenames = os.listdir(directory)
    except FileNotFoundError:
        print("choose_image : No such directory : %s" % directory)
    # Load the selected clouds file
    filename = random.choice(filenames)
    return cv2.imread("/".join((directory, filename)), cv2.IMREAD_GRAYSCALE), filename


def superimpose(image_1, image_2, mask):
    assert image_1.shape == image_2.shape
    mask[mask > 0] = 1
    image_2[mask > 0] = 0           # Set everything in the second image that the mask covers to black
    image_2 += image_1 * mask       # Add the contents of the first image to the space left by the mask
    return image_2


def find_extents(image, normalize=False):
    image[image < 50] = 0
    image[image > 0] = 255
    x = np.argwhere(image.any(axis=0))[:, 0]
    y = np.argwhere(image.any(axis=1))[:, 0]
    if normalize:
        x = x / image.shape[1]
        y = y / image.shape[0]
    return np.array((x[0], y[0], x[-1], y[-1]))


def normalize(image, lower=0, upper=255):
    image = image.astype(np.float32)
    image -= np.min(image)
    image *= (upper - lower) / max(np.max(image), 1)
    image += lower
    return image


def get_scale(orig_shape, min_shape):
    orig_shape = np.array(orig_shape)
    min_shape = np.array(min_shape)
    scale = max(min_shape / orig_shape)
    if scale < 1:
        scale = scale + np.random.random() * (1 - scale)
    return scale


def process_sea(sea, shape, average_intensity=20, intensity_range=30):
    # Random geometric transforms
    scale = get_scale(orig_shape=sea.shape[0:2], min_shape=shape)
    sea = cv2.resize(sea, (0, 0), fx=scale, fy=scale)  # Random scale
    if np.random.random() > 0.5:
        sea = cv2.flip(sea, 1)  # Random horizontal flip
    sea = random_crop(sea, shape=shape)  # Random crop

    # Scale pixel intensities
    sea = sea.astype(np.float32)
    lower = int(average_intensity - intensity_range / 2)
    upper = int(average_intensity + intensity_range / 2)
    sea = normalize(sea, lower=lower, upper=upper)
    sea[sea < 0] = 0
    sea[sea > 255] = 255
    return sea.astype(np.uint8)


def process_sky(sky, shape, lower=0, upper=255):
    # Random geometric transforms
    scale = get_scale(orig_shape=sky.shape[0:2], min_shape=shape)
    sky = cv2.resize(sky, (0, 0), fx=scale, fy=scale)  # Random resize
    if np.random.random() > 0.5:
        sky = cv2.flip(sky, 1)  # Random flip
    sky = random_crop(sky, shape=shape)  # Random crop

    # Scale pixel intensities
    sky = sky.astype(np.float32)
    sky[sky < 10] = 0
    sky[sky < 10] = 0
    sky = normalize(sky, lower, upper)
    sky[sky < 0] = 0
    sky[sky > 255] = 255
    return sky.astype(np.uint8)


def process_clutter(clutter, shape, intensity, clutter_height):
    im_h, im_w = shape[0:2]
    scale = clutter_height / clutter.shape[0]
    wrap = np.any(clutter[:, 0]) or np.any(clutter[:, -1])
    if np.random.random() > 0.5:
        clutter = cv2.flip(clutter, 1)                              # Random flip lr
    clutter = cv2.resize(clutter, (0, 0), fx=scale, fy=scale)       # Resize
    if clutter.shape[1] < im_w:
        clutter = stretch_image(clutter, im_w, wrap=wrap)
    else:
        clutter = random_crop(clutter, shape=(clutter.shape[0], im_w))
    clutter = normalize(clutter, 0, intensity)
    return clutter


def random_crop(image, scale=None, shape=None, resize=False, interpolation=cv2.INTER_LINEAR):
    """
    Author: Samuel Westlake
    One of scale or shape should be given.
    :param image: np.array: input image
    :param scale: int or tuple: scale of the cropped box relative to the original image shape
    :param shape: tuple: shape of the crop (height, width)
    :param resize: bool: whether to resize the crop to the original image size or not
    :param interpolation: interpolation method for resizing (from opencv) (linear by default)
    :return: np.array: cropped image
    """
    if scale is not None and shape is not None:
        print("%s.random_crop : both a scale and a shape were given : only one can be prescribed")
    elif scale is None and shape is None:
        print("%s.random_crop : neither a scale nor a shape were given :  one (and only one) of these should be given")
    im_h, im_w = image.shape[0:2]
    if scale is not None:
        with contextlib.suppress(TypeError):
            scale = get_random_value(*scale)
        if not 0 < scale <= 1:
            print("%s.random_crop : 0 < scale <=1 must be true : got scale = %f" % (__name__, scale))
        crop_w = int(im_w * scale)          # Get width of the crop
        crop_h = int(im_h * scale)          # Get height of the crop
    elif shape is not None:
        crop_h, crop_w = shape
    x0 = 0 if im_w - crop_w == 0 else np.random.randint(0, im_w - crop_w)
    y0 = 0 if im_h - crop_h == 0 else np.random.randint(0, im_h - crop_h)
    x1 = x0 + crop_w
    y1 = y0 + crop_h
    coords = (x0, y0, x1, y1)
    image = crop(image, coords, resize, interpolation)
    return image


def crop(image, coords, resize=False, interpolation=cv2.INTER_LINEAR):
    """
    :param image: [np.array] input image
    :param coords: [tuple] coordinates of crop: (x0, y0, x1, y1)
    :param resize: [bool] whether or not to resize the crop to the original image size
    :param interpolation: [int] interpolation method for resizing (from opencv) (linear by default)
    :return: [np.array] cropped image
    """
    x0, y0, x1, y1 = coords
    h, w = image.shape[0:2]
    image = image[y0:y1, x0:x1, ...]
    if resize:
        image = cv2.resize(image, (w, h), interpolation=interpolation)
    return image


def stretch_image(image, width, wrap=False):
    """
    :param image: [np.array] input image
    :param width: [int]
    :param wrap: [bool]
    :return: [np.array] stretched image
    """
    h, w = image.shape[0:2]
    new_image = np.zeros((h, width), np.uint8)
    if wrap:
        for i in range(int(np.ceil(width / w))):
            x0 = i * w
            x1 = min(x0 + w, width)
            new_image[:, x0:x1] = image[:, :x1-x0] if i % 2 else image[:, ::-1][:, :x1-x0]
    else:
        x0 = np.random.randint(0, width - w)
        x1 = x0 + w
        new_image[:, x0:x1] = image
    return new_image

