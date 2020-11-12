# IRShips

Supporting code for the IRShips dataset.

**Version 1.0**

## Dataloader

### Arguments

```python
utils.Dataloader( 
    root: str, 
    metadata: str = "metadata.csv",
    sea_intensity_range: tuple = (5, 30), 
    sky_intensity_range: tuple = (2, 40),
    clutter_intensity_range: dict = {"ice": (0, 5), "structure": (20, 70), "landscape": (15, 60)},
    clutter_height_range: tuple = (0.2, 0.7),
    clutter_probability: float = 0.5,
    augment_sea: str = "augment/sea",
    augment_sky: str = "augment/sky",
    augment_clutter: str = "augment/clutter",
    cache_sea_images: bool = True,
    cache_sky_images: bool = True,
    cache_clutter_images: bool = True,
    blur=3
)
```

#### root 

[str]
        
Path to the directory of IRShips data.

#### sea_intensity_range 

[tuple of ints]

The minimum and maximum permissible pixel intensities relating to sea-state augmentation.

#### sky_intensity_range

[tuple of ints]

The minimum and maximum permissible pixel intensities relating to sea-state augmentation.

#### clutter_intensity_range

The minimum and maximum permissible pixel intensities relating to each type of background clutter augmentation.

The given dictionary must contain a key for each *type* of background clutter. 
(The default types of background clutter are: ice, landscape and structure.)
        
#### clutter_height_range

[dict]

The minimum and maximum height of added background clutter as a fraction of the height between the horizon and the top of an image (down to a floor of 10 pixels).
        
#### clutter_probability

[float]

The probability that clutter will be added to a given image (provided that the sea-sky horizon line is visible in the image).
        
#### augment_sea

[bool]

Path to the directory of real-world sea images for data augmentation (if None, sea-state augmentation will not be included).
Note that the root directory will be prepended to the given file path.

#### augment_sky

[bool]

Path to the directory of real-world sky images for data augmentation (if None, sky-state augmentation will not be included).
Note that the root directory will be prepended to the given file path.

#### augment_clutter

[bool]

Path to the directory of real-world background clutter images for data augmentation (if None, background clutter augmentation will not be included).
Note that the root directory will be prepended to the given file path.

#### cache_sea_images

[bool] 

Whether or not real-world sea images should be stored in memory (for faster dataloading).

#### cache_sky_images

[bool] 

Whether or not real-world sky images should be stored in memory (for faster dataloading). 

#### cache_clutter_images

[bool] 

Whether or not real-world background clutter images should be stored in memory (for faster dataloading).

#### blur

[int] 

Kernel size to use for Gaussian blur, use 0 to omit blur. 

### Methods

#### shuffle()

```python
dataset.shuffle()
```

Shuffles the dataset indexes.

#### sort()

```python
dataset.sort()
```

Re-orders the dataset indexes.

#### load_metadata(metadata=None)

```python
dataset.load_metadata(metadata=None)
```

Re-load the IRShips metadata file.

If metadata is None, the previously given metadata file path will be used, otherwise the newly given filepath will be used and remembered for next time. 

**NB:** 

- The root directory will be prepended to the given file path.      

- Resetting the metadata dataframe with `dataset.load_metadata()` will cause indexes to be reset in an ordered fashion.
You may subsequently want to use the `shuffle` method. 

        
### Use Cases

#### Iterating

```python
from irships import utils

# Initialise IRShips dataloader
dataset = utils.Dataloader("path/to/irships/directory")

# Shuffle dataset
dataset.shuffle()

# Iterate
for inp, lab in dataset:
    # Do something
```

#### Filtering

The metadata file can be accessed directly and filtered by column if required, like so: 

```python
# It is possible to filter the dataset by metadata attributes, like so: 
df = dataset.metadata  # Get dataframe
df = df[df["range"] >= 5000]  # Example: remove all examples where range < 5000m
df = df[df["pitch"] == 0]  # Example: remove all examples where pitch != 0
dataset.metadata = df  # Set dataframe
```

**NB:** setting the metadata dataframe with `dataset.metadata = ...` will cause indexes to be reset in an ordered fashion.
You may subsequently want to use the `shuffle` method. 

## Licence

Code in this repository is published under the [MIT licence](LICENSE).

## Credit

If this dataset and supporting helps your research, please give recognition by citing the accompanying paper. 
Example BibTeX entry below:

```
@inproceedings{westlake2020deep,
  title={Deep learning for automatic target recognition with real and synthetic infrared maritime imagery},
  author={Westlake, Samuel T and Volonakis, Timothy N and Jackman, James and James, David B and Sherriff, Andy},
  booktitle={Artificial Intelligence and Machine Learning in Defense Applications II},
  volume={11543},
  pages={1154309},
  year={2020},
  organization={International Society for Optics and Photonics}
}
```
