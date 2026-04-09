# TDT4265 - Snowpole Detection

## Installation

```shell
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```


## Linting + formatting


```shell
# format + lint your code manually (before committing)
ruff format .
ruff check . --fix

# type-check manually
mypy .

# run all pre-commit hooks on every file right now (useful first run)
pre-commit run --all-files
```


## Image Augmentation

We are to apply some image augmentaiton to our dataset as to get more training examples
and make it more robust. For this, we have two files which can be ran:

### Step 1: Horizontal Augmentation
This script splits the images horizontally, making the poles a bigger percentage of the image.
```shell
python ds_split_images.py
```

### Step 2: Random augmentations
Here, we use the [AlbumentationsX](https://github.com/albumentations-team/AlbumentationsX) to apply different
amounts of augmentations for our images.


```shell
python ds_apply_augmentations.py
```

## Using WanDB

WanDB can be used to get nice visualizaiton.
This can be installed using:

```shell
wandb login
yolo settings wandb=True
```