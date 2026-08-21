import os
import gzip
import struct
import urllib.request
from PIL import Image

# Where to save the dataset
OUTPUT_DIR = "mnist"

# Working MNIST mirror
BASE_URL = "https://raw.githubusercontent.com/fgnt/mnist/master"

IMAGE_URL = f"{BASE_URL}/train-images-idx3-ubyte.gz"
LABEL_URL = f"{BASE_URL}/train-labels-idx1-ubyte.gz"

IMAGE_FILE = "train-images-idx3-ubyte.gz"
LABEL_FILE = "train-labels-idx1-ubyte.gz"


def download(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print("Done!")
    else:
        print(f"{filename} already exists.")


# Download files
download(IMAGE_URL, IMAGE_FILE)
download(LABEL_URL, LABEL_FILE)


# Create folders 0-9
for digit in range(10):
    os.makedirs(os.path.join(OUTPUT_DIR, str(digit)), exist_ok=True)


# Read images and labels
with gzip.open(IMAGE_FILE, "rb") as f:
    # MNIST image header:
    # magic number, number of images, rows, columns
    magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))

    print(f"Images: {num_images}")
    print(f"Size: {rows}x{cols}")

    images = f.read()


with gzip.open(LABEL_FILE, "rb") as f:
    # MNIST label header:
    # magic number, number of labels
    magic, num_labels = struct.unpack(">II", f.read(8))

    labels = f.read()

print(f"Labels: {num_labels}")


# Convert every image to PNG
image_size = rows * cols

for i in range(num_images):

    # Get image pixels
    start = i * image_size
    end = start + image_size

    pixels = images[start:end]

    # Get corresponding label
    label = labels[i]

    # Convert bytes to image
    image = Image.frombytes(
        "L",
        (cols, rows),
        pixels
    )

    # Save image
    filename = os.path.join(
        OUTPUT_DIR,
        str(label),
        f"{i:05d}.png"
    )

    image.save(filename)

    # Progress
    if (i + 1) % 1000 == 0:
        print(f"Converted {i + 1}/{num_images}")


print("\nFinished!")
print(f"Images saved to: {OUTPUT_DIR}/")