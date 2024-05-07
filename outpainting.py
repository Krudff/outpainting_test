import platform

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from zipfile import ZipFile
import openvino as ov

# Fetch `notebook_utils` module
import requests

r = requests.get(
    url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
)

open("notebook_utils.py", "w").write(r.text)
import os
import requests
import zipfile

# A directory where the model will be downloaded.
base_model_dir = "model"
# The name of the model from Open Model Zoo.
model_name = "gmcnn-places2-tf"

model_path = Path(f"{base_model_dir}/public/{model_name}/frozen_model.pb")
if not model_path.exists():
    model_url = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/gmcnn-places2-tf/{model_name}.zip"

    # Download the model using requests library
    response = requests.get(model_url)

    # Check for successful download
    if response.status_code == 200:
        # Create the base directory if it doesn't exist
        os.makedirs(base_model_dir, exist_ok=True)

        # Extract the filename from the response headers
        content_disposition = response.headers.get("content-disposition")
        if content_disposition:
            filename = content_disposition.split("filename=")[1].strip()
        else:
            filename = model_name + ".zip"

        # Save the downloaded file
        with open(os.path.join(base_model_dir, filename), "wb") as file:
            file.write(response.content)

        # Extract the model from the downloaded zip file
        with zipfile.ZipFile(os.path.join(base_model_dir, filename), "r") as zip_ref:
            zip_ref.extractall(path=Path(base_model_dir, "public"))

        print(f"Downloaded and extracted {model_name} model")
    else:
        print(f"Error downloading model: {response.status_code}")
else:
    print("Already downloaded")

model_dir = Path(base_model_dir, "public", "ir")
ir_path = Path(f"{model_dir}/frozen_model.xml")

# Run model conversion API to convert model to OpenVINO IR FP32 format, if the IR file does not exist.
if not ir_path.exists():
    ov_model = ov.convert_model(model_path, input=[[1, 512, 680, 3], [1, 512, 680, 1]])
    ov.save_model(ov_model, str(ir_path))
else:
    print(f"{ir_path} already exists.")

core = ov.Core()

# Read the model.xml and weights file
model = core.read_model(model=ir_path)

import ipywidgets as widgets

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value="AUTO",
    description="Device:",
    disabled=False,
)

device

# Load the model on to the device
compiled_model = core.compile_model(model=model, device_name=device.value)
# Store the input and output nodes
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

N, H, W, C = input_layer.shape
print(N)
print(H)
print(W)
print(C)

def create_mask(image_width, image_height, size_x=80, size_y=80, number=1):
    """
    Create a square mask of defined size on the right by 40px

    :param: image_width: width of the image
    :param: image_height: height of the image
    :param: size: size in pixels of one side
    :returns:
            mask: grayscale float32 mask of size shaped [image_height, image_width, 1]
    """

    mask = np.zeros((image_height, image_width, 1), dtype=np.float32)
    mask[:, -size_x:] = 1 # Fill the 40px in the right with 1 (white)
    return mask




def modify_image(input_image, iterations=0):
    """
    Modifies an image by removing 10px from the left and adding 10px with black on the right.
    Recursively applies the modification based on the iterations variable.

    Args:
        input_image: The input image (NumPy array).
        iterations: Number of times to recursively apply the modification (default: 0).

    Returns:
        The modified image (NumPy array).
    """

    height, width, channels = input_image.shape

    # Base case: stop recursion when iterations reach 0
    if iterations == 0:
        return input_image

    # Create a black patch with the same dimensions as the result image
    modified_image = np.zeros((height, width, channels), dtype=np.uint8)

    # Copy the rightmost portion of the result image (excluding the last 10px)
    # and the first 10px from the left to fill the black patch
    modified_image[:, :-5] = input_image[:, 5:, :]
    modified_image[:, -5:] = input_image[:, :5, :]

    # Generate a square mask 10px from the right
    mask = create_mask(image_width=W, image_height=H, size_x=5, size_y=40, number=15)
    # This mask will be laid over the input image as noise.

    masked_image = (modified_image * (1 - mask) + 255 * mask).astype(np.uint8)
    masked_image = masked_image[None, ...]
    mask = mask[None, ...]

    result = compiled_model([ov.Tensor(masked_image.astype(np.float32)), ov.Tensor(mask.astype(np.float32))])[output_layer]
    result = result.squeeze().astype(np.uint8)

    # Recursively apply the modification again if iterations are remaining
    return modify_image(result, iterations - 1)




img_path = Path("data/pic4.jpg")


# Read the image.
image = cv2.imread(str(img_path))
# Resize the image to meet network expected input sizes.
resized_image = cv2.resize(src=image, dsize=(W, H), interpolation=cv2.INTER_AREA)


out = modify_image(resized_image,90)

cv2.imwrite("data/pic4_restored.png", out)
print("it is done")