import json

import imageio.v3 as iio
import numpy as np
import pandas as pd
import zarr
from skimage.transform import resize


def read_tiff(input_path):
    """
    Read a TIFF image and:
    - If smaller than 2048x2048: pad to 2048x2048
    - If larger than 2048x2048: crop to 2048x2048
    Returns a 2048x2048 numpy array
    """
    # Read the TIFF file using imageio
    img_array = iio.imread(input_path)

    height, width = img_array.shape[:2]
    # target_size = 2048
    target_size = 3000

    if height > target_size or width > target_size:
        # Need to crop
        start_y = (height - target_size) // 2
        start_x = (width - target_size) // 2
        return img_array[
            start_y : start_y + target_size, start_x : start_x + target_size
        ]
    else:
        # Need to pad
        padded_array = np.zeros(
            (target_size, target_size, *img_array.shape[2:]), dtype=img_array.dtype
        )
        pad_y = (target_size - height) // 2
        pad_x = (target_size - width) // 2
        padded_array[pad_y : pad_y + height, pad_x : pad_x + width] = img_array
        return padded_array


def fast_downscale(image, scaled_h, scaled_w):
    """
    Fast image downscaling using striding/sampling approach.

    Args:
        image: Input image array (H, W) or (C, H, W)
        scaled_h: Target height
        scaled_w: Target width

    Returns:
        Downscaled image array
    """
    if image.ndim == 2:
        h, w = image.shape
        # Calculate stride sizes
        stride_h = h // scaled_h
        stride_w = w // scaled_w

        # Use striding to sample pixels
        return image[::stride_h, ::stride_w][:scaled_h, :scaled_w]

    else:  # Handle multi-channel images
        c, h, w = image.shape
        stride_h = h // scaled_h
        stride_w = w // scaled_w

        # Initialize output array
        scaled_image = np.empty((c, scaled_h, scaled_w), dtype=np.uint16)

        # Apply striding to each channel
        for channel in range(c):
            scaled_image[channel] = image[channel, ::stride_h, ::stride_w][
                :scaled_h, :scaled_w
            ]

        return scaled_image


class FOVMapper:
    def __init__(self, csv_file):
        """
        Initialize the FOV mapper with a CSV file containing the coordinate data.

        Parameters:
        csv_file (str): Path to the CSV file with columns: fov, x (mm), y (mm)
        """
        # Read the CSV file
        self.df = pd.read_csv(csv_file)

        # Get unique x and y coordinates, sorted
        self.unique_x = sorted(self.df["x (mm)"].unique())
        self.unique_y = sorted(self.df["y (mm)"].unique())

        # Create lookup dictionaries for quick conversion
        self.x_to_i = {x: i for i, x in enumerate(self.unique_x)}
        self.y_to_j = {y: j for j, y in enumerate(self.unique_y)}

        # Store grid dimensions
        self.grid_width = len(self.unique_x)
        self.grid_height = len(self.unique_y)

        # Create FOV lookup for validation
        self.valid_fovs = set(self.df["fov"].unique())

    def fov_to_ij(self, fov):
        """
        Convert FOV number to (i, j) grid coordinates.

        Parameters:
        fov (int): FOV number

        Returns:
        tuple: (i, j) coordinates where i is column index and j is row index
        """
        if fov not in self.valid_fovs:
            raise ValueError(f"Invalid FOV number: {fov}")

        # Get the actual x, y coordinates for this FOV
        row = self.df[self.df["fov"] == fov].iloc[0]
        x = row["x (mm)"]
        y = row["y (mm)"]

        # Convert to grid coordinates
        i = self.x_to_i[x]
        j = self.y_to_j[y]

        return (i, j)

    def ij_to_fov(self, i, j):
        """
        Convert (i, j) grid coordinates to FOV number.

        Parameters:
        i (int): Column index
        j (int): Row index

        Returns:
        int: FOV number
        """
        if not (0 <= i < self.grid_width and 0 <= j < self.grid_height):
            raise ValueError(
                f"i must be between 0 and {self.grid_width - 1}, j between 0 and {self.grid_height - 1}"
            )

        # Get physical coordinates
        x = self.unique_x[i]
        y = self.unique_y[j]

        # Find FOV at these coordinates
        matching_rows = self.df[(self.df["x (mm)"] == x) & (self.df["y (mm)"] == y)]

        if len(matching_rows) == 0:
            return None

        return matching_rows.iloc[0]["fov"]

    def get_xy_from_fov(self, fov):
        """
        Get physical (x, y) coordinates for a given FOV number.

        Parameters:
        fov (int): FOV number

        Returns:
        tuple: (x, y) physical coordinates in mm
        """
        if fov not in self.valid_fovs:
            raise ValueError(f"Invalid FOV number: {fov}")

        row = self.df[self.df["fov"] == fov].iloc[0]
        return (row["x (mm)"], row["y (mm)"])

    def get_grid_size(self):
        return self.grid_width, self.grid_height


def create_multiscale_zarr(acquisition_path, output_path, image_loader_func=None):
    """
    Create a multiscale (4 levels) OME-Zarr array from a Nx x Ny scan of 4-channel 2048x2048 images.
    Final level 0 image will be (T=1, C=4, Z=1, Y=20480, X=20480) for nx=ny=10.

    Parameters:
    -----------
    output_path : str
        Path where to save the Zarr array
    image_loader_func : callable
        Function that takes (x, y) coordinates and returns a 4-channel 2048x2048 image
    """
    # Original image dimensions
    channels = 5
    # img_height = 2048
    # img_width = 2048
    img_height = 3000
    img_width = 3000

    if image_loader_func is None:
        coordinate_path = acquisition_path + "/0/coordinates.csv"
        image_path = acquisition_path + "/0/"
        mapper = FOVMapper(coordinate_path)
        nx, ny = mapper.get_grid_size()
    else:
        nx = 8
        ny = 8

    # Calculate full dimensions
    full_height = ny * img_height  # 20480 for ny=10
    full_width = nx * img_width  # 20480 for nx=10

    # Create root group
    root = zarr.open(output_path, mode="w")

    # Create multiscales metadata
    multiscales = [
        {
            "version": "0.4",
            "name": "pyramid",
            "datasets": [
                {
                    "path": str(level),
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1, 1, 1, scale, scale]}
                    ],
                }
                for level, scale in enumerate([1, 2, 4, 8])
            ],
            "axes": ["t", "c", "z", "y", "x"],
            "type": "gaussian",
        }
    ]

    # Create OME metadata
    ome = {
        "id": "urn:uuid:",  # Would normally add a UUID here
        "name": "ome-zarr",
        "axes": [
            {"name": "t", "type": "time"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ],
        "channels": [{"label": f"Channel {i}"} for i in range(channels)],
        "rdefs": {"model": "color"},
    }

    # Save metadata
    root.attrs["multiscales"] = multiscales
    root.attrs["omero"] = ome

    # Create arrays for each scale level with TCZYX ordering
    arrays = []
    for level, scale in enumerate([1, 2, 4, 8]):
        arr = root.create_dataset(
            str(level),
            shape=(1, channels, 1, full_height // scale, full_width // scale),  # TCZYX
            chunks=(
                1,
                channels,
                1,
                img_height // scale,
                img_width // scale,
            ),  # Chunk size matches source image size
            dtype=np.uint16,
        )
        arrays.append(arr)

    # Process images one by one
    for y in range(ny):
        for x in range(nx):
            # Load single image
            if image_loader_func is None:
                fov = mapper.ij_to_fov(x, y)
                if fov is not None:
                    images = []
                    images.append(
                        read_tiff(
                            image_path
                            + "A1_"
                            + str(fov)
                            + "_0_Fluorescence_405_nm_Ex.tiff"
                        )
                    )
                    images.append(
                        read_tiff(
                            image_path
                            + "A1_"
                            + str(fov)
                            + "_0_Fluorescence_405_nm_Ex.tiff"
                        )
                    )
                    images.append(
                        read_tiff(
                            image_path
                            + "A1_"
                            + str(fov)
                            + "_0_Fluorescence_405_nm_Ex.tiff"
                        )
                    )
                    image = np.stack(images, axis=0)
                    # print(fov)
                else:
                    image = np.zeros((channels, img_height, img_width), dtype=np.uint16)
            else:
                image = image_loader_func(x, y)  # Shape: (channels, height, width)

            # Calculate the slice positions
            y_start = y * img_height
            x_start = x * img_width

            # Write to each scale level
            for level, scale in enumerate([1, 2, 4, 8]):
                if scale == 1:
                    # Write directly to the appropriate position in level 0
                    arrays[level][
                        0,
                        :,
                        0,
                        y_start : y_start + img_height,
                        x_start : x_start + img_width,
                    ] = image
                else:
                    # Calculate scaled positions
                    y_scaled_start = y_start // scale
                    x_scaled_start = x_start // scale
                    scaled_h = img_height // scale
                    scaled_w = img_width // scale

                    # Create scaled image
                    scaled_image = np.zeros(
                        (channels, scaled_h, scaled_w), dtype=np.uint16
                    )
                    for c in range(channels):
                        # scaled = resize(
                        #     image[c],
                        #     (scaled_h, scaled_w),
                        #     anti_aliasing=False,
                        #     preserve_range=True
                        # )
                        scaled = fast_downscale(image[c], scaled_h, scaled_w)
                        scaled_image[c] = np.clip(scaled, 0, 65535).astype(np.uint16)

                    # Write to the appropriate position
                    arrays[level][
                        0,
                        :,
                        0,
                        y_scaled_start : y_scaled_start + scaled_h,
                        x_scaled_start : x_scaled_start + scaled_w,
                    ] = scaled_image

            del image


# Example usage:
def dummy_image_loader(x, y):
    """Example function to load images. Replace with your actual image loading logic."""
    return np.random.randint(0, 65536, (3, 2048, 2048), dtype=np.uint16)


import skimage.io


def imread_from_8x8(x, y):
    idx = y * 8 + x

    channels = [405, 488, 561, 638, 730]

    full_item = np.zeros((5, 3000, 3000), dtype=np.uint16)

    for ci, c in enumerate(channels):
        full_item[ci, :, :] = skimage.io.imread(
            f"/Users/colin/cephla-images/8x8_2025-01-13_14-46-59.824096/0/R0_{idx}_0_Fluorescence_{c}_nm_Ex.tiff"
        )
    return full_item


if __name__ == "__main__":
    import time

    t0 = time.time()
    # # for testing
    create_multiscale_zarr(
        "/Users/colin/cephla-images/8x8_2025-01-13_14-46-59.824096",
        "output.ome.zarr",
        imread_from_8x8,
    )
    # create_multiscale_zarr("/Volumes/Extreme SSD/Test/10x_2025-01-20_02-47-59.815347",
    #    "/Volumes/Extreme SSD/Test/output.zarr")
    print(time.time() - t0)
