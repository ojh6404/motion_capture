#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import requests
import logging

URLS = {
    "mano": "https://download.is.tue.mpg.de/download.php?domain=mano&resume=1&sfile=mano_v1_2.zip",
    "smpl": "https://download.is.tue.mpg.de/download.php?domain=smplify&resume=1&sfile=mpips_smplify_public_v2.zip",
    "smplx": "https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip",
}


def download_data(target: str, out_folder: str):
    """
    Download files from a list of URLs.

    Args:
        target (str): Target model type, either "smplx", "smpl" or "mano".
        out_folder (str): Path to folder to store downloaded files.
    """

    assert target in URLS, f"Invalid target {target}. Must be one of {list(URLS.keys())}."

    url = URLS[target]

    try:
        username = os.environ[f"{target}_USERNAME".upper()]
        password = os.environ[f"{target}_PASSWORD".upper()]
        password_fake = "*" * len(password)
    except KeyError as e:
        logging.error(f"Environment variable {e} not set. Please set it and try again.")
        return

    logging.info(f"Username: {username}")
    logging.info(f"Password: {password_fake}")

    post_data = {"username": username, "password": password}

    logging.info(f"Start downloading from {url}")
    response = requests.post(
        url,
        data=post_data,
        stream=True,
        verify=False,
        allow_redirects=True,
    )

    if response.status_code == 401:
        logging.error(f"Authentication failed for URLs in {url_file}. Username/password correct?")
        return

    # Get the filename from the URL
    filename = url.split("/")[-1]
    if "models_smplx_v1_1" in url:  # smplx
        filename = "models_smplx_v1_1.zip"
    elif "mpips_smplify_public_v2" in url:  # smpl
        filename = "mpips_smplify_public_v2.zip"
    elif "mano_v1_2" in url:  # mano
        filename = "mano_v1_2.zip"
    else:
        raise ValueError(f"Unknown URL format: {url}")

    # Write the contents of the response to a file
    out_p = os.path.join(out_folder, filename)
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    with open(out_p, "wb") as f:
        f.write(response.content)

    logging.info(f"{filename} downloaded to {out_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MANO/SMPL-X data from URLs")
    parser.add_argument(
        "--target",
        type=str,
        choices=["smplx", "smpl", "mano"],
        required=True,
        help="Target model type to download.",
    )
    parser.add_argument(
        "--out-folder",
        type=str,
        required=True,
        help="Path to folder to store downloaded files.",
    )
    args = parser.parse_args()

    download_data(target=args.target, out_folder=args.out_folder)
