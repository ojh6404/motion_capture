from typing import Tuple, Any
import torch
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

from motion_capture.utils import (
    MANO_JOINTS_CONNECTION,
    MANO_ROOT,
    SMPL_ROOT,
    HAMER_CHECKPOINT_PATH,
    HAMER_CONFIG_PATH,
    WILOR_CHECKPOINT_PATH,
    WILOR_CONFIG_PATH,
    HAMBA_CHECKPOINT_PATH,
    HAMBA_CONFIG_PATH,
    HMR2_CHECKPOINT_PATH,
    HMR2_CONFIG_PATH,
)

# optical frame to world frame
OPTICAL_TO_WORLD = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])


def axes_to_quaternion(x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    """
    Convert a set of axes to a quaternion.

    Args:
        x_axis (np.ndarray): The x-axis.
        y_axis (np.ndarray): The y-axis.
        z_axis (np.ndarray): The z-axis.

    Returns:
        np.ndarray: A 4-element unit quaternion. (w, x, y, z)
    """

    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    r = R.from_matrix(rotation_matrix)
    quaternion = r.as_quat()
    quaternion = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]  # w, x, y, z
    return quaternion


def rotation_matrix_to_quaternion(R):
    """Convert a rotation matrix to a quaternion.
    Args:
        R (np.ndarray): A 3x3 rotation matrix.
    Returns:
        np.ndarray: A 4-element unit quaternion.
    """
    q = np.empty((4,), dtype=np.float32)
    q[0] = np.sqrt(np.maximum(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    q[1] = np.sqrt(np.maximum(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    q[2] = np.sqrt(np.maximum(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    q[3] = np.sqrt(np.maximum(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    q[1] *= np.sign(q[1] * (R[2, 1] - R[1, 2]))
    q[2] *= np.sign(q[2] * (R[0, 2] - R[2, 0]))
    q[3] *= np.sign(q[3] * (R[1, 0] - R[0, 1]))
    return q


def draw_axis(img, origin, axis, color, scale=20):
    point = origin + scale * axis
    img = cv2.line(img, (int(origin[0]), int(origin[1])), (int(point[0]), int(point[1])), color, 2)
    return img


def load_hamer(
    img_size: Tuple[int, int],
    focal_length: float,
    checkpoint_path: str = HAMER_CHECKPOINT_PATH,
    cfg_path: str = HAMER_CONFIG_PATH,
) -> Tuple[Any, dict]:
    """
    Load HaMeR model from checkpoint.

    Args:
        img_size (Tuple[int, int]): Image size (width, height).
        focal_length (float): Focal length.
        checkpoint_path (str): Path to the model checkpoint.
        cfg_path (str): Path to the model config file.

    Returns:
        Tuple[HAMER, dict]: Loaded model and its configuration.
    """

    from hamer.configs import get_config
    from hamer.models import HAMER

    model_cfg = get_config(cfg_path)
    model_cfg.defrost()
    model_cfg.MANO.MODEL_PATH = MANO_ROOT
    model_cfg.MANO.MEAN_PARAMS = MANO_ROOT + "/mano_mean_params.npz"
    model_cfg.EXTRA.FOCAL_LENGTH = int(focal_length * model_cfg.MODEL.IMAGE_SIZE / max(img_size))
    model_cfg.freeze()

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == "vit") and ("BBOX_SHAPE" not in model_cfg.MODEL):
        model_cfg.defrost()
        assert (
            model_cfg.MODEL.IMAGE_SIZE == 256
        ), f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()

    # Update config to be compatible with demo
    if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
        model_cfg.freeze()

    model = HAMER.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    return model, model_cfg


def load_wilor(
    img_size: Tuple[int, int],
    focal_length: float,
    checkpoint_path: str = WILOR_CHECKPOINT_PATH,
    cfg_path: str = WILOR_CONFIG_PATH,
) -> Tuple[Any, dict]:
    """
    Load WiLoR model from checkpoint.

    Args:
        img_size (Tuple[int, int]): Image size (width, height).
        focal_length (float): Focal length.
        checkpoint_path (str): Path to the model checkpoint.
        cfg_path (str): Path to the model config file.

    Returns:
        Tuple[WiLoR, dict]: Loaded model and its configuration.
    """

    from wilor.configs import get_config
    from wilor.models import WiLoR

    model_cfg = get_config(cfg_path, update_cachedir=True)
    model_cfg.defrost()
    model_cfg.MANO.DATA_DIR = MANO_ROOT
    model_cfg.MANO.MODEL_PATH = MANO_ROOT
    model_cfg.MANO.MEAN_PARAMS = MANO_ROOT + "/mano_mean_params.npz"
    model_cfg.EXTRA.FOCAL_LENGTH = int(focal_length * model_cfg.MODEL.IMAGE_SIZE / max(img_size))
    model_cfg.freeze()

    # Override some config values, to crop bbox correctly
    if ("vit" in model_cfg.MODEL.BACKBONE.TYPE) and ("BBOX_SHAPE" not in model_cfg.MODEL):
        model_cfg.defrost()
        assert (
            model_cfg.MODEL.IMAGE_SIZE == 256
        ), f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()

    # Update config to be compatible with demo
    if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
        model_cfg.freeze()

    model = WiLoR.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    return model, model_cfg


def load_hamba(
    img_size: Tuple[int, int],
    focal_length: float,
    checkpoint_path: str = HAMBA_CHECKPOINT_PATH,
    cfg_path: str = HAMBA_CONFIG_PATH,
) -> Tuple[Any, dict]:
    """
    Load Hamba model from checkpoint.

    Args:
        img_size (Tuple[int, int]): Image size (width, height).
        focal_length (float): Focal length.
        checkpoint_path (str): Path to the model checkpoint.
        cfg_path (str): Path to the model config file.

    Returns:
        Tuple[WiLoR, dict]: Loaded model and its configuration.
    """
    from hamba.configs import get_config
    from hamba.models import HAMBA

    model_cfg = get_config(cfg_path, update_cachedir=True)
    model_cfg.defrost()
    model_cfg.MANO.DATA_DIR = MANO_ROOT
    model_cfg.MANO.MODEL_PATH = MANO_ROOT
    model_cfg.MANO.MEAN_PARAMS = MANO_ROOT + "/mano_mean_params.npz"
    model_cfg.EXTRA.FOCAL_LENGTH = int(focal_length * model_cfg.MODEL.IMAGE_SIZE / max(img_size))
    model_cfg.freeze()

    # Override some config values, to crop bbox correctly
    if ("vit" in model_cfg.MODEL.BACKBONE.TYPE) and ("BBOX_SHAPE" not in model_cfg.MODEL):
        model_cfg.defrost()
        assert (
            model_cfg.MODEL.IMAGE_SIZE == 256
        ), f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()
    elif model_cfg.MODEL.BACKBONE.TYPE == "vmamba" or model_cfg.MODEL.BACKBONE.TYPE == "fastvit_ma36":
        model_cfg.defrost()
        assert (
            model_cfg.MODEL.IMAGE_SIZE == 224
        ), f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 224 for vmamba backbone"
        model_cfg.MODEL.BBOX_SHAPE = [224, 224]
        model_cfg.freeze()

    # Update config to be compatible with demo
    if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
        if "PRETRAINED_WEIGHTS_INIT_REGRESSION" in model_cfg.MODEL.keys():
            model_cfg.MODEL.pop("PRETRAINED_WEIGHTS_INIT_REGRESSION")
        model_cfg.freeze()

    model = HAMBA.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    return model, model_cfg


def load_hmr2(
    img_size: Tuple[int, int],
    focal_length: float,
    checkpoint_path: str = HMR2_CHECKPOINT_PATH,
    cfg_path: str = HMR2_CONFIG_PATH,
) -> Tuple[Any, dict]:
    """
    Load HMR2 model from checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        cfg_path (str): Path to the model config file.
        img_size (Tuple[int, int]): Image size (width, height).
        focal_length (float): Focal length.

    Returns:
        Tuple[HMR2, dict]: Loaded model and its configuration.
    """

    from hmr2.configs import get_config
    from hmr2.models.hmr2 import HMR2

    model_cfg = get_config(cfg_path, update_cachedir=True)
    model_cfg.defrost()
    model_cfg.SMPL.MODEL_PATH = SMPL_ROOT
    model_cfg.SMPL.JOINT_REGRESSOR_EXTRA = SMPL_ROOT + "/SMPL_to_J19.pkl"
    model_cfg.SMPL.MEAN_PARAMS = SMPL_ROOT + "/smpl_mean_params.npz"
    model_cfg.EXTRA.FOCAL_LENGTH = int(focal_length * model_cfg.MODEL.IMAGE_SIZE / max(img_size))
    model_cfg.freeze()

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == "vit") and ("BBOX_SHAPE" not in model_cfg.MODEL):
        model_cfg.defrost()
        assert (
            model_cfg.MODEL.IMAGE_SIZE == 256
        ), f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()

    def check_smpl_exists():
        import os

        candidates = [
            SMPL_ROOT + "/SMPL_NEUTRAL.pkl",
            SMPL_ROOT + "/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
        ]
        candidates_exist = [os.path.exists(c) for c in candidates]
        if not any(candidates_exist):
            raise FileNotFoundError(
                f"SMPL model not found. Please download it from https://smplify.is.tue.mpg.de/ and place it at {candidates[1]}"
            )

        def convert_pkl(old_pkl, new_pkl):
            """
            Convert a Python 2 pickle to Python 3
            """
            import dill
            import pickle

            # Convert Python 2 "ObjectType" to Python 3 object
            dill._dill._reverse_typemap["ObjectType"] = object

            # Open the pickle using latin1 encoding
            with open(old_pkl, "rb") as f:
                loaded = pickle.load(f, encoding="latin1")

            # Re-save as Python 3 pickle
            with open(new_pkl, "wb") as outfile:
                pickle.dump(loaded, outfile)

        # Code edxpects SMPL model at CACHE_DIR_4DHUMANS/data/smpl/SMPL_NEUTRAL.pkl. Copy there if needed
        if (not candidates_exist[0]) and candidates_exist[1]:
            convert_pkl(candidates[1], candidates[0])

        return True

    # Ensure SMPL model exists
    check_smpl_exists()

    model = HMR2.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    return model, model_cfg


def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.0):
    # Convert cam_bbox to full image
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2.0, img_h / 2.0
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam


def recursive_to(x, target: torch.device):
    if isinstance(x, dict):
        return {k: recursive_to(v, target) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(target)
    elif isinstance(x, list):
        return [recursive_to(i, target) for i in x]
    else:
        return x


def draw_hand_keypoints(img: np.ndarray, keypoints: np.ndarray):
    hsv_colors = np.zeros((21, 3), dtype=np.uint8)
    hsv_colors[:, 0] = np.linspace(0, 180, 21, endpoint=True)  # Hue
    hsv_colors[:, 1] = 255  # Saturation
    hsv_colors[:, 2] = 255  # Value
    colors = cv2.cvtColor(hsv_colors.reshape(1, 21, 3), cv2.COLOR_HSV2BGR).reshape(21, 3)

    # draw connections and keypoints
    for color, connection in zip(colors, MANO_JOINTS_CONNECTION):
        cv2.line(
            img,
            tuple(keypoints[connection[0]].astype(int)),
            tuple(keypoints[connection[1]].astype(int)),
            color.tolist(),
            2,
        )
    return img
