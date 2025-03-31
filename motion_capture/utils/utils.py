from typing import Tuple, Any
from pathlib import Path
import torch
import numpy as np
import cv2
import pyrender
import trimesh
from scipy.spatial.transform import Rotation as R

from motion_capture.utils import (
    THIRD_PARTY_ROOT,
    MANO_JOINTS_CONNECTION,
    HAND_COLOR,
    MANO_ROOT,
    SMPL_ROOT,
    WILOR_CHECKPOINT_PATH,
    WILOR_CONFIG_PATH,
    HAMER_CHECKPOINT_PATH,
    HAMER_CONFIG_PATH,
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


class Renderer(object):
    def __init__(self, faces, cfg, width=640, height=480):
        super(Renderer, self).__init__()
        self.width = width
        self.height = height

        self.focal_length = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * max(width, height)
        self.camera_center = [self.width / 2.0, self.height / 2.0]
        self.camera_pose = np.eye(4)
        self.camera = pyrender.IntrinsicsCamera(
            fx=self.focal_length, fy=self.focal_length, cx=self.camera_center[0], cy=self.camera_center[1]
        )

        self.lights = self.create_raymond_lights()

        self.faces = faces
        self.faces_left = self.faces[:, [0, 2, 1]]

        self.renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

    def vertices_to_trimesh(
        self,
        vertices,
        camera_translation,
        mesh_base_color=HAND_COLOR,
        rot_axis=[1, 0, 0],
        rot_angle=0,
        is_right=1,
    ):
        vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
        if is_right:
            mesh = trimesh.Trimesh(
                vertices.copy() + camera_translation, self.faces.copy(), vertex_colors=vertex_colors
            )
        else:
            mesh = trimesh.Trimesh(
                vertices.copy() + camera_translation, self.faces_left.copy(), vertex_colors=vertex_colors
            )

        rot = trimesh.transformations.rotation_matrix(np.radians(rot_angle), rot_axis)
        mesh.apply_transform(rot)

        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        return mesh

    def render_rgba_multiple(
        self,
        vertices,
        cam_t,
        rot_axis=[1, 0, 0],
        rot_angle=0,
        is_right=None,
    ):
        # Create pyrender scene
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        # Add meshes to the scene
        if is_right is None:
            is_right = [1 for _ in range(len(vertices))]
        mesh_list = [
            pyrender.Mesh.from_trimesh(
                self.vertices_to_trimesh(
                    vvv, ttt.copy(), rot_axis=rot_axis, rot_angle=rot_angle, is_right=sss
                )
            )
            for vvv, ttt, sss in zip(vertices, cam_t, is_right)
        ]
        for i, mesh in enumerate(mesh_list):
            scene.add(mesh, f"mesh_{i}")

        # Create camera node and add it to pyRender scene
        scene.add(self.camera, pose=self.camera_pose)

        # Add lights to the scene
        for node in self.lights:
            scene.add_node(node)

        rgba, depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

        return rgba, depth

    def create_raymond_lights(self):
        """
        Return raymond light nodes for the scene.
        """
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        nodes = []

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(
                pyrender.Node(light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0), matrix=matrix)
            )

        return nodes


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
