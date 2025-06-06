#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union
import torch
import numpy as np
import cv2
from skimage.filters import gaussian
import supervision as sv
from pyvirtualdisplay import Display
from scipy.spatial.transform import Rotation as R

# WiLoR
from wilor.datasets.utils import convert_cvimg_to_tensor, expand_to_aspect_ratio, generate_image_patch_cv2

# 4DHuman
from hmr2.datasets.vitdet_dataset import ViTDetDataset as HMR2ViTDetDataset

# utils and constants
from motion_capture.detector import DetectionResult
from motion_capture.utils.utils import (
    load_hamer,
    load_wilor,
    load_hamba,
    load_hmr2,
    recursive_to,
    cam_crop_to_full,
    draw_hand_keypoints,
    rotation_matrix_to_quaternion,
)
from motion_capture.utils import (
    MANO_KEYPOINT_NAMES,
    SPIN_KEYPOINT_NAMES,
)
from motion_capture.utils.renderer import PyrenderRenderer, Pytorch3DRenderer


BOX_ANNOTATOR = sv.BoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()


@dataclass
class MocapResult:
    detection: DetectionResult  # Detection result
    position: np.ndarray  # Wrist position
    orientation: np.ndarray  # MANO global orientation
    beta: np.ndarray  # MANO beta
    theta: np.ndarray  # MANO theta
    keypoint_names: List[str]  # keypoint names
    keypoints_2d: np.ndarray  # 2D keypoints
    keypoints: np.ndarray  # 3D keypoints
    vertices: Optional[np.ndarray] = None  # 3D vertices (optional, only for visualization)
    faces: Optional[np.ndarray] = None  # Faces (optional, only for visualization)


class MocapModelFactory:
    @staticmethod
    def from_config(model: str, model_config: dict):
        if model == "hamer":
            return HamerModel(**model_config)
        elif model == "wilor":
            return WiLoRModel(**model_config)
        elif model == "hamba":
            return HambaModel(**model_config)
        elif model == "4d-human":
            return HMR2Model(**model_config)
        else:
            raise ValueError(f"Invalid mocap model: {model_config['model']}")


class MocapModelBase(ABC):
    @abstractmethod
    def predict(
        self,
        img: np.ndarray,
        detections: Optional[List[DetectionResult]] = None,
        boxes: Optional[Union[List[List[float]], np.ndarray]] = None,
        vis_img: Optional[np.ndarray] = None,
    ) -> Tuple[List[MocapResult], np.ndarray]:
        pass


class HamerModel(MocapModelBase):
    def __init__(
        self,
        K: np.ndarray = np.array(
            [[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1]]
        ),  # camera intrinsic matrix
        rescale_factor: float = 2.0,  # rescale factor for hand detection
        img_size: tuple = (640, 480),  # (width, height)
        visualize: bool = True,  # whether to visualize the result
        renderer: str = "pyrender",  # renderer type, 'pyrender' or 'pytorch3d'
        device: str = "cuda:0",  # device
    ):
        self.focal_length = K[0, 0]  # focal length from camera intrinsic matrix
        self.rescale_factor = rescale_factor
        self.img_size = img_size
        self.visualize = visualize
        self.device = device

        # init model
        self.mocap, self.cfg = load_hamer(
            img_size=self.img_size,
            focal_length=self.focal_length,
        )
        self.mocap.to(self.device)
        self.mocap.eval()

        self.img_mean = 255.0 * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.img_std = 255.0 * np.array(self.cfg.MODEL.IMAGE_STD)
        self.bbox_shape = self.cfg.MODEL.get("BBOX_SHAPE", None)
        self.patch_width = self.patch_height = self.cfg.MODEL.IMAGE_SIZE

        if self.visualize:
            if renderer == "pyrender":
                self.renderer = PyrenderRenderer(
                    K=K,
                    width=self.img_size[0],
                    height=self.img_size[1],
                )
            elif renderer == "pytorch3d":
                self.renderer = Pytorch3DRenderer(
                    K=K,
                    width=self.img_size[0],
                    height=self.img_size[1],
                    device=self.device,
                    from_opencv=True,
                )
            else:
                raise ValueError(f"Invalid renderer type: {renderer}. Choose 'pyrender' or 'pytorch3d'.")

    def preprocess_input(self, img: np.ndarray, detections: List[DetectionResult]) -> Dict[str, torch.Tensor]:
        """
        Preprocess the input for the model

        Args:
            img (np.ndarray): Input image, BGR format (H, W, C)
            detections (List[DetectionResult]): List of detection results

        Returns:
            input_batch Dict[str, torch.Tensor]: Input batch for the model
                img (torch.Tensor): Image tensor (B, C, H, W)
                box_center (torch.Tensor): Box center (B, 2)
                box_size (torch.Tensor): Box size (B,)
                img_size (torch.Tensor): Image size (B, 2)
                right (torch.Tensor): Right hand flag (B,)
        """

        boxes = np.array([detection.rect for detection in detections]).astype(np.float32)  # x1, y1, x2, y2
        is_right = np.array([1 if detection.label == "right_hand" else 0 for detection in detections]).astype(
            np.float32
        )
        centers = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        scales = self.rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0

        # preprocess input for each detection
        processed_batch = dict()
        for center, scale, right in zip(centers, scales, is_right):
            center_x = center[0]
            center_y = center[1]

            bbox_size = expand_to_aspect_ratio(
                input_shape=scale * 200, target_aspect_ratio=self.bbox_shape
            ).max()
            flip = right == 0

            # 3. generate image patch
            downsampling_factor = (bbox_size * 1.0) / self.patch_width
            downsampling_factor = downsampling_factor / 2.0
            if downsampling_factor > 1.1:
                img = gaussian(img, sigma=(downsampling_factor - 1) / 2, channel_axis=2, preserve_range=True)

            img_patch_cv, _ = generate_image_patch_cv2(
                img=img,
                c_x=center_x,
                c_y=center_y,
                bb_width=bbox_size,
                bb_height=bbox_size,
                patch_width=self.patch_width,
                patch_height=self.patch_height,
                do_flip=flip,
                scale=1.0,
                rot=0,
                border_mode=cv2.BORDER_CONSTANT,
            )
            img_patch_cv = img_patch_cv[:, :, ::-1]
            img_patch = convert_cvimg_to_tensor(img_patch_cv)

            # apply normalization
            for n_c in range(min(img.shape[2], 3)):
                img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.img_mean[n_c]) / self.img_std[n_c]

            item = {
                "img": img_patch,
            }
            item["box_center"] = center.astype(np.float32)
            item["box_size"] = bbox_size.astype(np.float32)
            item["img_size"] = 1.0 * np.array([img.shape[1], img.shape[0]]).astype(np.float32)
            item["right"] = right.astype(np.float32)

            # processed batch is like batch['img'] = torch.Tensor (B, C, H, W)
            for key, value in item.items():
                if key not in processed_batch:
                    processed_batch[key] = []
                processed_batch[key].append(value)

        # Convertto torch.Tensor
        for key, value in processed_batch.items():
            val = np.stack(value, axis=0)
            processed_batch[key] = torch.from_numpy(val).float().to(self.device)

        return processed_batch

    @torch.no_grad()
    def predict(
        self,
        img: np.ndarray,
        detections: Optional[List[DetectionResult]] = None,
        boxes: Optional[Union[List[List[float]], np.ndarray]] = None,
        is_right: Optional[Union[List[int], np.ndarray]] = None,
        vis_img: Optional[np.ndarray] = None,
    ) -> Tuple[List[MocapResult], np.ndarray]:
        """
        Predict the hand pose from the input image

        Args:
            img (np.ndarray): Input image, BGR format (H, W, C)
            detections (List[DetectionResult]): List of detection results
            boxes (np.ndarray): Bounding boxes for the hands, shape (B, 4) in format [x1, y1, x2, y2]
            is_right (np.ndarray): Array indicating if the hand is right (1) or left (0), shape (B,)
            vis_img (np.ndarray): Optional image for visualization

        Returns:
            mocap_results List[MocapResult]: List of MocapResult
            vis_img np.ndarray: Visualization image
        """
        mocap_results = []
        if detections is None:  # if there are detections
            assert boxes is not None and is_right is not None, (
                "If detections are None, boxes and is_right must be provided"
            )
            detections = [
                DetectionResult(
                    label="right_hand" if right else "left_hand",
                    rect=hand_bbox,  # (x1, y1, x2, y2)
                    score=0.8,
                )
                for right, hand_bbox in zip(is_right, boxes)
            ]

        if detections:  # if there are detections
            # Predict
            batch = self.preprocess_input(img=img, detections=detections)
            out = self.mocap(
                batch
            )  # ['pred_cam', 'pred_mano_params', 'pred_cam_t', 'focal_length', 'pred_keypoints_2d', 'pred_keypoints_3d', 'pred_vertices']

            # MANO params
            mano_params = out["pred_mano_params"]
            global_orients = mano_params["global_orient"].squeeze(1).detach().cpu().numpy()  # (B, 3, 3)
            thetas = mano_params["hand_pose"].squeeze(1).detach().cpu().numpy()  # (B, 15, 3, 3)
            betas = mano_params["betas"].detach().cpu().numpy()  # (B, 10)

            # Camera params
            pred_cam = out["pred_cam"]  # (B, 3)
            pred_cam[:, 1] *= 2 * batch["right"] - 1  # flip y-axis for left hand
            box_center = batch["box_center"].float()  # (B, 2)
            box_size = batch["box_size"].float()  # (B,)
            img_size = batch["img_size"].float()  # (B, 2)
            scaled_focal_length = self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = (
                cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)
                .detach()
                .cpu()
                .numpy()
            )  # (B, 3)

            # 2D keypoints
            boxes = np.array(
                [detection.rect for detection in detections], dtype=np.float32
            )  # (B, 4) x1, y1, x2, y2
            is_right = batch["right"].detach().cpu().numpy().astype(np.int32)  # (B,)
            box_center = batch["box_center"].detach().cpu().numpy()  # (B, 2)
            box_size = batch["box_size"].detach().cpu().numpy()  # (B,)
            pred_keypoints_2d = out["pred_keypoints_2d"].detach().cpu().numpy()  # (B, 21, 2)
            pred_keypoints_2d[:, :, 0] = (2 * is_right[:, None] - 1) * pred_keypoints_2d[
                :, :, 0
            ]  # flip x-axis for left hand
            pred_keypoints_2d = pred_keypoints_2d * box_size[:, None, None] + box_center[:, None, :]

            # 3D keypoints
            pred_keypoints_3d = out["pred_keypoints_3d"].detach().cpu().numpy()  # (B, 21, 3)
            pred_keypoints_3d[:, :, 0] = (2 * is_right[:, None] - 1) * pred_keypoints_3d[:, :, 0]
            pred_keypoints_3d += pred_cam_t_full[:, None, :]  # add translation wrt camera

            # 3D vertices wrtcamera
            pred_verts = out["pred_vertices"].detach().cpu().numpy()  # (B, 778, 3)
            pred_verts[:, :, 0] = (2 * is_right[:, None] - 1) * pred_verts[
                :, :, 0
            ]  # flip x-axis for left hand
            pred_verts += pred_cam_t_full[:, None, :]  # add translation wrt camera

            for i, hand_id in enumerate(is_right):  # for each hand
                assert detections[i].label == "right_hand" if hand_id == 1 else "left_hand", (
                    "Hand ID and hand detection mismatch"
                )
                orientation = global_orients[i]
                if hand_id == 0:  # left hand
                    orientation[1::3] *= -1
                    orientation[2::3] *= -1

                assert len(MANO_KEYPOINT_NAMES) == len(pred_keypoints_3d[i]) == len(pred_keypoints_2d[i]), (
                    "Keypoint mismatch"
                )
                mocap_result = MocapResult(
                    detection=detections[i],
                    position=pred_keypoints_3d[i][0],
                    orientation=orientation,
                    beta=betas[i],
                    theta=thetas[i],
                    keypoint_names=MANO_KEYPOINT_NAMES,
                    keypoints=pred_keypoints_3d[i],
                    keypoints_2d=pred_keypoints_2d[i],
                    vertices=pred_verts[i],  # 3D vertices
                )
                mocap_results.append(mocap_result)

        if self.visualize:
            vis_img = self.visualize_mocap(
                img=img,
                mocap_results=mocap_results,
                vis_img=vis_img,
            )
        else:
            vis_img = img.copy() if vis_img is None else vis_img

        return mocap_results, vis_img

    def visualize_mocap(
        self,
        img: np.ndarray,
        mocap_results: List[MocapResult],
        vis_img: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Visualize the motion capture results on the input image

        Args:
            img (np.ndarray): Input image, BGR format (H, W, C)
            mocap_results (List[MocapResult]): List of MocapResult
            vis_img (np.ndarray): Optional image for visualization

        Returns:
            vis_img np.ndarray: Visualization image
        """

        if not mocap_results:
            return img.copy() if vis_img is None else vis_img

        boxes = np.array([mocap_result.detection.rect for mocap_result in mocap_results], dtype=np.float32)
        is_right = np.array(
            [1 if mocap_result.detection.label == "right_hand" else 0 for mocap_result in mocap_results],
            dtype=np.int32,
        )
        keypoints_2d = np.array([mocap_result.keypoints_2d for mocap_result in mocap_results])
        vertices = np.array([mocap_result.vertices for mocap_result in mocap_results])

        if vis_img is None:
            # Draw BBOX and LABEL with annotator
            vis_img = img.copy()
            vis_detections = sv.Detections(
                xyxy=boxes,
                class_id=is_right,
            )
            vis_img = BOX_ANNOTATOR.annotate(scene=vis_img, detections=vis_detections)
            vis_img = LABEL_ANNOTATOR.annotate(
                scene=vis_img,
                detections=vis_detections,
                labels=["right_hand" if class_id == 1 else "left_hand" for class_id in is_right],
            )

        verts = []
        faces = []
        for i, vert in enumerate(vertices):
            face = self.mocap.mano.faces if is_right[i] else self.mocap.mano.faces[:, [0, 2, 1]]
            verts.append(vert)
            faces.append(face)
        rgba, _, _ = self.renderer.render(verts=verts, faces=faces)
        rgb = rgba[..., :3].astype(np.float32)
        alpha = rgba[..., 3].astype(np.float32) / 255.0
        vis_img = (alpha[..., None] * rgb + (1 - alpha[..., None]) * vis_img).astype(np.uint8)

        for keypoint_2d in keypoints_2d:
            vis_img = draw_hand_keypoints(vis_img, keypoint_2d)

        return vis_img


class WiLoRModel(MocapModelBase):
    def __init__(
        self,
        K: np.ndarray = np.array(
            [[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1]]
        ),  # camera intrinsic matrix
        rescale_factor: float = 2.0,  # rescale factor for hand detection
        img_size: tuple = (640, 480),  # (width, height)
        visualize: bool = True,  # whether to visualize the result
        renderer: str = "pyrender",  # renderer type, 'pyrender' or 'pytorch3d'
        device: str = "cuda:0",  # device
    ):
        self.focal_length = K[0, 0]  # focal length from camera intrinsic matrix
        self.rescale_factor = rescale_factor
        self.img_size = img_size
        self.visualize = visualize
        self.device = device

        # init model
        self.mocap, self.cfg = load_wilor(
            img_size=self.img_size,
            focal_length=self.focal_length,
        )
        self.mocap.to(self.device)
        self.mocap.eval()

        self.img_mean = 255.0 * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.img_std = 255.0 * np.array(self.cfg.MODEL.IMAGE_STD)
        self.bbox_shape = self.cfg.MODEL.get("BBOX_SHAPE", None)
        self.patch_width = self.patch_height = self.cfg.MODEL.IMAGE_SIZE

        if self.visualize:
            if renderer == "pyrender":
                self.renderer = PyrenderRenderer(
                    K=K,
                    width=self.img_size[0],
                    height=self.img_size[1],
                )
            elif renderer == "pytorch3d":
                self.renderer = Pytorch3DRenderer(
                    K=K,
                    width=self.img_size[0],
                    height=self.img_size[1],
                    device=self.device,
                    from_opencv=True,
                )
            else:
                raise ValueError(f"Invalid renderer type: {renderer}. Choose 'pyrender' or 'pytorch3d'.")

    def preprocess_input(self, img: np.ndarray, detections: List[DetectionResult]) -> Dict[str, torch.Tensor]:
        """
        Preprocess the input for the model

        Args:
            img (np.ndarray): Input image, BGR format (H, W, C)
            detections (List[DetectionResult]): List of detection results

        Returns:
            input_batch Dict[str, torch.Tensor]: Input batch for the model
                img (torch.Tensor): Image tensor (B, C, H, W)
                box_center (torch.Tensor): Box center (B, 2)
                box_size (torch.Tensor): Box size (B,)
                img_size (torch.Tensor): Image size (B, 2)
                right (torch.Tensor): Right hand flag (B,)
        """

        boxes = np.array([detection.rect for detection in detections]).astype(np.float32)  # x1, y1, x2, y2
        is_right = np.array([1 if detection.label == "right_hand" else 0 for detection in detections]).astype(
            np.float32
        )
        centers = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        scales = self.rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0

        # preprocess input for each detection
        processed_batch = dict()
        for center, scale, right in zip(centers, scales, is_right):
            center_x = center[0]
            center_y = center[1]

            bbox_size = expand_to_aspect_ratio(
                input_shape=scale * 200, target_aspect_ratio=self.bbox_shape
            ).max()
            flip = right == 0

            # 3. generate image patch
            downsampling_factor = (bbox_size * 1.0) / self.patch_width
            downsampling_factor = downsampling_factor / 2.0
            if downsampling_factor > 1.1:
                img = gaussian(img, sigma=(downsampling_factor - 1) / 2, channel_axis=2, preserve_range=True)

            img_patch_cv, _ = generate_image_patch_cv2(
                img=img,
                c_x=center_x,
                c_y=center_y,
                bb_width=bbox_size,
                bb_height=bbox_size,
                patch_width=self.patch_width,
                patch_height=self.patch_height,
                do_flip=flip,
                scale=1.0,
                rot=0,
                border_mode=cv2.BORDER_CONSTANT,
            )
            img_patch_cv = img_patch_cv[:, :, ::-1]
            img_patch = convert_cvimg_to_tensor(img_patch_cv)

            # apply normalization
            for n_c in range(min(img.shape[2], 3)):
                img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.img_mean[n_c]) / self.img_std[n_c]

            item = {
                "img": img_patch,
            }
            item["box_center"] = center.astype(np.float32)
            item["box_size"] = bbox_size.astype(np.float32)
            item["img_size"] = 1.0 * np.array([img.shape[1], img.shape[0]]).astype(np.float32)
            item["right"] = right.astype(np.float32)

            # processed batch is like batch['img'] = torch.Tensor (B, C, H, W)
            for key, value in item.items():
                if key not in processed_batch:
                    processed_batch[key] = []
                processed_batch[key].append(value)

        # Convertto torch.Tensor
        for key, value in processed_batch.items():
            val = np.stack(value, axis=0)
            processed_batch[key] = torch.from_numpy(val).float().to(self.device)

        return processed_batch

    @torch.no_grad()
    def predict(
        self,
        img: np.ndarray,
        detections: Optional[List[DetectionResult]] = None,
        boxes: Optional[Union[List[List[float]], np.ndarray]] = None,
        is_right: Optional[Union[List[int], np.ndarray]] = None,
        vis_img: Optional[np.ndarray] = None,
    ) -> Tuple[List[MocapResult], np.ndarray]:
        """
        Predict the hand pose from the input image

        Args:
            img (np.ndarray): Input image, BGR format (H, W, C)
            detections (List[DetectionResult]): List of detection results
            boxes (np.ndarray): Bounding boxes for the hands, shape (B, 4) in format [x1, y1, x2, y2]
            is_right (np.ndarray): Array indicating if the hand is right (1) or left (0), shape (B,)
            vis_img (np.ndarray): Optional image for visualization

        Returns:
            mocap_results List[MocapResult]: List of MocapResult
            vis_img np.ndarray: Visualization image
        """
        mocap_results = []
        if detections is None:  # if there are detections
            assert boxes is not None and is_right is not None, (
                "If detections are None, boxes and is_right must be provided"
            )
            detections = [
                DetectionResult(
                    label="right_hand" if right else "left_hand",
                    rect=hand_bbox,  # (x1, y1, x2, y2)
                    score=0.8,
                )
                for right, hand_bbox in zip(is_right, boxes)
            ]

        if detections:  # if there are detections
            # Predict
            batch = self.preprocess_input(img=img, detections=detections)
            out = self.mocap(
                batch
            )  # ['pred_cam', 'pred_mano_params', 'pred_cam_t', 'focal_length', 'pred_keypoints_2d', 'pred_keypoints_3d', 'pred_vertices']

            # MANO params
            mano_params = out["pred_mano_params"]
            global_orients = mano_params["global_orient"].squeeze(1).detach().cpu().numpy()  # (B, 3, 3)
            thetas = mano_params["hand_pose"].squeeze(1).detach().cpu().numpy()  # (B, 15, 3, 3)
            betas = mano_params["betas"].detach().cpu().numpy()  # (B, 10)

            # Camera params
            pred_cam = out["pred_cam"]  # (B, 3)
            pred_cam[:, 1] *= 2 * batch["right"] - 1  # flip y-axis for left hand
            box_center = batch["box_center"].float()  # (B, 2)
            box_size = batch["box_size"].float()  # (B,)
            img_size = batch["img_size"].float()  # (B, 2)
            scaled_focal_length = self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = (
                cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)
                .detach()
                .cpu()
                .numpy()
            )  # (B, 3)

            # 2D keypoints
            boxes = np.array(
                [detection.rect for detection in detections], dtype=np.float32
            )  # (B, 4) x1, y1, x2, y2
            is_right = batch["right"].detach().cpu().numpy().astype(np.int32)  # (B,)
            box_center = batch["box_center"].detach().cpu().numpy()  # (B, 2)
            box_size = batch["box_size"].detach().cpu().numpy()  # (B,)
            pred_keypoints_2d = out["pred_keypoints_2d"].detach().cpu().numpy()  # (B, 21, 2)
            pred_keypoints_2d[:, :, 0] = (2 * is_right[:, None] - 1) * pred_keypoints_2d[
                :, :, 0
            ]  # flip x-axis for left hand
            pred_keypoints_2d = pred_keypoints_2d * box_size[:, None, None] + box_center[:, None, :]

            # 3D keypoints
            pred_keypoints_3d = out["pred_keypoints_3d"].detach().cpu().numpy()  # (B, 21, 3)
            pred_keypoints_3d[:, :, 0] = (2 * is_right[:, None] - 1) * pred_keypoints_3d[:, :, 0]
            pred_keypoints_3d += pred_cam_t_full[:, None, :]  # add translation wrt camera

            # 3D vertices wrtcamera
            pred_verts = out["pred_vertices"].detach().cpu().numpy()  # (B, 778, 3)
            pred_verts[:, :, 0] = (2 * is_right[:, None] - 1) * pred_verts[
                :, :, 0
            ]  # flip x-axis for left hand
            pred_verts += pred_cam_t_full[:, None, :]  # add translation wrt camera

            for i, hand_id in enumerate(is_right):  # for each hand
                assert detections[i].label == "right_hand" if hand_id == 1 else "left_hand", (
                    "Hand ID and hand detection mismatch"
                )
                orientation = global_orients[i]
                if hand_id == 0:  # left hand
                    orientation[1::3] *= -1
                    orientation[2::3] *= -1

                assert len(MANO_KEYPOINT_NAMES) == len(pred_keypoints_3d[i]) == len(pred_keypoints_2d[i]), (
                    "Keypoint mismatch"
                )
                mocap_result = MocapResult(
                    detection=detections[i],
                    position=pred_keypoints_3d[i][0],
                    orientation=orientation,
                    beta=betas[i],
                    theta=thetas[i],
                    keypoint_names=MANO_KEYPOINT_NAMES,
                    keypoints=pred_keypoints_3d[i],
                    keypoints_2d=pred_keypoints_2d[i],
                    vertices=pred_verts[i],  # 3D vertices
                )
                mocap_results.append(mocap_result)

        if self.visualize:
            vis_img = self.visualize_mocap(
                img=img,
                mocap_results=mocap_results,
                vis_img=vis_img,
            )
        else:
            vis_img = img.copy() if vis_img is None else vis_img

        return mocap_results, vis_img

    def visualize_mocap(
        self,
        img: np.ndarray,
        mocap_results: List[MocapResult],
        vis_img: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Visualize the motion capture results on the input image

        Args:
            img (np.ndarray): Input image, BGR format (H, W, C)
            mocap_results (List[MocapResult]): List of MocapResult
            vis_img (np.ndarray): Optional image for visualization

        Returns:
            vis_img np.ndarray: Visualization image
        """

        if not mocap_results:
            return img.copy() if vis_img is None else vis_img

        boxes = np.array([mocap_result.detection.rect for mocap_result in mocap_results], dtype=np.float32)
        is_right = np.array(
            [1 if mocap_result.detection.label == "right_hand" else 0 for mocap_result in mocap_results],
            dtype=np.int32,
        )
        keypoints_2d = np.array([mocap_result.keypoints_2d for mocap_result in mocap_results])
        vertices = np.array([mocap_result.vertices for mocap_result in mocap_results])

        if vis_img is None:
            # Draw BBOX and LABEL with annotator
            vis_img = img.copy()
            vis_detections = sv.Detections(
                xyxy=boxes,
                class_id=is_right,
            )
            vis_img = BOX_ANNOTATOR.annotate(scene=vis_img, detections=vis_detections)
            vis_img = LABEL_ANNOTATOR.annotate(
                scene=vis_img,
                detections=vis_detections,
                labels=["right_hand" if class_id == 1 else "left_hand" for class_id in is_right],
            )

        verts = []
        faces = []
        for i, vert in enumerate(vertices):
            face = self.mocap.mano.faces if is_right[i] else self.mocap.mano.faces[:, [0, 2, 1]]
            verts.append(vert)
            faces.append(face)
        rgba, mask, _ = self.renderer.render(verts=verts, faces=faces)
        rgb = rgba[..., :3].astype(np.float32)
        alpha = mask.astype(np.float32)
        vis_img = (alpha[..., None] * rgb + (1 - alpha[..., None]) * vis_img).astype(np.uint8)

        for keypoint_2d in keypoints_2d:
            vis_img = draw_hand_keypoints(vis_img, keypoint_2d)

        return vis_img


class HMR2Model(MocapModelBase):
    def __init__(
        self,
        K: np.ndarray = np.array(
            [[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1]]
        ),  # camera intrinsic matrix
        rescale_factor: float = 2.0,  # rescale factor for hand detection
        img_size: tuple = (640, 480),  # (width, height)
        visualize: bool = True,  # whether to visualize the result
        renderer: str = "pyrender",  # renderer type, 'pyrender' or 'pytorch3d'
        device: str = "cuda:0",  # device
    ):
        self.focal_length = K[0, 0]  # focal length from camera intrinsic matrix
        self.rescale_factor = rescale_factor
        self.img_size = img_size
        self.visualize = visualize
        self.device = device

        # init model
        self.mocap, self.cfg = load_hmr2(
            img_size=self.img_size,
            focal_length=self.focal_length,
        )
        self.mocap.to(self.device)
        self.mocap.eval()

        self.img_mean = 255.0 * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.img_std = 255.0 * np.array(self.cfg.MODEL.IMAGE_STD)
        self.bbox_shape = self.cfg.MODEL.get("BBOX_SHAPE", None)
        self.patch_width = self.patch_height = self.cfg.MODEL.IMAGE_SIZE

        if self.visualize:
            if renderer == "pyrender":
                self.renderer = PyrenderRenderer(
                    K=K,
                    width=self.img_size[0],
                    height=self.img_size[1],
                )
            elif renderer == "pytorch3d":
                self.renderer = Pytorch3DRenderer(
                    K=K,
                    width=self.img_size[0],
                    height=self.img_size[1],
                    device=self.device,
                    from_opencv=True,
                )
            else:
                raise ValueError(f"Invalid renderer type: {renderer}. Choose 'pyrender' or 'pytorch3d'.")

    @torch.no_grad()
    def predict(
        self,
        img: np.ndarray,
        detections: Optional[List[DetectionResult]] = None,
        boxes: Optional[Union[List[List[float]], np.ndarray]] = None,
        vis_img: Optional[np.ndarray] = None,
    ) -> Tuple[List[MocapResult], np.ndarray]:
        """
        Predict the hand pose from the input image

        Args:
            img (np.ndarray): Input image, BGR format (H, W, C)
            detections (List[DetectionResult]): List of detection results
            boxes (np.ndarray): Bounding boxes for the hands, shape (B, 4) in format [x1, y1, x2, y2]
            vis_img (np.ndarray): Optional image for visualization

        Returns:
            mocap_results List[MocapResult]: List of MocapResult
            vis_img np.ndarray: Visualization image
        """

        mocap_results = []
        if detections is None:  # if there are detections
            assert boxes is not None, "If detections are None, boxes must be provided"
            detections = [
                DetectionResult(
                    label="body",
                    rect=box,  # (x1, y1, x2, y2)
                    score=0.8,
                )
                for box in boxes
            ]

        if detections:  # if there are detections
            boxes = np.array([detection.rect for detection in detections])  # x1, y1, x2, y2

            # TODO clean it and fix this not to use datasetloader
            dataset = HMR2ViTDetDataset(self.cfg, img, boxes)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
            for batch in dataloader:
                batch = recursive_to(batch, self.device)  # to device
                out = self.mocap(batch)
            pred_cam = out["pred_cam"]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = (
                cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)
                .detach()
                .cpu()
                .numpy()
            )

            # this model uses 44 keypoints, but we use only 25 keypoints which corresponds to OpenPose keypoints
            # 2D keypoints
            box_center = batch["box_center"].detach().cpu().numpy()  # [N, 2]
            box_size = batch["box_size"].detach().cpu().numpy()  # [N,]
            pred_keypoints_2d = out["pred_keypoints_2d"].detach().cpu().numpy()  # [N, 44, 2]
            pred_keypoints_2d = pred_keypoints_2d * box_size[:, None, None] + box_center[:, None, :]
            pred_keypoints_2d = pred_keypoints_2d[:, :25, :]  # use only 25 keypoints

            # 3D keypoints
            pred_keypoints_3d = out["pred_keypoints_3d"].detach().cpu().numpy()  # [N, 44, 3]
            pred_keypoints_3d += pred_cam_t_full[:, None, :]
            pred_keypoints_3d = pred_keypoints_3d[:, :25, :]  # use only 25 keypoints

            # body pose
            body_origin = np.mean(pred_keypoints_2d, axis=1)  # [N, 2]
            body_origin = np.concatenate([body_origin, np.zeros((body_origin.shape[0], 1))], axis=1)  # [N, 3]
            global_orient = (
                out["pred_smpl_params"]["global_orient"].squeeze(1).detach().cpu().numpy()
            )  # [N, 3, 3]

            for i in range(len(detections)):  # for each body
                rotation = global_orient[i]

                quat = rotation_matrix_to_quaternion(rotation)  # [w, x, y, z]
                x_axis = np.array([0, 1, 0])
                y_axis = np.array([1, 0, 0])
                z_axis = np.array([0, 0, 1])
                rotated_result = R.from_rotvec(np.pi * np.array([1, 0, 0])) * R.from_quat(
                    quat
                )  # rotate 180 degree around x-axis
                quat = rotated_result.as_quat()  # [w, x, y, z]
                x_axis_rotated = rotation @ x_axis
                y_axis_rotated = rotation @ y_axis
                z_axis_rotated = rotation @ z_axis
                # visualize hand orientation
                # vis_im = draw_axis(vis_im, body_origin[i], x_axis_rotated, (0, 0, 255))  # x: red
                # vis_im = draw_axis(vis_im, body_origin[i], y_axis_rotated, (0, 255, 0))  # y: green
                # vis_im = draw_axis(vis_im, body_origin[i], z_axis_rotated, (255, 0, 0))  # z: blue

                assert len(SPIN_KEYPOINT_NAMES) == len(pred_keypoints_3d[i]), "Keypoint mismatch"
                mocap_result = MocapResult(
                    detection=detections[i],
                    position=pred_keypoints_3d[i][0],  #
                    keypoint_names=SPIN_KEYPOINT_NAMES,
                    keypoints=pred_keypoints_3d[i],
                    keypoints_2d=pred_keypoints_2d[i],
                )
                mocap_results.append(mocap_result)

            if self.visualize:
                all_verts = []
                all_cam_t = []

                # Render the result
                batch_size = batch["img"].shape[0]
                for n in range(batch_size):
                    # Add all verts and cams to list
                    verts = out["pred_vertices"][n].detach().cpu().numpy()
                    cam_t = pred_cam_t_full[n]
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)

                # Render front view
                if len(all_verts) > 0:
                    rgba, _ = self.renderer.render(all_verts, cam_t=all_cam_t)
                    rgb = rgba[..., :3].astype(np.float32)
                    alpha = rgba[..., 3].astype(np.float32) / 255.0
                    vis_img = (
                        alpha[..., None] * rgb + (1 - alpha[..., None]) * cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ).astype(np.uint8)
        else:  # no detections
            if vis_img is None:
                vis_im = img.copy()

        return mocap_results, vis_img


class HambaModel(MocapModelBase):
    def __init__(
        self,
        K: np.ndarray = np.array(
            [[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1]]
        ),  # camera intrinsic matrix
        rescale_factor: float = 2.0,  # rescale factor for hand detection
        img_size: tuple = (640, 480),  # (width, height)
        visualize: bool = True,  # whether to visualize the result
        renderer: str = "pyrender",  # renderer type, 'pyrender' or 'pytorch3d'
        device: str = "cuda:0",  # device
    ):
        self.focal_length = K[0, 0]  # focal length from camera intrinsic matrix
        self.rescale_factor = rescale_factor
        self.img_size = img_size
        self.visualize = visualize
        self.device = device

        # init model
        self.mocap, self.cfg = load_hamba(
            img_size=self.img_size,
            focal_length=self.focal_length,
        )
        self.mocap.to(self.device)
        self.mocap.eval()

        self.img_mean = 255.0 * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.img_std = 255.0 * np.array(self.cfg.MODEL.IMAGE_STD)
        self.bbox_shape = self.cfg.MODEL.get("BBOX_SHAPE", None)
        self.patch_width = self.patch_height = self.cfg.MODEL.IMAGE_SIZE

        if self.visualize:
            if renderer == "pyrender":
                self.renderer = PyrenderRenderer(
                    K=K,
                    width=self.img_size[0],
                    height=self.img_size[1],
                )
            elif renderer == "pytorch3d":
                self.renderer = Pytorch3DRenderer(
                    K=K,
                    width=self.img_size[0],
                    height=self.img_size[1],
                    device=self.device,
                    from_opencv=True,
                )
            else:
                raise ValueError(f"Invalid renderer type: {renderer}. Choose 'pyrender' or 'pytorch3d'.")

    def preprocess_input(self, img: np.ndarray, detections: List[DetectionResult]) -> Dict[str, torch.Tensor]:
        """
        Preprocess the input for the model

        Args:
            img (np.ndarray): Input image, BGR format (H, W, C)
            detections (List[DetectionResult]): List of detection results

        Returns:
            input_batch Dict[str, torch.Tensor]: Input batch for the model
                img (torch.Tensor): Image tensor (B, C, H, W)
                box_center (torch.Tensor): Box center (B, 2)
                box_size (torch.Tensor): Box size (B,)
                img_size (torch.Tensor): Image size (B, 2)
                right (torch.Tensor): Right hand flag (B,)
        """

        boxes = np.array([detection.rect for detection in detections]).astype(np.float32)  # x1, y1, x2, y2
        is_right = np.array([1 if detection.label == "right_hand" else 0 for detection in detections]).astype(
            np.float32
        )
        centers = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        scales = self.rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0

        # preprocess input for each detection
        processed_batch = dict()
        for center, scale, right in zip(centers, scales, is_right):
            center_x = center[0]
            center_y = center[1]

            bbox_size = expand_to_aspect_ratio(
                input_shape=scale * 200, target_aspect_ratio=self.bbox_shape
            ).max()
            flip = right == 0

            # 3. generate image patch
            downsampling_factor = (bbox_size * 1.0) / self.patch_width
            downsampling_factor = downsampling_factor / 2.0
            if downsampling_factor > 1.1:
                img = gaussian(img, sigma=(downsampling_factor - 1) / 2, channel_axis=2, preserve_range=True)

            img_patch_cv, _ = generate_image_patch_cv2(
                img=img,
                c_x=center_x,
                c_y=center_y,
                bb_width=bbox_size,
                bb_height=bbox_size,
                patch_width=self.patch_width,
                patch_height=self.patch_height,
                do_flip=flip,
                scale=1.0,
                rot=0,
                border_mode=cv2.BORDER_CONSTANT,
            )
            img_patch_cv = img_patch_cv[:, :, ::-1]
            img_patch = convert_cvimg_to_tensor(img_patch_cv)

            # apply normalization
            for n_c in range(min(img.shape[2], 3)):
                img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.img_mean[n_c]) / self.img_std[n_c]

            item = {
                "img": img_patch,
            }
            item["box_center"] = center.astype(np.float32)
            item["box_size"] = bbox_size.astype(np.float32)
            item["img_size"] = 1.0 * np.array([img.shape[1], img.shape[0]]).astype(np.float32)
            item["right"] = right.astype(np.float32)

            # processed batch is like batch['img'] = torch.Tensor (B, C, H, W)
            for key, value in item.items():
                if key not in processed_batch:
                    processed_batch[key] = []
                processed_batch[key].append(value)

        # Convertto torch.Tensor
        for key, value in processed_batch.items():
            val = np.stack(value, axis=0)
            processed_batch[key] = torch.from_numpy(val).float().to(self.device)

        return processed_batch

    @torch.no_grad()
    def predict(
        self,
        img: np.ndarray,
        detections: Optional[List[DetectionResult]] = None,
        boxes: Optional[Union[List[List[float]], np.ndarray]] = None,
        is_right: Optional[Union[List[int], np.ndarray]] = None,
        vis_img: Optional[np.ndarray] = None,
    ) -> Tuple[List[MocapResult], np.ndarray]:
        """
        Predict the hand pose from the input image

        Args:
            img (np.ndarray): Input image, BGR format (H, W, C)
            detections (List[DetectionResult]): List of detection results
            boxes (np.ndarray): Bounding boxes for the hands, shape (B, 4) in format [x1, y1, x2, y2]
            is_right (np.ndarray): Array indicating if the hand is right (1) or left (0), shape (B,)
            vis_img (np.ndarray): Optional image for visualization

        Returns:
            mocap_results List[MocapResult]: List of MocapResult
            vis_img np.ndarray: Visualization image
        """
        mocap_results = []
        if detections is None:  # if there are detections
            assert boxes is not None and is_right is not None, (
                "If detections are None, boxes and is_right must be provided"
            )
            detections = [
                DetectionResult(
                    label="right_hand" if right else "left_hand",
                    rect=hand_bbox,  # (x1, y1, x2, y2)
                    score=0.8,
                )
                for right, hand_bbox in zip(is_right, boxes)
            ]

        if detections:  # if there are detections
            # Predict
            batch = self.preprocess_input(img=img, detections=detections)
            out = self.mocap(
                batch
            )  # ['pred_cam', 'pred_mano_params', 'pred_cam_t', 'focal_length', 'pred_keypoints_2d', 'pred_keypoints_3d', 'pred_vertices']

            # MANO params
            mano_params = out["pred_mano_params"]
            global_orients = mano_params["global_orient"].squeeze(1).detach().cpu().numpy()  # (B, 3, 3)
            thetas = mano_params["hand_pose"].squeeze(1).detach().cpu().numpy()  # (B, 15, 3, 3)
            betas = mano_params["betas"].detach().cpu().numpy()  # (B, 10)

            # Camera params
            pred_cam = out["pred_cam"]  # (B, 3)
            pred_cam[:, 1] *= 2 * batch["right"] - 1  # flip y-axis for left hand
            box_center = batch["box_center"].float()  # (B, 2)
            box_size = batch["box_size"].float()  # (B,)
            img_size = batch["img_size"].float()  # (B, 2)
            scaled_focal_length = self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = (
                cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)
                .detach()
                .cpu()
                .numpy()
            )  # (B, 3)

            # 2D keypoints
            boxes = np.array(
                [detection.rect for detection in detections], dtype=np.float32
            )  # (B, 4) x1, y1, x2, y2
            is_right = batch["right"].detach().cpu().numpy().astype(np.int32)  # (B,)
            box_center = batch["box_center"].detach().cpu().numpy()  # (B, 2)
            box_size = batch["box_size"].detach().cpu().numpy()  # (B,)
            pred_keypoints_2d = out["pred_keypoints_2d"].detach().cpu().numpy()  # (B, 21, 2)
            pred_keypoints_2d[:, :, 0] = (2 * is_right[:, None] - 1) * pred_keypoints_2d[
                :, :, 0
            ]  # flip x-axis for left hand
            pred_keypoints_2d = pred_keypoints_2d * box_size[:, None, None] + box_center[:, None, :]

            # 3D keypoints
            pred_keypoints_3d = out["pred_keypoints_3d"].detach().cpu().numpy()  # (B, 21, 3)
            pred_keypoints_3d[:, :, 0] = (2 * is_right[:, None] - 1) * pred_keypoints_3d[:, :, 0]
            pred_keypoints_3d += pred_cam_t_full[:, None, :]  # add translation wrt camera

            # 3D vertices wrtcamera
            pred_verts = out["pred_vertices"].detach().cpu().numpy()  # (B, 778, 3)
            pred_verts[:, :, 0] = (2 * is_right[:, None] - 1) * pred_verts[
                :, :, 0
            ]  # flip x-axis for left hand
            pred_verts += pred_cam_t_full[:, None, :]  # add translation wrt camera

            for i, hand_id in enumerate(is_right):  # for each hand
                assert detections[i].label == "right_hand" if hand_id == 1 else "left_hand", (
                    "Hand ID and hand detection mismatch"
                )
                orientation = global_orients[i]
                if hand_id == 0:  # left hand
                    orientation[1::3] *= -1
                    orientation[2::3] *= -1

                assert len(MANO_KEYPOINT_NAMES) == len(pred_keypoints_3d[i]) == len(pred_keypoints_2d[i]), (
                    "Keypoint mismatch"
                )
                mocap_result = MocapResult(
                    detection=detections[i],
                    position=pred_keypoints_3d[i][0],
                    orientation=orientation,
                    beta=betas[i],
                    theta=thetas[i],
                    keypoint_names=MANO_KEYPOINT_NAMES,
                    keypoints=pred_keypoints_3d[i],
                    keypoints_2d=pred_keypoints_2d[i],
                    vertices=pred_verts[i],  # 3D vertices
                )
                mocap_results.append(mocap_result)

        if self.visualize:
            vis_img = self.visualize_mocap(
                img=img,
                mocap_results=mocap_results,
                vis_img=vis_img,
            )
        else:
            vis_img = img.copy() if vis_img is None else vis_img

        return mocap_results, vis_img

    def visualize_mocap(
        self,
        img: np.ndarray,
        mocap_results: List[MocapResult],
        vis_img: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Visualize the motion capture results on the input image

        Args:
            img (np.ndarray): Input image, BGR format (H, W, C)
            mocap_results (List[MocapResult]): List of MocapResult
            vis_img (np.ndarray): Optional image for visualization

        Returns:
            vis_img np.ndarray: Visualization image
        """

        if not mocap_results:
            return img.copy() if vis_img is None else vis_img

        boxes = np.array([mocap_result.detection.rect for mocap_result in mocap_results], dtype=np.float32)
        is_right = np.array(
            [1 if mocap_result.detection.label == "right_hand" else 0 for mocap_result in mocap_results],
            dtype=np.int32,
        )
        keypoints_2d = np.array([mocap_result.keypoints_2d for mocap_result in mocap_results])
        vertices = np.array([mocap_result.vertices for mocap_result in mocap_results])

        if vis_img is None:
            # Draw BBOX and LABEL with annotator
            vis_img = img.copy()
            vis_detections = sv.Detections(
                xyxy=boxes,
                class_id=is_right,
            )
            vis_img = BOX_ANNOTATOR.annotate(scene=vis_img, detections=vis_detections)
            vis_img = LABEL_ANNOTATOR.annotate(
                scene=vis_img,
                detections=vis_detections,
                labels=["right_hand" if class_id == 1 else "left_hand" for class_id in is_right],
            )

        verts = []
        faces = []
        for i, vert in enumerate(vertices):
            face = self.mocap.mano.faces if is_right[i] else self.mocap.mano.faces[:, [0, 2, 1]]
            verts.append(vert)
            faces.append(face)
        rgba, mask, _ = self.renderer.render(verts=verts, faces=faces)
        rgb = rgba[..., :3].astype(np.float32)
        alpha = mask.astype(np.float32)
        vis_img = (alpha[..., None] * rgb + (1 - alpha[..., None]) * vis_img).astype(np.uint8)

        for keypoint_2d in keypoints_2d:
            vis_img = draw_hand_keypoints(vis_img, keypoint_2d)

        return vis_img


class HMR2Model(MocapModelBase):
    def __init__(
        self,
        focal_length: float = 525.0,
        rescale_factor: float = 2.0,
        img_size: tuple = (640, 480),
        visualize: bool = True,
        device: str = "cuda:0",
    ):
        self.display = Display(visible=0, size=img_size)
        self.display.start()
        self.focal_length = focal_length
        self.rescale_factor = rescale_factor
        self.img_size = img_size
        self.visualize = visualize
        self.device = device

        # init model
        self.mocap, self.model_cfg = load_hmr2(
            img_size=self.img_size,
            focal_length=self.focal_length,
        )
        self.mocap.to(self.device)
        self.mocap.eval()
        if self.visualize:
            self.renderer = Renderer(
                faces=self.mocap.smpl.faces,
                cfg=self.model_cfg,
                width=self.img_size[0],
                height=self.img_size[1],
            )

    @torch.no_grad()
    def predict(
        self, img: np.ndarray, detections: List[DetectionResult], vis_img: Optional[np.ndarray] = None
    ) -> Tuple[List[MocapResult], np.ndarray]:
        """
        Predict the hand pose from the input image

        Args:
            img (np.ndarray): Input image, BGR format (H, W, C)
            detections (List[DetectionResult]): List of detection results
            vis_img (np.ndarray): Optional image for visualization

        Returns:
            mocap_results List[MocapResult]: List of MocapResult
            vis_img np.ndarray: Visualization image
        """

        mocap_results = []
        if detections:  # if there are detections
            boxes = np.array([detection.rect for detection in detections])  # x1, y1, x2, y2

            # TODO clean it and fix this not to use datasetloader
            dataset = HMR2ViTDetDataset(self.model_cfg, img, boxes)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
            for batch in dataloader:
                batch = recursive_to(batch, self.device)  # to device
                out = self.mocap(batch)
            pred_cam = out["pred_cam"]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = (
                self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            )
            pred_cam_t_full = (
                cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)
                .detach()
                .cpu()
                .numpy()
            )

            # this model uses 44 keypoints, but we use only 25 keypoints which corresponds to OpenPose keypoints
            # 2D keypoints
            box_center = batch["box_center"].detach().cpu().numpy()  # [N, 2]
            box_size = batch["box_size"].detach().cpu().numpy()  # [N,]
            pred_keypoints_2d = out["pred_keypoints_2d"].detach().cpu().numpy()  # [N, 44, 2]
            pred_keypoints_2d = pred_keypoints_2d * box_size[:, None, None] + box_center[:, None, :]
            pred_keypoints_2d = pred_keypoints_2d[:, :25, :]  # use only 25 keypoints

            # 3D keypoints
            pred_keypoints_3d = out["pred_keypoints_3d"].detach().cpu().numpy()  # [N, 44, 3]
            pred_keypoints_3d += pred_cam_t_full[:, None, :]
            pred_keypoints_3d = pred_keypoints_3d[:, :25, :]  # use only 25 keypoints

            # body pose
            body_origin = np.mean(pred_keypoints_2d, axis=1)  # [N, 2]
            body_origin = np.concatenate([body_origin, np.zeros((body_origin.shape[0], 1))], axis=1)  # [N, 3]
            global_orient = (
                out["pred_smpl_params"]["global_orient"].squeeze(1).detach().cpu().numpy()
            )  # [N, 3, 3]

            for i in range(len(detections)):  # for each body
                rotation = global_orient[i]

                quat = rotation_matrix_to_quaternion(rotation)  # [w, x, y, z]
                x_axis = np.array([0, 1, 0])
                y_axis = np.array([1, 0, 0])
                z_axis = np.array([0, 0, 1])
                rotated_result = R.from_rotvec(np.pi * np.array([1, 0, 0])) * R.from_quat(
                    quat
                )  # rotate 180 degree around x-axis
                quat = rotated_result.as_quat()  # [w, x, y, z]
                x_axis_rotated = rotation @ x_axis
                y_axis_rotated = rotation @ y_axis
                z_axis_rotated = rotation @ z_axis
                # visualize hand orientation
                # vis_im = draw_axis(vis_im, body_origin[i], x_axis_rotated, (0, 0, 255))  # x: red
                # vis_im = draw_axis(vis_im, body_origin[i], y_axis_rotated, (0, 255, 0))  # y: green
                # vis_im = draw_axis(vis_im, body_origin[i], z_axis_rotated, (255, 0, 0))  # z: blue

                assert len(SPIN_KEYPOINT_NAMES) == len(pred_keypoints_3d[i]), "Keypoint mismatch"
                mocap_result = MocapResult(
                    detection=detections[i],
                    position=pred_keypoints_3d[i][0],  #
                    keypoint_names=SPIN_KEYPOINT_NAMES,
                    keypoints=pred_keypoints_3d[i],
                    keypoints_2d=pred_keypoints_2d[i],
                )
                mocap_results.append(mocap_result)

            if self.visualize:
                all_verts = []
                all_cam_t = []

                # Render the result
                batch_size = batch["img"].shape[0]
                for n in range(batch_size):
                    # Add all verts and cams to list
                    verts = out["pred_vertices"][n].detach().cpu().numpy()
                    cam_t = pred_cam_t_full[n]
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)

                # Render front view
                if len(all_verts) > 0:
                    rgba, _ = self.renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t)
                    rgb = rgba[..., :3].astype(np.float32)
                    alpha = rgba[..., 3].astype(np.float32) / 255.0
                    vis_img = (
                        alpha[..., None] * rgb + (1 - alpha[..., None]) * cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ).astype(np.uint8)
        else:  # no detections
            if vis_img is None:
                vis_im = img.copy()

        return mocap_results, vis_img
