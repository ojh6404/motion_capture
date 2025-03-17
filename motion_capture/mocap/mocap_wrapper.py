#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
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
from hmr2.models import DEFAULT_CHECKPOINT
from hmr2.datasets.vitdet_dataset import ViTDetDataset as HMR2ViTDetDataset

# utils and constants
from motion_capture.detector import DetectionResult
from motion_capture.utils.utils import (
    rotation_matrix_to_quaternion,
    draw_axis,
    load_hamer,
    load_wilor,
    load_hmr2,
    recursive_to,
    cam_crop_to_full,
    Renderer,
    draw_hand_keypoints,
)
from motion_capture.utils import (
    MANO_JOINTS_CONNECTION,
    MANO_CONNECTION_NAMES,
    MANO_KEYPOINT_NAMES,
    SPIN_KEYPOINT_NAMES,
    HAMER_CHECKPOINT_PATH,
    HAMER_CONFIG_PATH,
    WILOR_CHECKPOINT_PATH,
    WILOR_CONFIG_PATH,
)


BOX_ANNOTATOR = sv.BoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()


@dataclass
class MocapResult:
    detection: DetectionResult
    position: np.ndarray
    orientation: np.ndarray
    keypoint_names: List[str]
    keypoints: np.ndarray
    keypoints_2d: np.ndarray
    # betas: np.ndarray # MANO betas
    # thetas: np.ndarray # MANO thetas


class MocapModelFactory:
    @staticmethod
    def from_config(model: str, model_config: dict):
        if model == "hamer":
            return HamerModel(**model_config)
        elif model == "wilor":
            return WiLoRModel(**model_config)
        elif model == "4d-human":
            return HMR2Model(**model_config)
        else:
            raise ValueError(f"Invalid mocap model: {model_config['model']}")


class MocapModelBase(ABC):
    @abstractmethod
    def predict(
        self, img: np.ndarray, detections: List[DetectionResult], vis_img: Optional[np.ndarray] = None
    ) -> Tuple[List[MocapResult], np.ndarray]:
        pass


class HamerModel(MocapModelBase):
    def __init__(
        self,
        focal_length: float = 525.0,  # focal length
        rescale_factor: float = 2.0,  # rescale factor for hand detection
        img_size: tuple = (640, 480),  # (width, height)
        visualize: bool = True,  # whether to visualize the result
        device: str = "cuda:0",  # device
    ):
        self.display = Display(visible=0, size=img_size)
        self.display.start()
        self.focal_length = focal_length
        self.rescale_factor = rescale_factor
        self.img_size = img_size
        self.visualize = visualize
        self.device = device

        # init model
        self.mocap, self.model_cfg = load_hamer(
            HAMER_CHECKPOINT_PATH,
            HAMER_CONFIG_PATH,
            img_size=self.img_size,
            focal_length=self.focal_length,
        )
        self.mocap.to(self.device)
        self.mocap.eval()

        self.img_mean = 255.0 * np.array(self.model_cfg.MODEL.IMAGE_MEAN)
        self.img_std = 255.0 * np.array(self.model_cfg.MODEL.IMAGE_STD)
        self.bbox_shape = self.model_cfg.MODEL.get("BBOX_SHAPE", None)
        self.patch_width = self.patch_height = self.model_cfg.MODEL.IMAGE_SIZE

        if self.visualize:
            self.renderer = Renderer(
                faces=self.mocap.mano.faces,
                cfg=self.model_cfg,
                width=self.img_size[0],
                height=self.img_size[1],
            )

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
        rights = np.array([1 if detection.label == "right_hand" else 0 for detection in detections]).astype(
            np.float32
        )
        centers = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        scales = self.rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0

        # preprocess input for each detection
        processed_batch = dict()
        for center, scale, right in zip(centers, scales, rights):
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
            right = np.array([1 if detection.label == "right_hand" else 0 for detection in detections])

            batch = self.preprocess_input(img=img, detections=detections)
            out = self.mocap(batch)
            pred_cam = out["pred_cam"]
            pred_cam[:, 1] *= 2 * batch["right"] - 1
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

            # 2D keypoints
            box_center = batch["box_center"].detach().cpu().numpy()  # [N, 2]
            box_size = batch["box_size"].detach().cpu().numpy()  # [N,]
            pred_keypoints_2d = out["pred_keypoints_2d"].detach().cpu().numpy()  # [N, 21, 2]
            pred_keypoints_2d[:, :, 0] = (2 * right[:, None] - 1) * pred_keypoints_2d[
                :, :, 0
            ]  # flip x-axis for left hand
            pred_keypoints_2d = pred_keypoints_2d * box_size[:, None, None] + box_center[:, None, :]

            # 3D keypoints
            pred_keypoints_3d = out["pred_keypoints_3d"].detach().cpu().numpy()  # [N, 21, 3]
            pred_keypoints_3d[:, :, 0] = (2 * right[:, None] - 1) * pred_keypoints_3d[:, :, 0]
            pred_keypoints_3d += pred_cam_t_full[:, None, :]

            # hand pose
            hand_origin = np.mean(pred_keypoints_2d, axis=1)  # [N, 2]
            hand_origin = np.concatenate([hand_origin, np.zeros((hand_origin.shape[0], 1))], axis=1)  # [N, 3]
            global_orient = (
                out["pred_mano_params"]["global_orient"].squeeze(1).detach().cpu().numpy()
            )  # [N, 3, 3]

            for i, hand_id in enumerate(right):  # for each hand
                assert (
                    detections[i].label == "right_hand" if hand_id == 1 else "left_hand"
                ), "Hand ID and hand detection mismatch"
                rotation = global_orient[i]
                if hand_id == 0:
                    rotation[1::3] *= -1
                    rotation[2::3] *= -1

                quat = rotation_matrix_to_quaternion(rotation)  # [w, x, y, z]
                if right[i] == 1:
                    x_axis = np.array([0, 0, 1])
                    y_axis = np.array([0, -1, 0])
                    z_axis = np.array([-1, 0, 0])
                    rotated_result = R.from_rotvec(np.pi * np.array([1, 0, 0])) * R.from_quat(
                        quat
                    )  # rotate 180 degree around x-axis
                    quat = rotated_result.as_quat()  # [w, x, y, z]
                else:
                    x_axis = np.array([0, 0, -1])
                    y_axis = np.array([0, -1, 0])
                    z_axis = np.array([1, 0, 0])
                    rotated_result = R.from_rotvec(np.pi * np.array([0, 0, 1])) * R.from_quat(
                        quat
                    )  # rotate 180 degree around x-axis
                    quat = rotated_result.as_quat()  # [w, x, y, z]
                x_axis_rotated = rotation @ x_axis
                y_axis_rotated = rotation @ y_axis
                z_axis_rotated = rotation @ z_axis

                assert len(MANO_KEYPOINT_NAMES) == len(pred_keypoints_3d[i]), "Keypoint mismatch"
                mocap_result = MocapResult(
                    detection=detections[i],
                    position=pred_keypoints_3d[i][0],  # wrist position
                    orientation=quat,
                    keypoint_names=MANO_KEYPOINT_NAMES,
                    keypoints=pred_keypoints_3d[i],
                    keypoints_2d=pred_keypoints_2d[i],
                )
                mocap_results.append(mocap_result)

            if self.visualize:
                if vis_img is None:
                    # Draw BBOX and LABEL with annotator
                    vis_img = img.copy()
                    vis_detections = sv.Detections(
                        xyxy=boxes,
                        class_id=right,
                    )
                    vis_img = BOX_ANNOTATOR.annotate(scene=vis_img, detections=vis_detections)
                    vis_img = LABEL_ANNOTATOR.annotate(
                        scene=vis_img,
                        detections=vis_detections,
                        labels=["right_hand" if class_id == 1 else "left_hand" for class_id in right],
                    )

                all_verts = []
                all_cam_t = []
                all_right = []

                # Render the result
                batch_size = batch["img"].shape[0]
                for n in range(batch_size):
                    # Add all verts and cams to list
                    verts = out["pred_vertices"][n].detach().cpu().numpy()
                    is_right = batch["right"][n].cpu().numpy()
                    verts[:, 0] = (2 * is_right - 1) * verts[:, 0]  # Flip x-axis
                    cam_t = pred_cam_t_full[n]
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
                    all_right.append(is_right)

                # Render front view
                if len(all_verts) > 0:
                    rgba, _ = self.renderer.render_rgba_multiple(
                        all_verts, cam_t=all_cam_t, is_right=all_right
                    )
                    rgb = rgba[..., :3].astype(np.float32)
                    alpha = rgba[..., 3].astype(np.float32) / 255.0
                    vis_img = (alpha[..., None] * rgb + (1 - alpha[..., None]) * vis_img).astype(np.uint8)

                for pred_keypoint_2d in pred_keypoints_2d:
                    vis_img = draw_hand_keypoints(vis_img, pred_keypoint_2d)

                # for i in range(len(detections)):
                #     # visualize hand orientation
                #     vis_im = draw_axis(vis_im, hand_origin[i], x_axis_rotated, (0, 0, 255))  # x: red
                #     vis_im = draw_axis(vis_im, hand_origin[i], y_axis_rotated, (0, 255, 0))  # y: green
                #     vis_im = draw_axis(vis_im, hand_origin[i], z_axis_rotated, (255, 0, 0))  # z: blue

        else:  # no detections
            if vis_img is None:
                vis_img = img.copy()

        return mocap_results, vis_img

    def __del__(self):
        self.display.stop()


class WiLoRModel(MocapModelBase):
    def __init__(
        self,
        focal_length: float = 525.0,  # focal length
        rescale_factor: float = 2.0,  # rescale factor for hand detection
        img_size: tuple = (640, 480),  # (width, height)
        visualize: bool = True,  # whether to visualize the result
        device: str = "cuda:0",  # device
    ):
        self.display = Display(visible=0, size=img_size)
        self.display.start()
        self.focal_length = focal_length
        self.rescale_factor = rescale_factor
        self.img_size = img_size
        self.visualize = visualize
        self.device = device

        # init model
        self.mocap, self.model_cfg = load_wilor(
            WILOR_CHECKPOINT_PATH,
            WILOR_CONFIG_PATH,
            img_size=self.img_size,
            focal_length=self.focal_length,
        )
        self.mocap.to(self.device)
        self.mocap.eval()

        self.img_mean = 255.0 * np.array(self.model_cfg.MODEL.IMAGE_MEAN)
        self.img_std = 255.0 * np.array(self.model_cfg.MODEL.IMAGE_STD)
        self.bbox_shape = self.model_cfg.MODEL.get("BBOX_SHAPE", None)
        self.patch_width = self.patch_height = self.model_cfg.MODEL.IMAGE_SIZE

        if self.visualize:
            self.renderer = Renderer(
                faces=self.mocap.mano.faces,
                cfg=self.model_cfg,
                width=self.img_size[0],
                height=self.img_size[1],
            )

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
        rights = np.array([1 if detection.label == "right_hand" else 0 for detection in detections]).astype(
            np.float32
        )
        centers = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        scales = self.rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0

        # preprocess input for each detection
        processed_batch = dict()
        for center, scale, right in zip(centers, scales, rights):
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
            right = np.array([1 if detection.label == "right_hand" else 0 for detection in detections])

            batch = self.preprocess_input(img=img, detections=detections)
            out = self.mocap(batch)
            pred_cam = out["pred_cam"]
            pred_cam[:, 1] *= 2 * batch["right"] - 1
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

            # 2D keypoints
            box_center = batch["box_center"].detach().cpu().numpy()  # [N, 2]
            box_size = batch["box_size"].detach().cpu().numpy()  # [N,]
            pred_keypoints_2d = out["pred_keypoints_2d"].detach().cpu().numpy()  # [N, 21, 2]
            pred_keypoints_2d[:, :, 0] = (2 * right[:, None] - 1) * pred_keypoints_2d[
                :, :, 0
            ]  # flip x-axis for left hand
            pred_keypoints_2d = pred_keypoints_2d * box_size[:, None, None] + box_center[:, None, :]

            # 3D keypoints
            pred_keypoints_3d = out["pred_keypoints_3d"].detach().cpu().numpy()  # [N, 21, 3]
            pred_keypoints_3d[:, :, 0] = (2 * right[:, None] - 1) * pred_keypoints_3d[:, :, 0]
            pred_keypoints_3d += pred_cam_t_full[:, None, :]

            # hand pose
            hand_origin = np.mean(pred_keypoints_2d, axis=1)  # [N, 2]
            hand_origin = np.concatenate([hand_origin, np.zeros((hand_origin.shape[0], 1))], axis=1)  # [N, 3]
            global_orient = (
                out["pred_mano_params"]["global_orient"].squeeze(1).detach().cpu().numpy()
            )  # [N, 3, 3]

            for i, hand_id in enumerate(right):  # for each hand
                assert (
                    detections[i].label == "right_hand" if hand_id == 1 else "left_hand"
                ), "Hand ID and hand detection mismatch"
                rotation = global_orient[i]
                if hand_id == 0:
                    rotation[1::3] *= -1
                    rotation[2::3] *= -1

                quat = rotation_matrix_to_quaternion(rotation)  # [w, x, y, z]
                if right[i] == 1:
                    x_axis = np.array([0, 0, 1])
                    y_axis = np.array([0, -1, 0])
                    z_axis = np.array([-1, 0, 0])
                    rotated_result = R.from_rotvec(np.pi * np.array([1, 0, 0])) * R.from_quat(
                        quat
                    )  # rotate 180 degree around x-axis
                    quat = rotated_result.as_quat()  # [w, x, y, z]
                else:
                    x_axis = np.array([0, 0, -1])
                    y_axis = np.array([0, -1, 0])
                    z_axis = np.array([1, 0, 0])
                    rotated_result = R.from_rotvec(np.pi * np.array([0, 0, 1])) * R.from_quat(
                        quat
                    )  # rotate 180 degree around x-axis
                    quat = rotated_result.as_quat()  # [w, x, y, z]
                x_axis_rotated = rotation @ x_axis
                y_axis_rotated = rotation @ y_axis
                z_axis_rotated = rotation @ z_axis

                assert len(MANO_KEYPOINT_NAMES) == len(pred_keypoints_3d[i]), "Keypoint mismatch"
                mocap_result = MocapResult(
                    detection=detections[i],
                    position=pred_keypoints_3d[i][0],  # wrist position
                    orientation=quat,
                    keypoint_names=MANO_KEYPOINT_NAMES,
                    keypoints=pred_keypoints_3d[i],
                    keypoints_2d=pred_keypoints_2d[i],
                )
                mocap_results.append(mocap_result)

            if self.visualize:
                if vis_img is None:
                    # Draw BBOX and LABEL with annotator
                    vis_img = img.copy()
                    vis_detections = sv.Detections(
                        xyxy=boxes,
                        class_id=right,
                    )
                    vis_img = BOX_ANNOTATOR.annotate(scene=vis_img, detections=vis_detections)
                    vis_img = LABEL_ANNOTATOR.annotate(
                        scene=vis_img,
                        detections=vis_detections,
                        labels=["right_hand" if class_id == 1 else "left_hand" for class_id in right],
                    )

                all_verts = []
                all_cam_t = []
                all_right = []

                # Render the result
                batch_size = batch["img"].shape[0]
                for n in range(batch_size):
                    # Add all verts and cams to list
                    verts = out["pred_vertices"][n].detach().cpu().numpy()
                    is_right = batch["right"][n].cpu().numpy()
                    verts[:, 0] = (2 * is_right - 1) * verts[:, 0]  # Flip x-axis
                    cam_t = pred_cam_t_full[n]
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
                    all_right.append(is_right)

                # Render front view
                if len(all_verts) > 0:
                    rgba, _ = self.renderer.render_rgba_multiple(
                        all_verts, cam_t=all_cam_t, is_right=all_right
                    )
                    rgb = rgba[..., :3].astype(np.float32)
                    alpha = rgba[..., 3].astype(np.float32) / 255.0
                    vis_img = (alpha[..., None] * rgb + (1 - alpha[..., None]) * vis_img).astype(np.uint8)

                for pred_keypoint_2d in pred_keypoints_2d:
                    vis_img = draw_hand_keypoints(vis_img, pred_keypoint_2d)

                # for i in range(len(detections)):
                #     # visualize hand orientation
                #     vis_im = draw_axis(vis_im, hand_origin[i], x_axis_rotated, (0, 0, 255))  # x: red
                #     vis_im = draw_axis(vis_im, hand_origin[i], y_axis_rotated, (0, 255, 0))  # y: green
                #     vis_im = draw_axis(vis_im, hand_origin[i], z_axis_rotated, (255, 0, 0))  # z: blue

        else:  # no detections
            if vis_img is None:
                vis_img = img.copy()

        return mocap_results, vis_img

    def __del__(self):
        self.display.stop()


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
            DEFAULT_CHECKPOINT,
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
                    orientation=quat,
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

    def __del__(self):
        self.display.stop()
