#! /usr/bin/env python
# -*- coding:utf-8 -*-

import os
import numpy as np

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
THIRD_PARTY_ROOT = os.path.join(PACKAGE_ROOT, "third_party")

# Hand Object Detector paths
HAND_OBJECT_DETECTOR_ROOT = os.path.join(THIRD_PARTY_ROOT, "hand_object_detector")
HAND_OBJECT_MODEL_PATH = HAND_OBJECT_DETECTOR_ROOT + "/lib"
FONT_PATH = HAND_OBJECT_MODEL_PATH + "/hand_object_detector/utils/times_b.ttf"
HAND_OBJECT_DETECTOR_CHECKPOINT_PATH = PACKAGE_ROOT + "/weights/hand_object_detector.pth"
PASCAL_CLASSES = np.asarray(["__background__", "targetobject", "hand"])  # for hand object detector

# Body model paths
DATA_ROOT = os.path.join(PACKAGE_ROOT, "data")
MANO_ROOT = os.path.join(DATA_ROOT, "mano")
SMPL_ROOT = os.path.join(DATA_ROOT, "smpl")
SMPLX_ROOT = os.path.join(DATA_ROOT, "smplx")

# HaMeR paths
HAMER_ROOT = os.path.join(THIRD_PARTY_ROOT, "hamer")
HAMER_CHECKPOINT_PATH = PACKAGE_ROOT + "/weights/hamer.ckpt"
HAMER_CONFIG_PATH = PACKAGE_ROOT + "/cfgs/hamer.yaml"

# WiLoR paths
WILOR_ROOT = os.path.join(THIRD_PARTY_ROOT, "WiLoR")
WILOR_CHECKPOINT_PATH = PACKAGE_ROOT + "/weights/wilor.ckpt"
WILOR_CONFIG_PATH = PACKAGE_ROOT + "/cfgs/wilor.yaml"

# Hamba paths
HAMBA_ROOT = os.path.join(THIRD_PARTY_ROOT, "Hamba")
HAMBA_CHECKPOINT_PATH = PACKAGE_ROOT + "/weights/hamba.ckpt"
HAMBA_CONFIG_PATH = PACKAGE_ROOT + "/cfgs/hamba.yaml"

# HMR2 paths
HMR2_ROOT = os.path.join(THIRD_PARTY_ROOT, "4D-Humans")
HMR2_CHECKPOINT_PATH = PACKAGE_ROOT + "/weights/hmr2.ckpt"
HMR2_CONFIG_PATH = PACKAGE_ROOT + "/cfgs/hmr2.yaml"

# Hand colors
HAND_COLOR = (0.65098039, 0.74117647, 0.85882353)


# hand constants
MANO_KEYPOINT_NAMES = [
    "wrist",
    "thumb0",
    "thumb1",
    "thumb2",
    "thumb3",
    "index0",
    "index1",
    "index2",
    "index3",
    "middle0",
    "middle1",
    "middle2",
    "middle3",
    "ring0",
    "ring1",
    "ring2",
    "ring3",
    "pinky0",
    "pinky1",
    "pinky2",
    "pinky3",
]
MANO_JOINTS_CONNECTION = [
    (0, 1),  # wrist -> thumb0
    (1, 2),  # thumb0 -> thumb1
    (2, 3),  # thumb1 -> thumb2
    (3, 4),  # thumb2 -> thumb3
    (0, 5),  # wrist -> index0
    (5, 6),  # index0 -> index1
    (6, 7),  # index1 -> index2
    (7, 8),  # index2 -> index3
    (0, 9),  # wrist -> middle0
    (9, 10),  # middle0 -> middle1
    (10, 11),  # middle1 -> middle2
    (11, 12),  # middle2 -> middle3
    (0, 13),  # wrist -> ring0
    (13, 14),  # ring0 -> ring1
    (14, 15),  # ring1 -> ring2
    (15, 16),  # ring2 -> ring3
    (0, 17),  # wrist -> pinky0
    (17, 18),  # pinky0 -> pinky1
    (18, 19),  # pinky1 -> pinky2
    (19, 20),  # pinky2 -> pinky3
]
MANO_CONNECTION_NAMES = [
    "wrist->thumb0",
    "thumb0->thumb1",
    "thumb1->thumb2",
    "thumb2->thumb3",
    "wrist->index0",
    "index0->index1",
    "index1->index2",
    "index2->index3",
    "wrist->middle0",
    "middle0->middle1",
    "middle1->middle2",
    "middle2->middle3",
    "wrist->ring0",
    "ring0->ring1",
    "ring1->ring2",
    "ring2->ring3",
    "wrist->pinky0",
    "pinky0->pinky1",
    "pinky1->pinky2",
    "pinky2->pinky3",
]

# SMPL constants
SMPL_KEYPOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]
SMPL_JOINTS_CONNECTION = [
    (0, 1),  # pelvis -> left_hip
    (0, 2),  # pelvis -> right_hip
    (0, 3),  # pelvis -> spine1
    (1, 4),  # left_hip -> left_knee
    (2, 5),  # right_hip -> right_knee
    (3, 6),  # spine1 -> spine2
    (4, 7),  # left_knee -> left_ankle
    (5, 8),  # right_knee -> right_ankle
    (6, 9),  # spine2 -> spine3
    (7, 10),  # left_ankle -> left_foot
    (8, 11),  # right_ankle -> right_foot
    (9, 12),  # spine3 -> neck
    (12, 13),  # neck -> left_collar
    (12, 14),  # neck -> right_collar
    (13, 15),  # left_collar -> head
    (14, 16),  # right_collar -> head
    (15, 17),  # head -> left_shoulder
    (16, 18),  # head -> right_shoulder
    (17, 19),  # left_shoulder -> left_elbow
    (18, 20),  # right_shoulder -> right_elbow
    (19, 21),  # left_elbow -> left_wrist
    (20, 22),  # right_elbow -> right_wrist
    (21, 23),  # left_wrist -> left_hand
    (22, 24),  # right_wrist -> right_hand
]
SMPL_CONNECTION_NAMES = [
    "pelvis->left_hip",
    "pelvis->right_hip",
    "pelvis->spine1",
    "left_hip->left_knee",
    "right_hip->right_knee",
    "spine1->spine2",
    "left_knee->left_ankle",
    "right_knee->right_ankle",
    "spine2->spine3",
    "left_ankle->left_foot",
    "right_ankle->right_foot",
    "spine3->neck",
    "neck->left_collar",
    "neck->right_collar",
    "left_collar->head",
    "right_collar->head",
    "head->left_shoulder",
    "head->right_shoulder",
    "left_shoulder->left_elbow",
    "right_shoulder->right_elbow",
    "left_elbow->left_wrist",
    "right_elbow->right_wrist",
    "left_wrist->left_hand",
    "right_wrist->right_hand",
]

# for 4D-Humans, See https://github.com/shubham-goel/4D-Humans/issues/45
SPIN_KEYPOINT_NAMES = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "MidHip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
    "LBigToe",
    "LSmallToe",
    "LHeel",
    "RBigToe",
    "RSmallToe",
    "RHeel",
    # 24 Ground Truth joints (superset of joints from different datasets)
    # 'Right Ankle',
    # 'Right Knee',
    # 'Right Hip',
    # 'Left Hip',
    # 'Left Knee',
    # 'Left Ankle',
    # 'Right Wrist',
    # 'Right Elbow',
    # 'Right Shoulder',
    # 'Left Shoulder',
    # 'Left Elbow',
    # 'Left Wrist',
    # 'Neck (LSP)',
    # 'Top of Head (LSP)',
    # 'Pelvis (MPII)',
    # 'Thorax (MPII)',
    # 'Spine (H36M)',
    # 'Jaw (H36M)',
    # 'Head (H36M)',
]
SPIN_JOINTS_CONNECTION = [
    (0, 1),  # Nose -> Neck
    (1, 2),  # Neck -> RShoulder
    (2, 3),  # RShoulder -> RElbow
    (3, 4),  # RElbow -> RWrist
    (1, 5),  # Neck -> LShoulder
    (5, 6),  # LShoulder -> LElbow
    (6, 7),  # LElbow -> LWrist
    (1, 8),  # Neck -> MidHip
    (8, 9),  # MidHip -> RHip
    (9, 10),  # RHip -> RKnee
    (10, 11),  # RKnee -> RAnkle
    (8, 12),  # MidHip -> LHip
    (12, 13),  # LHip -> LKnee
    (13, 14),  # LKnee -> LAnkle
    (0, 15),  # Nose -> REye
    (0, 16),  # Nose -> LEye
    (15, 17),  # REye -> REar
    (16, 18),  # LEye -> LEar
    (14, 19),  # LAnkle -> LBigToe
    (19, 20),  # LBigToe -> LSmallToe
    (20, 21),  # LSmallToe -> LHeel
    (11, 22),  # RAnkle -> RBigToe
    (22, 23),  # RBigToe -> RSmallToe
    (23, 24),  # RSmallToe -> RHeel
]
SPIN_CONNECTION_NAMES = [
    "Nose->Neck",
    "Neck->RShoulder",
    "RShoulder->RElbow",
    "RElbow->RWrist",
    "Neck->LShoulder",
    "LShoulder->LElbow",
    "LElbow->LWrist",
    "Neck->MidHip",
    "MidHip->RHip",
    "RHip->RKnee",
    "RKnee->RAnkle",
    "MidHip->LHip",
    "LHip->LKnee",
    "LKnee->LAnkle",
    "Nose->REye",
    "Nose->LEye",
    "REye->REar",
    "LEye->LEar",
    "LAnkle->LBigToe",
    "LBigToe->LSmallToe",
    "LSmallToe->LHeel",
    "RAnkle->RBigToe",
    "RBigToe->RSmallToe",
    "RSmallToe->RHeel",
]
