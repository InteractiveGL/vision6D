import numpy as np
import pathlib
import os

CWD = pathlib.Path(os.path.abspath(__file__)).parent
GITROOT = CWD.parent
OP_DATA_DIR = GITROOT.parent / 'ossicles_6D_pose_estimation' / 'data'
YOLOV8_DATA_DIR = GITROOT.parent / 'yolov8'
DATA_DIR = GITROOT / 'data'

# right ossicles
IMAGE_PATH_455 = OP_DATA_DIR / "frames" /"CIP.455.8381493978235_video_trim" / "CIP.455.8381493978235_video_trim_0.png"
OSSICLES_MESH_PATH_455_right = OP_DATA_DIR / "surgical_planning" / "CIP.455.8381493978235_video_trim" / "mesh" / "455_right_ossicles_processed.mesh"
FACIAL_NERVE_MESH_PATH_455_right = OP_DATA_DIR / "surgical_planning"/ "CIP.455.8381493978235_video_trim" / "mesh" / "455_right_facial_nerve_processed.mesh"
CHORDA_MESH_PATH_455_right = OP_DATA_DIR / "surgical_planning"/ "CIP.455.8381493978235_video_trim" / "mesh" / "455_right_chorda_processed.mesh"

IMAGE_PATH_5997 = OP_DATA_DIR / "frames" /"CIP.5997.8381493978443_video_trim" / "CIP.5997.8381493978443_video_trim_0.png"
OSSICLES_MESH_PATH_5997_right = OP_DATA_DIR / "surgical_planning" / "CIP.5997.8381493978443_video_trim" / "mesh" / "5997_right_ossicles_processed.mesh"
FACIAL_NERVE_MESH_PATH_5997_right = OP_DATA_DIR / "surgical_planning"/ "CIP.5997.8381493978443_video_trim" / "mesh" / "5997_right_facial_nerve_processed.mesh"
CHORDA_MESH_PATH_5997_right = OP_DATA_DIR / "surgical_planning"/ "CIP.5997.8381493978443_video_trim" / "mesh" / "5997_right_chorda_processed.mesh"

IMAGE_PATH_6088 = OP_DATA_DIR / "frames" / "CIP.6088.1681356523312_video_trim" / "CIP.6088.1681356523312_video_trim_0.png"
OSSICLES_MESH_PATH_6088_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6088.1681356523312_video_trim" / "mesh" / "6088_right_ossicles_processed.mesh"
FACIAL_NERVE_MESH_PATH_6088_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6088.1681356523312_video_trim" / "mesh" / "6088_right_facial_nerve_processed.mesh"
CHORDA_MESH_PATH_6088_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6088.1681356523312_video_trim" / "mesh" / "6088_right_chorda_processed.mesh"

IMAGE_PATH_6108 = YOLOV8_DATA_DIR / "runs" / "segment" / "CIP.6108.1638408845868_video_trim_right" / "images" / "ossicles" / "CIP.6108.1638408845868_video_trim2.png"
OSSICLES_MESH_PATH_6108_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6108.1638408845868_video_trim" / "mesh" / "6108_right_ossicles_processed.mesh"
FACIAL_NERVE_MESH_PATH_6108_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6108.1638408845868_video_trim" / "mesh" / "6108_right_facial_nerve_processed.mesh"
CHORDA_MESH_PATH_6108_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6108.1638408845868_video_trim" / "mesh" / "6108_right_chorda_processed.mesh"

OSSICLES_MESH_PATH_6742_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6742.8381574350255_video_trim" / "mesh" / "6742_right_ossicles_processed.mesh"
FACIAL_NERVE_MESH_PATH_6742_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6742.8381574350255_video_trim" / "mesh" / "6742_right_facial_nerve_processed.mesh"
CHORDA_MESH_PATH_6742_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6742.8381574350255_video_trim" / "mesh" / "6742_right_chorda_processed.mesh"

IMAGE_PATH_632 = OP_DATA_DIR / "frames" /"CIP.632.8381493978443_video_trim" / "CIP.632.8381493978443_video_trim_0.png"
OSSICLES_MESH_PATH_632_right = OP_DATA_DIR / "surgical_planning" / "CIP.632.8381493978443_video_trim" / "mesh" / "632_right_ossicles_processed.mesh"
FACIAL_NERVE_MESH_PATH_632_right = OP_DATA_DIR / "surgical_planning"/ "CIP.632.8381493978443_video_trim" / "mesh" / "632_right_facial_nerve_processed.mesh"
CHORDA_MESH_PATH_632_right = OP_DATA_DIR / "surgical_planning"/ "CIP.632.8381493978443_video_trim" / "mesh" / "632_right_chorda_processed.mesh"

IMAGE_PATH_6320 = OP_DATA_DIR / "frames" /"CIP.6320.5959167268122_video_trim" / "CIP.6320.5959167268122_video_trim_0.png"
OSSICLES_MESH_PATH_6320_right = OP_DATA_DIR / "surgical_planning" / "CIP.6320.5959167268122_video_trim" / "mesh" / "6320_right_ossicles_processed.mesh"
FACIAL_NERVE_MESH_PATH_6320_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6320.5959167268122_video_trim" / "mesh" / "6320_right_facial_nerve_processed.mesh"
CHORDA_MESH_PATH_6320_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6320.5959167268122_video_trim" / "mesh" / "6320_right_chorda_processed.mesh"

IMAGE_PATH_6329 = OP_DATA_DIR / "frames" /"CIP.6329.6010707468438_vieo_trim" / "CIP.6329.6010707468438_vieo_trim_0.png"
OSSICLES_MESH_PATH_6329_right = OP_DATA_DIR / "surgical_planning" / "CIP.6329.6010707468438_vieo_trim" / "mesh" / "6329_right_ossicles_processed.mesh"
FACIAL_NERVE_MESH_PATH_6329_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6329.6010707468438_vieo_trim" / "mesh" / "6329_right_facial_nerve_processed.mesh"
CHORDA_MESH_PATH_6329_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6329.6010707468438_vieo_trim" / "mesh" / "6329_right_chorda_processed.mesh"

IMAGE_PATH_6602 = OP_DATA_DIR / "frames" /"CIP.6602.7866163404219_video_trim" / "CIP.6602.7866163404219_video_trim_0.png"
OSSICLES_MESH_PATH_6602_right = OP_DATA_DIR / "surgical_planning" / "CIP.6602.7866163404219_video_trim" / "mesh" / "6602_right_ossicles_processed.mesh"
FACIAL_NERVE_MESH_PATH_6602_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6602.7866163404219_video_trim" / "mesh" / "6602_right_facial_nerve_processed.mesh"
CHORDA_MESH_PATH_6602_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6602.7866163404219_video_trim" / "mesh" / "6602_right_chorda_processed.mesh"

IMAGE_PATH_6751 = OP_DATA_DIR / "frames" /"CIP.6751.1844636424484_video_trim" / "CIP.6751.1844636424484_video_trim_0.png"
OSSICLES_MESH_PATH_6751_right = OP_DATA_DIR / "surgical_planning" / "CIP.6751.1844636424484_video_trim" / "mesh" / "6751_right_ossicles_processed.mesh"
FACIAL_NERVE_MESH_PATH_6751_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6751.1844636424484_video_trim" / "mesh" / "6751_right_facial_nerve_processed.mesh"
CHORDA_MESH_PATH_6751_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6751.1844636424484_video_trim" / "mesh" / "6751_right_chorda_processed.mesh"

# left ossicles
IMAGE_PATH_6742 = OP_DATA_DIR / "frames" / "CIP.6742.8381574350255_video_trim" / "CIP.6742.8381574350255_video_trim_0.png"
OSSICLES_MESH_PATH_6742_left = OP_DATA_DIR / "surgical_planning"/ "CIP.6742.8381574350255_video_trim" / "mesh" / "6742_left_ossicles_processed.mesh"
FACIAL_NERVE_MESH_PATH_6742_left = OP_DATA_DIR / "surgical_planning"/ "CIP.6742.8381574350255_video_trim" / "mesh" / "6742_left_facial_nerve_processed.mesh"
CHORDA_MESH_PATH_6742_left = OP_DATA_DIR / "surgical_planning"/ "CIP.6742.8381574350255_video_trim" / "mesh" / "6742_left_chorda_processed.mesh"

IMAGE_PATH_6087 = OP_DATA_DIR / "frames" /"CIP.6087.8415865242263_video_trim" / "CIP.6087.8415865242263_video_trim_0.png"
OSSICLES_MESH_PATH_6087_left = OP_DATA_DIR / "surgical_planning" / "CIP.6087.8415865242263_video_trim" / "mesh" / "6087_left_ossicles_centered.mesh"
FACIAL_NERVE_MESH_PATH_6087_left = OP_DATA_DIR / "surgical_planning"/ "CIP.6087.8415865242263_video_trim" / "mesh" / "6087_left_facial_nerve_centered.mesh"
CHORDA_MESH_PATH_6087_left = OP_DATA_DIR / "surgical_planning"/ "CIP.6087.8415865242263_video_trim" / "mesh" / "6087_left_chorda_centered.mesh"

# right ossicles
gt_pose_455_right = np.array([[ 0.03575464,  0.10393758, -0.99394093, -5.59835303],
                                [ 0.32682429,  0.93867174,  0.10991473, -3.63446021],
                                [ 0.94440854, -0.328774,   -0.00040746, 17.88284243],
                                [ 0.,          0.,          0.,          1.        ]])
gt_pose_5997_right = np.array([[  -0.00957227,   -0.3024023,    -0.95313232,   -2.80850298],
                            [   0.26782242,    0.91757626,   -0.29381108,   -1.1994414 ],
                            [   0.96342074,   -0.25808264,    0.07220682, -123.25325947],
                            [   0.,            0.,            0.,            1.        ]])
gt_pose_6088_right = np.array([[ 0.15329824,  0.02588047, -0.987841,   -3.96729915],
                                [ 0.54896258,  0.82898159,  0.10690934, -1.73777328],
                                [ 0.82166887, -0.55867676,  0.11287402, -8.0172838 ],
                                [ 0.,          0.,          0.,          1.        ]])
gt_pose_6108_right = np.array([[-0.18876861, -0.20334526, -0.9607378 , -2.24840364],
                            [ 0.36843877,  0.89219498, -0.26122975, -4.57413415],
                            [ 0.91028528, -0.40328503, -0.09349813, 40.47957643],
                            [ 0.        ,  0.        ,  0.        ,  1.        ]])
gt_pose_632_right = np.array([[ -0.01120854,  -0.07776647,  -0.99690859,  -3.81118655],
                            [  0.30193174,   0.95017292,  -0.07751544,  -1.11610844],
                            [  0.95326365,  -0.30186718,   0.01283011, 456.68699975],
                            [  0.,           0.,           0.,           1.        ]])
gt_pose_6320_right = np.array([[  0.01569333,   0.01785505,  -0.99971742,  -6.34585943],
                                [  0.49459745,   0.86881031,   0.02328111,  -2.67564713],
                                [  0.86898048,  -0.49482305,   0.00480346, 248.85133073],
                                [  0.,           0.,           0.,           1.        ]])
gt_pose_6329_right = np.array([[  0.00999926,   0.06594057,  -0.99777345,  -4.7735089 ],
                            [  0.44806811,   0.89174682,   0.06342386,  -6.21329688],
                            [  0.8939435,   -0.44770465,  -0.02062905, 189.88310478],
                            [  0.,           0.,           0.,           1.        ]])
gt_pose_6602_right = np.array([[  0.21515756,   0.06148525,  -0.97464188,  -6.29202817],
                            [  0.41045512,   0.89989208,   0.14737989,  -3.14477412],
                            [  0.8861342,   -0.43175665,   0.16838165, 321.69488304],
                            [  0.,           0.,           0.,           1.        ]])
gt_pose_6751_right = np.array([[  0.18732852,  -0.2824477,   -0.94081418,  -4.88357211],
                            [  0.49525699,   0.85428417,  -0.15785778,  -3.6422948 ],
                            [  0.84830923,  -0.43637353,   0.29991599, 160.70069836],
                            [  0.,           0.,           0.,           1.        ]])

# left ossicles
gt_pose_6742_left = np.array([[-0.0970554,   0.20898817,  0.97309003,  0.35713253],
                            [-0.73955466,  0.63915871, -0.21103328, -3.68900679],
                            [-0.66606242, -0.74013518,  0.09252437, 38.38261253],
                            [ 0.,          0.,          0.,          1.        ]])
gt_pose_6087_left = np.eye(4)

# hand draw masks
mask_5997_hand_draw_numpy = DATA_DIR / "hand_draw_mask" / "mask_5997_hand_draw_numpy.npy"
mask_6088_hand_draw_numpy = DATA_DIR / "hand_draw_mask" / "mask_6088_hand_draw_numpy.npy"
mask_6108_hand_draw_numpy = DATA_DIR / "hand_draw_mask" / "mask_6108_hand_draw_numpy.npy"
mask_6742_hand_draw_numpy = DATA_DIR / "hand_draw_mask" / "mask_6742_hand_draw_numpy.npy"