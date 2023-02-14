import numpy as np
import pathlib
import os

CWD = pathlib.Path(os.path.abspath(__file__)).parent
GITROOT = CWD.parent
OP_DATA_DIR = GITROOT.parent / 'ossicles_6D_pose_estimation' / 'data'
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

IMAGE_PATH_6108 = OP_DATA_DIR / "frames" / "CIP.6108.1638408845868_video_trim" / "CIP.6108.1638408845868_video_trim_0.png"
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
OSSICLES_MESH_PATH_6087_left = OP_DATA_DIR / "surgical_planning" / "CIP.6087.8415865242263_video_trim" / "mesh" / "6087_left_ossicles.mesh"
FACIAL_NERVE_MESH_PATH_6087_left = OP_DATA_DIR / "surgical_planning"/ "CIP.6087.8415865242263_video_trim" / "mesh" / "6087_left_facial_nerve.mesh"
CHORDA_MESH_PATH_6087_left = OP_DATA_DIR / "surgical_planning"/ "CIP.6087.8415865242263_video_trim" / "mesh" / "6087_left_chorda_centered.mesh"

# left2right ossicles
IMAGE_PATH_6742 = OP_DATA_DIR / "frames" / "CIP.6742.8381574350255_video_trim" / "CIP.6742.8381574350255_video_trim_0.png"
OSSICLES_MESH_PATH_6742_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6742.8381574350255_video_trim" / "mesh" / "6742_left_ossicles_processed_right.mesh"
FACIAL_NERVE_MESH_PATH_6742_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6742.8381574350255_video_trim" / "mesh" / "6742_left_facial_nerve_processed_right.mesh"
CHORDA_MESH_PATH_6742_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6742.8381574350255_video_trim" / "mesh" / "6742_left_chorda_processed_right.mesh"

# right ossicles
# gt_pose_455_right = np.eye(4)
gt_pose_455_right = np.array([[ 0.38100506, -0.036584,   -0.92384888, -4.78796069],
                            [ 0.68508419,  0.68217898,  0.25552199, -2.45344322],
                            [ 0.62088227, -0.73026943,  0.28497676,  0.01822901],
                            [ 0.,          0.,          0.,          1.        ]])
gt_pose_5997_right = np.eye(4)
gt_pose_6088_right = np.eye(4)
gt_pose_6108_right = np.eye(4)
gt_pose_632_right = np.eye(4)
gt_pose_6320_right = np.eye(4)
gt_pose_6329_right = np.eye(4)
gt_pose_6602_right = np.eye(4)
gt_pose_6751_right = np.eye(4)

# left ossicles
# gt_pose_6742_left = np.eye(4)
gt_pose_6742_left = np.array([[-0.0970554,   0.20898817,  0.97309003,  0.35713253],
                            [-0.73955466,  0.63915871, -0.21103328, -3.68900679],
                            [-0.66606242, -0.74013518,  0.09252437, 38.38261253],
                            [ 0.,          0.,          0.,          1.        ]])


gt_pose_6087_left = np.eye(4)

# left2right ossicles
# gt_pose_6742_right = np.eye(4)
gt_pose_6742_right = np.array([[-0.0970554,   0.20898817,  0.97309003,  0.35713253],
                            [-0.73955466,  0.63915871, -0.21103328, -3.68900679],
                            [-0.66606242, -0.74013518,  0.09252437, 38.38261253],
                            [ 0.,          0.,          0.,          1.        ]]) @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

# hand draw masks
mask_5997_hand_draw_numpy = DATA_DIR / "hand_draw_mask" / "mask_5997_hand_draw_numpy.npy"
mask_6088_hand_draw_numpy = DATA_DIR / "hand_draw_mask" / "mask_6088_hand_draw_numpy.npy"
mask_6108_hand_draw_numpy = DATA_DIR / "hand_draw_mask" / "mask_6108_hand_draw_numpy.npy"
mask_6742_hand_draw_numpy = DATA_DIR / "hand_draw_mask" / "mask_6742_hand_draw_numpy.npy"

# gt_pose_455_right = np.array([[  0.36189961,   0.31967712,  -0.87569128,   5.33823202],
#                             [  0.40967285,   0.78925644,   0.45743024, -32.5816239 ],
#                             [  0.83737496,  -0.52429077,   0.15466856,  12.5083066 ],
#                             [  0.,           0.,           0.,           1.        ]])

# gt_pose_5997_right = np.array([[  -0.14498174,   -0.34676691,   -0.92667849,   29.36436624],
#                         [   0.33413722,    0.86439266,   -0.3757361,   -13.54538251],
#                         [   0.93130693,   -0.36411267,   -0.00945343, -104.0636636 ],
#                         [   0.,            0.,            0.,            1.        ]])


# gt_pose_6088_right = np.array([[  0.36049218,  -0.12347807,  -0.93605796,  17.37936422],
#                         [  0.31229879,   0.96116227,  -0.00651795, -27.17513405],
#                         [  0.89102231,  -0.28692541,   0.38099733, -19.1631882 ],
#                         [  0.,           0.,           0.,           1.        ]])


# gt_pose_6108_right = np.array([[  0.20755796,   0.33304378,  -0.9197834,   10.89388084],
#                         [  0.61199071,   0.68931778,   0.38769624, -36.58529423],
#                         [  0.76314289,  -0.64336834,  -0.06074633, 229.45832825],
#                         [  0.,           0.,           0.,           1.        ]]) #  GT pose

# gt_pose_6742_left = np.array([[ -0.00205008,  -0.27174699,   0.96236655, -18.75660285],
#                         [ -0.4431008,    0.86298269,   0.24273971, -13.34068231],
#                         [ -0.89646944,  -0.42592774,  -0.1221805,  458.83536963],
#                         [  0.,           0.,           0.,           1.        ]]) #  GT pose

# gt_pose_6742_right = np.eye(4)

# gt_pose_632_right = np.array([[  0.01213903,  -0.23470041,  -0.97199196,  23.83199935],
#                             [  0.7709136,    0.62127575,  -0.14038752, -19.05412711],
#                             [  0.63682404,  -0.74761766,   0.18847542, 602.2021275 ],
#                             [  0.,           0.,           0.,           1.        ]])

# gt_pose_6087_left = np.array([[  0.20370912,  -0.21678892,   0.95472779, -20.79224732],
#                                 [ -0.62361071,   0.72302221,   0.29723487,  -4.9027381 ],
#                                 [ -0.75472663,  -0.65592793,   0.01209432,  30.25550556],
#                                 [  0.,           0.,           0.,           1.        ]])

# gt_pose_6320_right = np.array([[  0.13992712,  -0.06843788,  -0.98779384,  19.45358842],
#                             [  0.50910393,   0.86061441,   0.01249129, -27.0485824 ],
#                             [  0.84925473,  -0.50463759,   0.15526529, 305.97544605],
#                             [  0.,           0.,           0.,           1.        ]])

# gt_pose_6329_right = np.array([[  0.09418739,   0.38382761,  -0.91858865,   7.3397235 ],
#                                 [  0.66883124,   0.65905617,   0.34396181, -33.31646256],
#                                 [  0.73742355,  -0.64677765,  -0.19464113, 302.59170409],
#                                 [  0.,           0.,           0.,           1.        ]])

# gt_pose_6602_right = np.array([[  0.22534241,   0.44679594,  -0.86579107,   3.33442317],
#                                 [  0.49393868,   0.7135863,    0.496809,   -28.9554841 ],
#                                 [  0.83978889,  -0.53959983,  -0.05988854, 299.38210116],
#                                 [  0.,           0.,           0.,           1.        ]])

# gt_pose_6751_right = np.array([[  0.14325502,  -0.47155627,  -0.87012222,  22.98132783],
#                                 [  0.88314596,   0.45772661,  -0.10266232, -23.0138221 ],
#                                 [  0.44668916,  -0.75373804,   0.48202465, 137.22705977],
#                                 [  0.,           0.,           0.,           1.        ]])