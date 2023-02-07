import numpy as np
import pathlib
import os

CWD = pathlib.Path(os.path.abspath(__file__)).parent
GITROOT = CWD.parent
OP_DATA_DIR = GITROOT.parent / 'ossicles_6D_pose_estimation' / 'data'
DATA_DIR = GITROOT / 'data'

OSSICLES_MESH_PATH_5997_right = OP_DATA_DIR / "surgical_planning" / "CIP.5997.8381493978443_video_trim" / "mesh" / "5997_right_ossicles.mesh"
FACIAL_NERVE_MESH_PATH_5997_right = OP_DATA_DIR / "surgical_planning"/ "CIP.5997.8381493978443_video_trim" / "mesh" / "5997_right_facial_nerve.mesh"
CHORDA_MESH_PATH_5997_right = OP_DATA_DIR / "surgical_planning"/ "CIP.5997.8381493978443_video_trim" / "mesh" / "5997_right_chorda.mesh"

OSSICLES_MESH_PATH_6088_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6088.1681356523312_video_trim" / "mesh" / "6088_right_ossicles.mesh"
FACIAL_NERVE_MESH_PATH_6088_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6088.1681356523312_video_trim" / "mesh" / "6088_right_facial_nerve.mesh"
CHORDA_MESH_PATH_6088_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6088.1681356523312_video_trim" / "mesh" / "6088_right_facial_nerve.mesh"

OSSICLES_MESH_PATH_6108_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6108.1638408845868_video_trim" / "mesh" / "6108_right_ossicles.mesh"
FACIAL_NERVE_MESH_PATH_6108_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6108.1638408845868_video_trim" / "mesh" / "6108_right_facial_nerve.mesh"
CHORDA_MESH_PATH_6108_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6108.1638408845868_video_trim" / "mesh" / "6108_right_chorda.mesh"

OSSICLES_MESH_PATH_6742_left = OP_DATA_DIR / "surgical_planning"/ "CIP.6742.8381574350255_video_trim" / "mesh" / "6742_left_ossicles.mesh"
FACIAL_NERVE_MESH_PATH_6742_left = OP_DATA_DIR / "surgical_planning"/ "CIP.6742.8381574350255_video_trim" / "mesh" / "6742_left_facial_nerve.mesh"
CHORDA_MESH_PATH_6742_left = OP_DATA_DIR / "surgical_planning"/ "CIP.6742.8381574350255_video_trim" / "mesh" / "6742_left_chorda.mesh"

OSSICLES_MESH_PATH_6742_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6742.8381574350255_video_trim" / "mesh" / "6742_right_ossicles.mesh"
FACIAL_NERVE_MESH_PATH_6742_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6742.8381574350255_video_trim" / "mesh" / "6742_right_facial_nerve.mesh"
CHORDA_MESH_PATH_6742_right = OP_DATA_DIR / "surgical_planning"/ "CIP.6742.8381574350255_video_trim" / "mesh" / "6742_right_chorda.mesh"

mask_5997_hand_draw_numpy = DATA_DIR / "hand_draw_mask" / "mask_5997_hand_draw_numpy.npy"
mask_6088_hand_draw_numpy = DATA_DIR / "hand_draw_mask" / "mask_6088_hand_draw_numpy.npy"
mask_6108_hand_draw_numpy = DATA_DIR / "hand_draw_mask" / "mask_6108_hand_draw_numpy.npy"
mask_6742_hand_draw_numpy = DATA_DIR / "hand_draw_mask" / "mask_6742_hand_draw_numpy.npy"

gt_pose_5997_right = np.array([[  -0.14498174,   -0.34676691,   -0.92667849,   29.36436624],
                        [   0.33413722,    0.86439266,   -0.3757361,   -13.54538251],
                        [   0.93130693,   -0.36411267,   -0.00945343, -104.0636636 ],
                        [   0.,            0.,            0.,            1.        ]])


gt_pose_6088_right = np.array([[  0.36049218,  -0.12347807,  -0.93605796,  17.37936422],
                        [  0.31229879,   0.96116227,  -0.00651795, -27.17513405],
                        [  0.89102231,  -0.28692541,   0.38099733, -19.1631882 ],
                        [  0.,           0.,           0.,           1.        ]])


gt_pose_6108_right = np.array([[  0.20755796,   0.33304378,  -0.9197834,   10.89388084],
                        [  0.61199071,   0.68931778,   0.38769624, -36.58529423],
                        [  0.76314289,  -0.64336834,  -0.06074633, 229.45832825],
                        [  0.,           0.,           0.,           1.        ]]) #  GT pose

gt_pose_6742_left = np.array([[ -0.00205008,  -0.27174699,   0.96236655, -18.75660285],
                        [ -0.4431008,    0.86298269,   0.24273971, -13.34068231],
                        [ -0.89646944,  -0.42592774,  -0.1221805,  458.83536963],
                        [  0.,           0.,           0.,           1.        ]]) #  GT pose

gt_pose_6742_right = np.eye(4)