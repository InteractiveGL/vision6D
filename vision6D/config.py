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

IMAGE_PATH_6108 = OP_DATA_DIR / "frames" / "CIP.6108.1638408845868_video_trim" / "CIP.6108.1638408845868_video_trim_0.png"
# IMAGE_PATH_6108 = YOLOV8_DATA_DIR / "runs" / "segment" / "CIP.6108.1638408845868_video_trim_right" / "images" / "ossicles" / "CIP.6108.1638408845868_video_trim2.png"
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
gt_pose_455_right = np.array([[  0.17173876,   0.02410131,  -0.98484766,  -4.55591751],
                            [  0.16926624,   0.98411177,   0.05360012,  -4.69918678],
                            [  0.97049201,  -0.17590668,   0.1649306,  -42.87675677],
                            [  0.,           0.,           0.,           1.        ]])
gt_pose_5997_right = np.array([[  -0.14498174,   -0.34676691,   -0.92667849,   -3.05033148],
                                [   0.33413722,    0.86439266,   -0.3757361,    -1.54732514],
                                [   0.93130693,   -0.36411267,   -0.00945343, -119.95253896],
                                [   0.,            0.,            0.,            1.        ]])
gt_pose_6088_right = np.array([[  0.33489846,  -0.11054591,  -0.94711592,  -3.94126091],
                                [  0.50463018,   0.87228607,   0.07662444,  -1.86101644],
                                [  0.80907188,  -0.49829965,   0.34424711, -19.11294255],
                                [  0.,           0.,           0.,           1.        ]])
gt_pose_6108_right = np.array([[ -0.0574247,   -0.2138052,   -0.97518703,  -1.54899177],
                            [  0.33897252,   0.91459581,  -0.22048151,  -5.36405508],
                            [  0.93904208,  -0.3432227,    0.01995368, 226.94335282],
                            [  0.,           0.,           0.,           1.        ]])
gt_pose_632_right = np.array([[  0.02819178,  -0.4119172,   -0.91078507,  -4.02890658],
                            [  0.67434175,   0.68042273,  -0.28685904,  -0.9163176 ],
                            [  0.73788103,  -0.60609333,   0.29695529, 662.67647298],
                            [  0.,           0.,           0.,           1.        ]])
gt_pose_6320_right = np.array([[  0.06985898,  -0.09355006,  -0.99316067,  -8.90010064],
                            [  0.66925667,   0.74267893,  -0.02288056,  -3.39902356],
                            [  0.73973997,  -0.663081,     0.11449178, 454.62081293],
                            [  0.,           0.,           0.,           1.        ]])
gt_pose_6329_right = np.array([[ -0.01477684,   0.0530045,   -0.99848494,  -4.49119053],
                                [  0.4105838,    0.91084231,   0.04227566,  -6.99620193],
                                [  0.91170312,  -0.40933703,  -0.03522216, 256.55591363],
                                [  0.,           0.,           0.,           1.        ]])
gt_pose_6602_right = np.array([[  0.21515756,   0.06148525,  -0.97464188,  -6.29202817],
                            [  0.41045512,   0.89989208,   0.14737989,  -3.14477412],
                            [  0.8861342,   -0.43175665,   0.16838165, 321.69488304],
                            [  0.,           0.,           0.,           1.        ]])
gt_pose_6751_right = np.array([[  0.20413225,  -0.24419054,  -0.94799842,  -4.80912376],
                            [  0.38805066,   0.90924235,  -0.1506487,   -4.34837084],
                            [  0.8987473,   -0.33711915,   0.28036399, 215.39152615],
                            [  0.,           0.,           0.,           1.        ]])

# left ossicles
gt_pose_6742_left = np.array([[ -0.34036243,   0.07007941,   0.93767921,  -1.4661713 ],
                            [  0.02377673,   0.9975414,   -0.06592275,  -5.3604821 ],
                            [ -0.93999365,  -0.00014267,  -0.34119188, 325.05241877],
                            [  0.,           0.,           0.,           1.        ]])
gt_pose_6087_left = np.array([[-0.23073306, -0.45187525,  0.8617256,   4.26468955],
                            [-0.36062739,  0.86226372,  0.35559692, -0.94730022],
                            [-0.90372017, -0.22871389, -0.36191133, 79.78194931],
                            [ 0.,          0.,          0.,          1.        ]])

# hand draw masks
mask_5997_hand_draw_numpy = DATA_DIR / "hand_draw_mask" / "mask_5997_hand_draw_numpy.npy"
mask_6088_hand_draw_numpy = DATA_DIR / "hand_draw_mask" / "mask_6088_hand_draw_numpy.npy"
mask_6108_hand_draw_numpy = DATA_DIR / "hand_draw_mask" / "mask_6108_hand_draw_numpy.npy"
mask_6742_hand_draw_numpy = DATA_DIR / "hand_draw_mask" / "mask_6742_hand_draw_numpy.npy"