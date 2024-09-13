import json
import numpy as np

def convert_to_4x4(cam_R_m2c, cam_t_m2c, scale=1e-3):
    # Convert rotation list to a 3x3 matrix
    R = np.array(cam_R_m2c).reshape(3, 3)
    
    # Scale translation vector
    t = np.array(cam_t_m2c) * scale
    
    # Create a 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R  # Rotation part
    T[:3, 3] = t   # Translation part
    
    return T

def compute_transformation_matrix(data):
    """
    Computes the 4x4 transformation matrix from the given JSON data.
    
    Parameters:
        data (dict): A dictionary containing 'cam_R_m2c', 'cam_t_m2c', and 'obj_id'.
        
    Returns:
        tuple: A tuple containing the 4x4 transformation matrix and the object ID.
    """
    # Extract rotation and translation
    cam_R_m2c = data.get('cam_R_m2c')
    cam_t_m2c = data.get('cam_t_m2c')
    obj_id = data.get('obj_id')
    
    if cam_R_m2c is None or cam_t_m2c is None or obj_id is None:
        raise ValueError("JSON data must contain 'cam_R_m2c', 'cam_t_m2c', and 'obj_id'.")

    # Reshape rotation matrix
    R = np.array(cam_R_m2c).reshape(3, 3)
    
    # Translation vector
    t = np.array(cam_t_m2c).reshape(3, 1) * 1e-3
    
    # Construct the 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    
    return T, obj_id

def process_lmo_scene_gt():
    with open('workspace/lmo_scene_gt.json') as f: data = json.load(f)

    category = 3
    # Process all objects in each category
    for category, objects in data.items():
        print(f"Category {category}:")
        if category != "3": continue
        for obj in objects:
            cam_R_m2c = obj["cam_R_m2c"]
            cam_t_m2c = obj["cam_t_m2c"]
            obj_id = obj["obj_id"]
            
            # Convert to 4x4 matrix
            transformation_matrix = convert_to_4x4(cam_R_m2c, cam_t_m2c)
            
            # Print with comma-separated values
            formatted_matrix = np.array2string(transformation_matrix, separator=', ')
            
            print(f"Object ID {obj_id}:")
            print(formatted_matrix)
        """
        Category 3:
        Object ID 1:
        [[ 0.87547542,  0.47867615, -0.06858282,  0.16168946],
        [ 0.3746311 , -0.76110463, -0.52974822, -0.11374033],
        [-0.30574968,  0.43803819, -0.84552592,  1.11283113],
        [ 0.        ,  0.        ,  0.        ,  1.        ]]
        Object ID 5:
        [[ 0.94893088,  0.30725587, -0.07208124,  0.13436598],
        [ 0.24200515, -0.85502122, -0.45872652,  0.04577287],
        [-0.20257109,  0.41784038, -0.88568011,  0.96478389],
        [ 0.        ,  0.        ,  0json_data = '''
    {
        "cam_R_m2c": [
            0.10129456886055979, 0.9189831447333948, -0.3810634986656045,
            -0.7756012795391442, -0.1669324403572813, -0.6087497150172678,
            -0.6230425872254604, 0.3572163770773676, 0.6958551533552954
        ],
        "cam_t_m2c": [
            -142.05746364858163, -220.30830799745368, 1466.9391337786517
        ],
        "obj_id": 10
    }
    '''

    # Load JSON data
    data = json.loads(json_data)
    
    # Compute transformation matrix and get obj_id
    T, obj_id = compute_transformation_matrix(data)
    
    # Display results
    print("4x4 Transformation Matrix:\n", T)
    print("\nObject ID:", obj_id).        ,  1.        ]]
        Object ID 8:
        [[ 0.4322479 ,  0.90168944, -0.04564078,  0.10211878],
        [ 0.78623622, -0.40077365, -0.47233169, -0.08897942],
        [-0.44376608,  0.16802271, -0.88134225,  1.03377192],
        [ 0.        ,  0.        ,  0.        ,  1.        ]]
        Object ID 9:
        [[-0.97252357, -0.22470979, -0.06116191,  0.06912369],
        [-0.17436893,  0.87669435, -0.44837068,  0.13383816],
        [ 0.15437147, -0.42537857, -0.89177123,  0.98419986],
        [ 0.        ,  0.        ,  0.        ,  1.        ]]
        Object ID 10:
        [[-0.01490196,  0.99987889,  0.00447346,  0.3832873 ],
        [ 0.88840196,  0.01529312, -0.45881469, -0.1579811 ],
        [-0.45882761, -0.0028639 , -0.88852059,  1.14798292],
        [ 0.        ,  0.        ,  0.        ,  1.        ]]
        Object ID 11:
        [[ 0.23897695,  0.97089011, -0.02381454, -0.00772225],
        [ 0.83098555, -0.21712037, -0.51239421, -0.14392245],
        [-0.5025906 ,  0.10264193, -0.85851171,  1.10075075],
        [ 0.        ,  0.        ,  0.        ,  1.        ]]
        Object ID 12:
        [[-0.470049 ,  0.881505 , -0.0454911, -0.0593367],
        [ 0.805976 ,  0.407628 , -0.429296 ,  0.120607 ],
        [-0.359872 , -0.238445 , -0.902048 ,  1.00097  ],
        [ 0.       ,  0.       ,  0.       ,  1.       ]]
        """

def process_scene_gt():
    json_data = '''
    {"cam_R_m2c": [-0.5333916925995245, 0.4594610258696043, 0.7102033990151456, 0.8373402634661522, 0.16788322425581503, 0.5202658034042862, 0.11981062325290513, 0.8721873587253609, -0.4742727335989573], "cam_t_m2c": [-178.80281681910716, -239.37622509866674, 1058.7328499019634], "obj_id": 33}
    '''

    # Load JSON data
    data = json.loads(json_data)
    
    # Compute transformation matrix and get obj_id
    T, obj_id = compute_transformation_matrix(data)
    
    # Print with comma-separated values
    formatted_matrix = np.array2string(T, separator=', ')
    
    print(f"Object ID {obj_id}:")
    print(formatted_matrix)

if __name__ == "__main__":
    process_scene_gt()