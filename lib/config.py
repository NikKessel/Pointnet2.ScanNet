import os
from easydict import EasyDict

CONF = EasyDict()

# BASE PATH
CONF.ROOT = r"D:\Pointnet2.ScanNet" # TODO change this
CONF.SCANNET_DIR =  r"D:\Scannet\scans" # TODO change this

# Uncomment the followings if you're NOT on slurm
CONF.SCANNET_FRAMES_ROOT = os.path.join(CONF.ROOT, "frames_square")
CONF.PROJECTION = os.path.join(CONF.ROOT, "multiview_projection_pointnet")
CONF.ENET_FEATURES_ROOT = os.path.join(CONF.ROOT, "enet_features")

# Uncomment the followings if you're on slurm
#CONF.CLUSTER = "/cluster/balrog/dchen/Pointnet2.ScanNet"
#CONF.SCANNET_FRAMES_ROOT = os.path.join(CONF.CLUSTER, "frames_square")
#CONF.PROJECTION = os.path.join(CONF.CLUSTER, "multiview_projection_pointnet")
#CONF.ENET_FEATURES_ROOT = os.path.join(CONF.CLUSTER, "enet_features")

CONF.ENET_FEATURES_SUBROOT = os.path.join(CONF.ENET_FEATURES_ROOT, "{}") # scene_id
CONF.ENET_FEATURES_PATH = os.path.join(CONF.ENET_FEATURES_SUBROOT, "{}.npy") # frame_id
CONF.SCANNET_FRAMES = os.path.join(CONF.SCANNET_FRAMES_ROOT, "{}/{}") # scene_id, mode 
CONF.SCENE_NAMES = sorted(os.listdir(CONF.SCANNET_DIR))

CONF.PREP = os.path.join(CONF.ROOT, "preprocessing")
CONF.PREP_SCANS = os.path.join(CONF.PREP, "scannet_scenes")
CONF.SCAN_LABELS = os.path.join(CONF.PREP, "label_point_clouds")
CONF.OUTPUT_ROOT = os.path.join(CONF.ROOT, "outputs")
CONF.ENET_WEIGHTS = os.path.join(CONF.ROOT, "data/scannetv2_enet.pth")
CONF.MULTIVIEW = os.path.join(CONF.PREP_SCANS, "enet_feats.hdf5")

CONF.SCANNETV2_TRAIN = os.path.join(CONF.ROOT, "data/scannetv2_train.txt")
CONF.SCANNETV2_VAL = os.path.join(CONF.ROOT, "data/scannetv2_val.txt")
CONF.SCANNETV2_TEST = os.path.join(CONF.ROOT, "data/scannetv2_test.txt")   
CONF.SCANNETV2_LIST = os.path.join(CONF.ROOT, "data/scannetv2.txt")   
CONF.SCANNETV2_FILE = os.path.join(CONF.PREP_SCANS, "{}.npy") # scene_id   
CONF.SCANNETV2_LABEL = os.path.join(CONF.SCAN_LABELS, "{}.ply") # scene_id   
   
CONF.NYUCLASSES = [   
    'floor',    
    'wall',    
    'cabinet',    
    'bed',    
    'chair',    
    'sofa',    
    'table',    
    'door', 
    'window', 
    'bookshelf', 
    'picture', 
    'counter', 
    'desk', 
    'curtain', 
    'refrigerator', 
    'bathtub', 
    'shower curtain', 
    'toilet', 
    'sink', 
    'otherprop',
    'book',               # New class
    'trash can',           # New class
    'box',                 # New class
    'shelf',               # New class
    'mirror',              # New class
    'plant',               # New class
    'whiteboard',          # New class
    'keyboard',            # New class
    'tv',                  # New class
    'computer tower',      # New class
    'telephone',           # New class
    'refrigerator',        # Already exists
    'microwave',           # New class
    'laptop',              # New class
    'printer',             # New class
    'soap dispenser',      # New class
    'light',               # New class
    'fan',                 # New class
    'ceiling light',       # New class
    'clock',               # New class
    'rail',                # New class
    'bulletin board',      # New class
    'trash bin',           # New class
    'mouse',               # New class
    'person',              # New class
    'fire extinguisher',   # New class
    'ladder',              # New class
    'pipe',                # New class
    'bookshelf',           # Already exists
    'projector screen',    # New class
    'fire alarm',          # New class
    'projector',           # New class
    'smoke detector',      # New class
    'heater',              # New class
    'scanner',             # New class
    'stair',               # New class
    'car'                  # New class
]

# Update the number of classes based on the new list
CONF.NUM_CLASSES = len(CONF.NYUCLASSES)

# Update the palette with new colors for new classes
# You can customize these colors to your preference, but I'm adding placeholder RGB values here
CONF.PALETTE = [
    (152, 223, 138),        # floor
    (174, 199, 232),        # wall
    (31, 119, 180),         # cabinet
    (255, 187, 120),        # bed
    (188, 189, 34),         # chair
    (140, 86, 75),          # sofa
    (255, 152, 150),        # table
    (214, 39, 40),          # door
    (197, 176, 213),        # window
    (148, 103, 189),        # bookshelf
    (196, 156, 148),        # picture
    (23, 190, 207),         # counter
    (247, 182, 210),        # desk
    (219, 219, 141),        # curtain
    (255, 127, 14),         # refrigerator
    (227, 119, 194),        # bathtub
    (158, 218, 229),        # shower curtain
    (44, 160, 44),          # toilet
    (112, 128, 144),        # sink
    (82, 84, 163),          # otherfurn
    (60, 180, 75),          # book (new color)
    (0, 128, 128),          # trash can (new color)
    (255, 99, 71),          # box (new color)
    (238, 130, 238),        # shelf (new color)
    (127, 255, 0),          # mirror (new color)
    (173, 216, 230),        # plant (new color)
    (245, 222, 179),        # whiteboard (new color)
    (255, 228, 181),        # keyboard (new color)
    (255, 255, 0),          # tv (new color)
    (139, 69, 19),          # computer tower (new color)
    (218, 165, 32),         # telephone (new color)
    (255, 127, 80),         # microwave (new color)
    (0, 255, 255),          # laptop (new color)
    (255, 215, 0),          # printer (new color)
    (123, 104, 238),        # soap dispenser (new color)
    (72, 209, 204),         # light (new color)
    (139, 0, 139),          # fan (new color)
    (255, 140, 0),          # ceiling light (new color)
    (65, 105, 225),         # clock (new color)
    (255, 69, 0),           # rail (new color)
    (50, 205, 50),          # bulletin board (new color)
    (199, 21, 133),         # trash bin (new color)
    (255, 20, 147),         # mouse (new color)
    (72, 61, 139),          # person (new color)
    (255, 165, 0),          # fire extinguisher (new color)
    (32, 178, 170),         # ladder (new color)
    (47, 79, 79),           # pipe (new color)
    (127, 255, 212),        # projector screen (new color)
    (189, 183, 107),        # fire alarm (new color)
    (144, 238, 144),        # projector (new color)
    (255, 69, 0),           # smoke detector (new color)
    (139, 0, 0),            # heater (new color)
    (255, 20, 147),         # scanner (new color)
    (0, 0, 128),            # stair (new color)
    (135, 206, 250)         # car (new color)
]
