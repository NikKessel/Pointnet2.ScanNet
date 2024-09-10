import sys
sys.path.append(".")
from lib.config import CONF

g_label_names = CONF.NYUCLASSES

def get_raw2scannet_label_map():
    lines = [line.rstrip() for line in open('preprocessing/scannetv2-labels.combined.tsv')]
    lines = lines[1:]
    raw2scannet = {}
    for i in range(len(lines)):
        label_classes_set = set(g_label_names)
        elements = lines[i].split('\t')
        raw_name = elements[1]
        nyu40_name = elements[7]
        
        if nyu40_name not in label_classes_set:
            print(f"Unmapped label: {nyu40_name}, assigning to 'otherprop'")
            raw2scannet[raw_name] = 'otherprop'
        else:
            print(f"Mapped: {nyu40_name} to {raw_name}")
            raw2scannet[raw_name] = nyu40_name
    return raw2scannet



g_raw2scannet = get_raw2scannet_label_map()
