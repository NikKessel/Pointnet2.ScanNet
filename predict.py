import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import Counter
import sys
import os
import importlib
from collections import Counter
from PIL import Image
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


# Add the project root directory to Python path
sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../pointnet2/'))
sys.path.insert(0, r"D:\Pointnet2.ScanNet\pointnet2")

from lib.config import CONF

try:
    import open3d as o3d
except ImportError:
    print("Error: open3d is not installed. Please install it using 'pip install open3d'")
    sys.exit(1)

try:
    Pointnet = importlib.import_module("pointnet2_semseg")
except ImportError:
    print("Error: Could not import pointnet2_semseg. Please ensure the module exists and is in the Python path.")
    sys.exit(1)


NYUCLASSES = [
    'floor', 'wall', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
    'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'bathtub', 'shower curtain', 'toilet', 'sink',
    'otherprop', 'books', 'trash can', 'box', 'shelf', 'mirror', 'plant', 'whiteboard', 'keyboard', 'tv',
    'computer tower', 'person', 'telephone', 'microwave', 'laptop', 'printer', 'soap dispenser', 'light',
    'fan', 'ceiling light', 'clock', 'rail', 'bulletin board', 'trash bin', 'mouse', 'fire extinguisher',
    'ladder', 'pipe', 'projector screen', 'fire alarm', 'projector', 'smoke detector', 'heater', 'scanner',
    'stair', 'car'
]

NUM_CLASSES = len(NYUCLASSES)

PALETTE = [
    (152, 223, 138), (174, 199, 232), (31, 119, 180), (255, 187, 120), (188, 189, 34),
    (140, 86, 75), (255, 152, 150), (214, 39, 40), (197, 176, 213), (148, 103, 189),
    (196, 156, 148), (23, 190, 207), (247, 182, 210), (219, 219, 141), (255, 127, 14),
    (227, 119, 194), (158, 218, 229), (44, 160, 44), (112, 128, 144), (82, 84, 163),
    (200, 150, 100), (80, 200, 120), (250, 150, 150), (180, 180, 180), (120, 120, 200),
    (250, 200, 100), (120, 200, 200), (200, 100, 200), (250, 100, 100), (100, 250, 100),
    (200, 200, 100), (100, 100, 250), (250, 250, 100), (100, 250, 250), (250, 100, 250),
    (100, 200, 150), (150, 100, 200), (200, 150, 100), (150, 200, 100), (200, 100, 150),
    (100, 150, 200), (200, 200, 200), (100, 100, 100), (150, 150, 150), (50, 50, 50),
    (250, 180, 100), (180, 250, 100), (100, 250, 180), (180, 100, 250), (250, 100, 180),
    (100, 180, 250), (180, 180, 250), (250, 250, 180), (200, 200, 150),  (150, 200, 200)   
]
COLOR_NAMES = [
    "light green", "light blue", "blue", "light orange", "olive",
    "brown", "light red", "red", "light purple", "purple",
    "light brown", "cyan", "pink", "beige", "orange",
    "magenta", "sky blue", "dark green", "slate gray", "indigo",
    "tan", "sea green", "salmon", "gray", "periwinkle",
    "gold", "turquoise", "orchid", "coral", "lime",
    "khaki", "royal blue", "yellow", "aqua", "fuchsia",
    "teal", "lavender", "peach", "chartreuse", "rose",
    "steel blue", "silver", "dark gray", "medium gray", "charcoal",
    "marigold", "lime green", "mint", "plum", "hot pink",
    "powder blue", "lilac", "lemon",     "light khaki",    
    "light cyan"
]

def get_color_name(class_id):
    if 0 <= class_id < len(PALETTE):
        return COLOR_NAMES[class_id]
    else:
        return f"Unknown (Class ID: {class_id})"

class PLYDataset(Dataset):
    def __init__(self, file_paths, use_color=False, use_normal=False):
        self.file_paths = file_paths if isinstance(file_paths, list) else [file_paths]
        self.use_color = use_color
        self.use_normal = use_normal

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        pcd = o3d.io.read_point_cloud(str(self.file_paths[idx]))
        coords = np.asarray(pcd.points).astype(np.float32)
        
        feats = []
        if self.use_color:
            colors = np.asarray(pcd.colors).astype(np.float32)
            feats.append(colors)

        if self.use_normal:
            if not pcd.has_normals():
                pcd.estimate_normals()
            normals = np.asarray(pcd.normals).astype(np.float32)
            feats.append(normals)

        feats = np.hstack(feats) if feats else np.zeros((coords.shape[0], 0), dtype=np.float32)
        
        print(f"Coords shape: {coords.shape}, Feats shape: {feats.shape}")
        return torch.from_numpy(coords), torch.from_numpy(feats)

def load_model(model_path, use_color, use_normal, use_msg):
    input_channels = int(args.use_color) * 3 + int(args.use_normal) * 3 #+ int(args.use_multiview) * 128
    try:
        model = Pointnet.get_model(
            num_classes=CONF.NUM_CLASSES,
            is_msg=use_msg,
            input_channels=input_channels,
            use_xyz=True,
            bn=True
        ).cuda()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def cluster_and_count_objects(points, predictions, eps=0.1, min_samples=100):
    object_counts = {}
    object_ids = np.full(len(predictions), -1)
    for class_id in np.unique(predictions):
        class_mask = predictions == class_id
        class_points = points[class_mask]
        if len(class_points) > min_samples:
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(class_points)
            object_counts[class_id] = len(np.unique(clustering.labels_[clustering.labels_ != -1]))
            object_ids[class_mask] = clustering.labels_
    return object_counts, object_ids

def save_colored_pointcloud(points, predictions, output_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.array([PALETTE[p] for p in predictions]) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Colored point cloud saved to: {output_path}")
    return pcd

def add_bounding_boxes(pcd, predictions, object_ids):
    bbox_lines = []
    for class_id in np.unique(predictions):
        class_points = np.asarray(pcd.points)[predictions == class_id]
        for obj_id in np.unique(object_ids[predictions == class_id]):
            if obj_id == -1:  # Skip noise points
                continue
            obj_points = class_points[object_ids[predictions == class_id] == obj_id]
            bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(obj_points))
            bbox.color = np.array(PALETTE[class_id]) / 255.0
            bbox_lines.append(bbox)
    return bbox_lines

def analyze_object_sizes(pcd, predictions, object_ids):
    sizes = {}
    for class_id in np.unique(predictions):
        class_points = np.asarray(pcd.points)[predictions == class_id]
        class_sizes = []
        for obj_id in np.unique(object_ids[predictions == class_id]):
            if obj_id == -1:  # Skip noise points
                continue
            obj_points = class_points[object_ids[predictions == class_id] == obj_id]
            bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(obj_points))
            class_sizes.append(np.prod(bbox.get_extent()))
        sizes[NYUCLASSES[class_id]] = np.mean(class_sizes) if class_sizes else 0
    return sizes

def analyze_spatial_relationships(pcd, predictions, object_ids):
    relationships = []
    centroids = {}
    for class_id in np.unique(predictions):
        class_points = np.asarray(pcd.points)[predictions == class_id]
        for obj_id in np.unique(object_ids[predictions == class_id]):
            if obj_id == -1:  # Skip noise points
                continue
            obj_points = class_points[object_ids[predictions == class_id] == obj_id]
            centroid = np.mean(obj_points, axis=0)
            centroids[(class_id, obj_id)] = centroid
    
    for (class_id1, obj_id1), centroid1 in centroids.items():
        for (class_id2, obj_id2), centroid2 in centroids.items():
            if class_id1 != class_id2 or obj_id1 != obj_id2:
                distance = np.linalg.norm(centroid1 - centroid2)
                if distance < 1.0:  # Threshold for "near"
                    relationships.append(f"{NYUCLASSES[class_id1]} near {NYUCLASSES[class_id2]}")
    
    return relationships

def generate_confidence_heatmap(points, predictions, certainties, output_path):
    plt.figure(figsize=(10, 10))
    plt.scatter(points[:, 0], points[:, 1], c=certainties, cmap='viridis', s=1)
    plt.colorbar(label='Prediction Confidence')
    plt.title('Prediction Confidence Heatmap')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(output_path)
    plt.close()
    print(f"Confidence heatmap saved to: {output_path}")

def visualize_class_distribution(predictions, output_path):
    class_counts = Counter(predictions)
    labels = [NYUCLASSES[class_id] for class_id in class_counts.keys()]
    sizes = list(class_counts.values())
    colors = [np.array(PALETTE[class_id])/255.0 for class_id in class_counts.keys()]

    plt.figure(figsize=(12, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Class Distribution')
    plt.savefig(output_path)
    plt.close()
    print(f"Class distribution chart saved to: {output_path}")

def predict(model, dataloader, device, batch_size):
    predictions = []
    certainties = []
    with torch.no_grad():
        for coords, feats in dataloader:
            coords, feats = coords.to(device), feats.to(device)
            
            pred = []
            cert = []
            for i in range(0, coords.shape[1], batch_size):
                coord_chunk = coords[:, i:i+batch_size, :]
                feat_chunk = feats[:, i:i+batch_size, :]
                input_data = torch.cat([coord_chunk, feat_chunk], dim=2)
                output = model(input_data)
                pred.append(output)
                cert.append(torch.max(torch.softmax(output, dim=2), dim=2)[0])

            pred = torch.cat(pred, dim=1)
            cert = torch.cat(cert, dim=1)
            pred_labels = pred.max(2)[1]
            predictions.extend(pred_labels.cpu().numpy())
            certainties.extend(cert.cpu().numpy())
    
    return np.concatenate(predictions), np.concatenate(certainties)

def visualize_prediction(points, pred, output_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Assign colors based on predicted classes
    colors = np.array([PALETTE[p] for p in pred]) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    # Optionally, set some view parameters
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.8)

    # Render the image
    vis.poll_events()
    vis.update_renderer()
    
    # Capture the image
    image = vis.capture_screen_float_buffer(False)
    
    # Convert to PIL Image and save
    img = Image.fromarray((np.asarray(image) * 255).astype(np.uint8))
    img.save(output_path)

    # Close the window
    vis.destroy_window()

    print(f"Visualization saved to: {output_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.model_path, args.use_color, args.use_normal, args.use_msg).to(device)

    dataset = PLYDataset(args.input, use_color=args.use_color, use_normal=args.use_normal)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    predictions, certainties = predict(model, dataloader, device, args.batch_size)

    points = np.asarray(o3d.io.read_point_cloud(str(args.input)).points)

    print(f"File: {args.input}")
    unique_predictions = np.unique(predictions)
    object_counts, object_ids = cluster_and_count_objects(points, predictions)
    
    # Sort classes by point count
    class_point_counts = {class_id: np.sum(predictions == class_id) for class_id in np.unique(predictions)}
    sorted_classes = sorted(class_point_counts.items(), key=lambda x: x[1], reverse=True)

    print("Class predictions:")
    for class_id, point_count in sorted_classes:
        color_name = get_color_name(class_id)
        object_count = object_counts.get(class_id, 0)
        avg_certainty = np.mean(certainties[predictions == class_id])
        class_name = NYUCLASSES[class_id] if class_id < len(NYUCLASSES) else f"Unknown Class {class_id}"
        print(f"{class_name} ({color_name}): {object_count} (Points: {point_count}, Certainty: {avg_certainty:.2f})")

    print(f"\nTotal unique classes found: {len(object_counts)}")
    print(f"Total points: {len(predictions)}")

    # Save colored point cloud
    output_dir = r"D:\Pointnet2.ScanNet\outputs\Pred"
    os.makedirs(output_dir, exist_ok=True)
    
    input_filename = os.path.basename(args.input)
    output_filename = f"prediction_{os.path.splitext(input_filename)[0]}.ply"
    output_path = os.path.join(output_dir, output_filename)
    
    pcd = save_colored_pointcloud(points, predictions, output_path)

    # Feature 1: Bounding Box Visualization
    bbox_lines = add_bounding_boxes(pcd, predictions, object_ids)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    for bbox in bbox_lines:
        vis.add_geometry(bbox)
    vis.capture_screen_image(os.path.join(output_dir, f"bounding_boxes_{os.path.splitext(input_filename)[0]}.png"))
    vis.destroy_window()
    o3d.visualization.draw_geometries([pcd] + bbox_lines)

    # Feature 2: Object Size Analysis
    object_sizes = analyze_object_sizes(pcd, predictions, object_ids)
    print("\nAverage object sizes (volume in cubic meters):")
    for class_name, size in object_sizes.items():
        print(f"{class_name}: {size:.3f}")
    with open(os.path.join(output_dir, f"object_sizes_{os.path.splitext(input_filename)[0]}.txt"), 'w') as f:
        for class_name, size in object_sizes.items():
            f.write(f"{class_name}: {size:.3f}\n")

    # Feature 3: Spatial Relationship Analysis
    spatial_relationships = analyze_spatial_relationships(pcd, predictions, object_ids)
    print("\nSpatial Relationships:")
    for relationship in set(spatial_relationships):  # Use set to remove duplicates
        print(relationship)
    with open(os.path.join(output_dir, f"spatial_relationships_{os.path.splitext(input_filename)[0]}.txt"), 'w') as f:
        for relationship in set(spatial_relationships):
            f.write(f"{relationship}\n")

    # Feature 4: Confidence Heatmap
    heatmap_path = os.path.join(output_dir, f"confidence_heatmap_{os.path.splitext(input_filename)[0]}.png")
    generate_confidence_heatmap(points, predictions, certainties, heatmap_path)

    # Feature 5: Class Distribution Visualization
    distribution_path = os.path.join(output_dir, f"class_distribution_{os.path.splitext(input_filename)[0]}.png")
    visualize_class_distribution(predictions, distribution_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PointNet++ prediction for ScanNet")
    parser.add_argument("input", type=str, help="Input PLY file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--use_color", action="store_true", help="Use color features")
    parser.add_argument("--use_normal", action="store_true", help="Use normal features")
    parser.add_argument("--use_msg", action="store_true", help="Use multi-scale grouping")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size for processing large point clouds")
    args = parser.parse_args()

    main(args)