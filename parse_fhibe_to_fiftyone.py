#!/usr/bin/env python3
"""Fast parallel FHIBE dataset parser for FiftyOne"""

import fiftyone as fo
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp


# Canonical keypoint schema - all possible keypoints in order
CANONICAL_KEYPOINTS = [
    "Nose",  # 0
    "Right eye inner",  # 1
    "Right eye",  # 2
    "Right eye outer",  # 3
    "Left eye inner",  # 4
    "Left eye",  # 5
    "Left eye outer",  # 6
    "Right ear",  # 7
    "Left ear",  # 8
    "Mouth right",  # 9
    "Mouth left",  # 10
    "Right shoulder",  # 11
    "Left shoulder",  # 12
    "Right elbow",  # 13
    "Left elbow",  # 14
    "Right wrist",  # 15
    "Left wrist",  # 16
    "Right pinky knuckle",  # 17
    "Left pinky knuckle",  # 18
    "Right index knuckle",  # 19
    "Left index knuckle",  # 20
    "Right thumb knuckle",  # 21
    "Left thumb knuckle",  # 22
    "Right hip",  # 23
    "Left hip",  # 24
    "Right knee",  # 25
    "Left knee",  # 26
    "Right ankle",  # 27
    "Left ankle",  # 28
    "Right heel",  # 29
    "Left heel",  # 30
    "Right foot index",  # 31
    "Left foot index",  # 32
]


def strip_prefix(value):
    """Strip leading number like '5. Brown' -> 'Brown'"""
    if isinstance(value, str):
        return re.sub(r'^\d+\.\s*', '', value)
    elif isinstance(value, list):
        return [strip_prefix(v) for v in value]
    return value


def to_classification(value, multi=False):
    """Convert to Classification or Classifications"""
    if value is None:
        return None
    value = strip_prefix(value)
    if multi:
        if isinstance(value, list):
            return fo.Classifications(classifications=[fo.Classification(label=v) for v in value if v])
        return fo.Classifications(classifications=[fo.Classification(label=value)])
    else:
        if isinstance(value, list):
            value = value[0] if value else None
        return fo.Classification(label=value) if value else None


def load_json(path):
    """Load JSON file"""
    with open(path, 'r') as f:
        return json.load(f)


def get_image_size(img_path):
    """Get image dimensions efficiently"""
    img = Image.open(img_path)
    return img.size


def keypoints_to_fiftyone(keypoints_dict, img_w, img_h):
    """Convert keypoints dict to FiftyOne Keypoints using canonical schema"""
    # Create mapping from numeric ID to coordinates
    keypoint_map = {}
    for key, coords in keypoints_dict.items():
        idx = int(key.split('.')[0])
        keypoint_map[idx] = coords
    
    # Build points array according to canonical schema
    points = []
    for idx in range(len(CANONICAL_KEYPOINTS)):
        if idx in keypoint_map:
            coords = keypoint_map[idx]
            # Check visibility flag (coords[2])
            if len(coords) >= 3 and coords[2] > 0:
                # Visible keypoint - normalize coordinates
                points.append([coords[0] / img_w, coords[1] / img_h])
            else:
                # Invisible or occluded - use NaN
                points.append([float("nan"), float("nan")])
        else:
            # Not annotated at all - use NaN
            points.append([float("nan"), float("nan")])
    
    return fo.Keypoints(keypoints=[fo.Keypoint(points=points)]) if points else None


def segments_to_polylines(segments, img_w, img_h):
    """Convert segmentation to polylines"""
    polylines = []
    for seg in segments:
        if "polygon" in seg and len(seg["polygon"]) >= 3:
            points = [[p["x"] / img_w, p["y"] / img_h] for p in seg["polygon"]]
            label = strip_prefix(seg.get("class_name", "unknown"))
            polylines.append(fo.Polyline(label=label, points=[points], closed=True, filled=True))
    return fo.Polylines(polylines=polylines) if polylines else None


def extract_demographics(annos):
    """Extract subject demographics"""
    if not annos or "subject_annotation" not in annos:
        return {}
    
    subj = annos["subject_annotation"][0]
    return {
        "age": subj.get("age"),
        "pronouns": strip_prefix(subj.get("pronoun", [])),
        "ancestry": strip_prefix(subj.get("ancestry", [])),
        "nationality": strip_prefix(subj.get("nationality", [])),
        "skin_color": subj.get("natural_skin_color"),
        "hair_type": to_classification(subj.get("natural_hair_type")),
        "hair_color": to_classification(subj.get("natural_hair_color"), multi=True),
        "eye_color_left": to_classification(subj.get("natural_left_eye_color"), multi=True),
        "eye_color_right": to_classification(subj.get("natural_right_eye_color"), multi=True),
        "facial_hairstyle": to_classification(subj.get("facial_hairstyle"), multi=True),
        "facial_hair_color": to_classification(subj.get("natural_facial_haircolor"), multi=True),
        "facial_marks": to_classification(subj.get("facial_marks"), multi=True),
    }

def create_sample_dict(img_path, annos, subject_id, image_id, demographics, slice_name, img_w, img_h):
    """Create sample as dictionary (for serialization across processes)"""
    is_main = slice_name.startswith("main_")
    
    sample_data = {
        "filepath": str(img_path),
        "slice_name": slice_name,
        "subject_id": subject_id,
        "image_id": image_id,
        **demographics
    }
    
    # Image metadata
    if "image_annotation" in annos:
        img_ann = annos["image_annotation"]
        sample_data.update({
            "capture_date": img_ann.get("user_date_captured"),
            "capture_time": img_ann.get("user_hour_captured"),
            "location_country": img_ann.get("location_country"),
            "location_region": img_ann.get("location_region"),
            "scene": to_classification(img_ann.get("scene")),
            "lighting": to_classification(img_ann.get("lighting"), multi=True),
            "weather": to_classification(img_ann.get("weather"), multi=True),
            "camera_position": to_classification(img_ann.get("camera_position")),
            "camera_distance": to_classification(img_ann.get("camera_distance")),
        })
        
        manufacturer = img_ann.get("manufacturer")
        sample_data["manufacturer"] = manufacturer.strip('\x00') if manufacturer else None
        model = img_ann.get("model")
        sample_data["camera_model"] = model.strip('\x00') if model else None
    
    # Subject appearance & actions
    if "subject_annotation" in annos and annos["subject_annotation"]:
        subj = annos["subject_annotation"][0]
        sample_data.update({
            "hairstyle": to_classification(subj.get("hairstyle")),
            "apparent_hair_type": to_classification(subj.get("apparent_hair_type")),
            "apparent_hair_color": to_classification(subj.get("apparent_hair_color"), multi=True),
            "body_pose": to_classification(subj.get("action_body_pose")),
            "head_pose": to_classification(subj.get("head_pose")),
            "interaction_object": to_classification(subj.get("action_subject_object_interaction"), multi=True),
            "interaction_subject": to_classification(subj.get("action_subject_subject_interaction"), multi=True),
        })
        
        # Face bbox - ONLY for main images
        if is_main and "face_bbox" in subj:
            bbox = subj["face_bbox"]
            x, y, w, h = bbox
            sample_data["face_bbox"] = fo.Detection(
                label="face",
                bounding_box=[x/img_w, y/img_h, w/img_w, h/img_h]
            )
        
        # Keypoints - ONLY for main images
        if is_main and "keypoints" in subj:
            kps = keypoints_to_fiftyone(subj["keypoints"], img_w, img_h)
            if kps:
                sample_data["keypoints"] = kps
        
        # Segmentation - for ALL image types
        if "segments" in subj:
            polys = segments_to_polylines(subj["segments"], img_w, img_h)
            if polys:
                sample_data["segmentations"] = polys
    
    return sample_data


def process_subject(subject_dir, base_path):
    """Process a single subject (runs in parallel)"""
    subject_id = subject_dir.name
    session_dirs = sorted([d for d in subject_dir.iterdir() if d.is_dir()])
    
    if not session_dirs:
        return []
    
    # Get demographics from first session
    first_session = session_dirs[0]
    image_id = first_session.name
    main_anno_path = first_session / f"main_annos_{image_id}.json"
    demographics = extract_demographics(load_json(main_anno_path))
    
    # Generate unique group ID for this subject
    group_id = subject_id
    samples = []
    
    for session_idx, session_dir in enumerate(session_dirs):
        image_id = session_dir.name
        
        # Main image
        main_img = session_dir / f"main_{image_id}.png"
        main_anno = session_dir / f"main_annos_{image_id}.json"
        if main_img.exists() and main_anno.exists():
            img_w, img_h = get_image_size(main_img)
            sample_data = create_sample_dict(
                main_img, load_json(main_anno), subject_id, image_id,
                demographics, f"main_{session_idx + 1}", img_w, img_h
            )
            sample_data["group_id"] = group_id
            samples.append(sample_data)
        
        # Face crop only
        face_crop_img = session_dir / f"faces_crop_only_{image_id}_{subject_id}.png"
        face_crop_anno = session_dir / f"faces_crop_only_annos_{image_id}_{subject_id}.json"
        if face_crop_img.exists() and face_crop_anno.exists():
            img_w, img_h = get_image_size(face_crop_img)
            sample_data = create_sample_dict(
                face_crop_img, load_json(face_crop_anno), subject_id, image_id,
                demographics, f"face_crop_{session_idx + 1}", img_w, img_h
            )
            sample_data["group_id"] = group_id
            samples.append(sample_data)
        
        # Face aligned
        face_align_img = session_dir / f"faces_crop_and_align_{image_id}_{subject_id}.png"
        face_align_anno = session_dir / f"faces_crop_and_align_annos_{image_id}_{subject_id}.json"
        if face_align_img.exists() and face_align_anno.exists():
            img_w, img_h = get_image_size(face_align_img)
            sample_data = create_sample_dict(
                face_align_img, load_json(face_align_anno), subject_id, image_id,
                demographics, f"face_aligned_{session_idx + 1}", img_w, img_h
            )
            sample_data["group_id"] = group_id
            samples.append(sample_data)
    
    return samples


def setup_keypoint_skeleton(dataset):
    """Setup keypoint skeleton from canonical schema"""
    
    # Define skeleton connections using fixed indices
    edges = [
        # Face connections
        [1, 2], [2, 3],      # Right eye
        [4, 5], [5, 6],      # Left eye
        [9, 10],             # Mouth
        
        # Upper body
        [11, 12],            # Shoulders connected
        
        # Right arm
        [11, 13], [13, 15],  # Shoulder to elbow to wrist
        [15, 17],            # Wrist to pinky
        [15, 19],            # Wrist to index
        [15, 21],            # Wrist to thumb
        
        # Left arm  
        [12, 14], [14, 16],
        [16, 18],
        [16, 20],
        [16, 22],
        
        # Torso
        [11, 23],            # Right shoulder to hip
        [12, 24],            # Left shoulder to hip
        [23, 24],            # Hips connected
        
        # Right leg
        [23, 25], [25, 27],  # Hip to knee to ankle
        [27, 29],            # Ankle to heel
        [27, 31],            # Ankle to foot index
        
        # Left leg
        [24, 26], [26, 28],
        [28, 30],
        [28, 32],
    ]
    
    dataset.default_skeleton = fo.KeypointSkeleton(
        labels=CANONICAL_KEYPOINTS,
        edges=edges
    )
    dataset.save()


def parse_fhibe(base_path, dataset_name="fhibe", max_subjects=None, num_workers=None):
    """Parse FHIBE dataset into FiftyOne with parallelization"""
    base_path = Path(base_path)
    
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    print(f"Using {num_workers} workers")
    
    dataset = fo.Dataset(dataset_name, persistent=True, overwrite=True)
    dataset.add_group_field("subject_group", default="main_1")
    
    subject_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and not d.name.startswith('.')])
    if max_subjects:
        subject_dirs = subject_dirs[:max_subjects]
    
    print(f"Processing {len(subject_dirs)} subjects...")
    
    # Process subjects in parallel
    all_sample_dicts = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        process_func = partial(process_subject, base_path=base_path)
        futures = {executor.submit(process_func, subj_dir): subj_dir for subj_dir in subject_dirs}
        
        with tqdm(total=len(subject_dirs), desc="Loading subjects") as pbar:
            for future in as_completed(futures):
                samples = future.result()
                all_sample_dicts.extend(samples)
                pbar.update(1)
    
    print(f"Creating {len(all_sample_dicts)} FiftyOne samples...")
    
    # Convert sample dicts to FiftyOne samples with groups
    groups = {}
    fo_samples = []
    
    for sample_data in tqdm(all_sample_dicts, desc="Converting to FiftyOne"):
        group_id = sample_data.pop("group_id")
        slice_name = sample_data.pop("slice_name")
        
        if group_id not in groups:
            groups[group_id] = fo.Group()
        
        sample = fo.Sample(filepath=sample_data.pop("filepath"))
        sample["subject_group"] = groups[group_id].element(slice_name)
        
        for key, value in sample_data.items():
            sample[key] = value
        
        fo_samples.append(sample)
    
    dataset.add_samples(fo_samples)
    setup_keypoint_skeleton(dataset)
    
    return dataset


if __name__ == "__main__":
    dataset = parse_fhibe(
        "fhibe.20250716.u.gT5_rFTA_downsampled_public/data/raw/fhibe_downsampled",
        dataset_name="fhibe",
        max_subjects=None,  # None = all subjects
        num_workers=None  # None = auto-detect CPUs
    )
    print(f"\nDataset created: {dataset.name}")
    print(f"Total samples: {len(dataset)}")
    print(f"Total subjects: {len(dataset.distinct('subject_id'))}")