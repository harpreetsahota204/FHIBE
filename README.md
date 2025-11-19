# FHIBE Dataset - FiftyOne Format
![image](fhibe_sample.gif)


## Dataset Details

### Dataset Description

The FHIBE (Fairness in Human Identity Bias Evaluation) dataset is a consent-driven benchmark designed specifically for evaluating fairness in computer vision AI systems. Unlike traditional datasets that prioritize scale and performance metrics, FHIBE places human dignity, consent, and data rights at its core.

The dataset contains high-quality images of consenting participants with comprehensive annotations including facial keypoints, segmentations, demographic information, and contextual metadata. Each participant provided informed consent and retained ownership of their data, with the ability to withdraw at any time.

- **Curated by:** Sony AI
- **Language(s):** Multilingual (annotations in English)
- **License:** Requires registration and agreement to terms of use
- **Paper:** [Introducing FHIBE: A Consent-Driven Benchmark for AI Fairness Evaluation](https://ai.sony/blog/Introducing-FHIBE-A-Consent-Driven-Benchmark-for-AI-Fairness-Evaluation/)

### Dataset Sources

- **Registration & Download:** https://fairnessbenchmark.ai.sony/register
- **Blog Posts:**
  - [What If Fairness Started at the Dataset Level?](https://ai.sony/blog/What-If-Fairness-Started-at-the-Dataset-Level/)
  - [The FHIBE Team: Data Dignity and the People Who Made It Possible](https://ai.sony/blog/The-FHIBE-Team-Data-Dignity-and-the-People-Who-Made-It-Possible/)
  - [Introducing FHIBE: A Consent-Driven Benchmark for AI Fairness Evaluation](https://ai.sony/blog/Introducing-FHIBE-A-Consent-Driven-Benchmark-for-AI-Fairness-Evaluation/)

# Parsing and Using the FHIBE Dataset

## Downloading the Dataset

1. **Register and download** the FHIBE dataset from https://fairnessbenchmark.ai.sony/register
2. Extract the downloaded archive to your local machine
3. Note the path to the extracted `fhibe_downsampled` directory

## Parsing to FiftyOne

This repository includes a parser script that converts the raw FHIBE dataset into a FiftyOne dataset with proper structure, annotations, and metadata.

### Setup

```bash
# Clone this repository
git clone https://github.com/harpreetsahota204/FHIBE
cd FHIBE

# Install dependencies
pip install fiftyone pillow tqdm
```

### Running the Parser

```bash
python parse_fhibe_to_fiftyone.py
```

By default, the script expects the dataset at:
```
fhibe.20250716.u.gT5_rFTA_downsampled_public/data/raw/fhibe_downsampled
```

You can modify the path in the script's `__main__` section if your dataset is located elsewhere:

```python
if __name__ == "__main__":
    dataset = parse_fhibe(
        "/path/to/your/fhibe_downsampled",  # Change this path
        dataset_name="fhibe",
        max_subjects=None,  # Set to a number to parse only N subjects
        num_workers=None    # Set to control parallelization (default: auto)
    )
```

The parser will:
- Process all subjects in parallel (using all available CPU cores by default)
- Create grouped samples linking main images with face crops for each subject
- Parse all 33 keypoints with proper skeleton structure
- Extract demographics, context metadata, and segmentations
- Display progress bars showing parsing status

Parsing the full dataset typically takes 5-15 minutes depending on your system.

## Working with the FiftyOne Dataset

![image](fhibe_dashboard.gif)

### Launching the FiftyOne App

Once parsed, you can explore the dataset interactively:

```python
import fiftyone as fo

# Load the dataset
dataset = fo.load_dataset("fhibe")

# Launch the FiftyOne App
session = fo.launch_app(dataset)
```

### Basic Dataset Operations

```python
import fiftyone as fo
from fiftyone import ViewField as F

# Load dataset
dataset = fo.load_dataset("fhibe")

# View dataset summary
print(dataset)

# Count total samples
print(f"Total samples: {len(dataset)}")

# Count unique subjects
print(f"Total subjects: {len(dataset.distinct('subject_id'))}")

# View field schema
print(dataset.get_field_schema())

# Access a sample
sample = dataset.first()
print(f"Subject ID: {sample.subject_id}")
print(f"Age: {sample.age}")
print(f"Location: {sample.location_country}")
```

### Filtering and Querying

```python
from fiftyone import ViewField as F

# Filter by demographics
young_adults = dataset.match(F("age") < 30)

# Filter by location
kenya_samples = dataset.match(F("location_country") == "Kenya")

# Filter main images only (with keypoints)
main_images = dataset.match(F("slice_name").starts_with("main_"))

# Filter by specific group slice
face_crops = dataset.select_group_slices("face_crop_1")

# Filter multiple slices
main_and_crops = dataset.select_group_slices(["main_1", "face_crop_1"])

# Complex queries using logical operators
outdoor_standing = dataset.match(
    (F("scene.label").contains("Outdoor")) & (F("body_pose.label") == "Standing")
)

# Filter by age range
adults_25_to_40 = dataset.match((F("age") >= 25) & (F("age") <= 40))

# Filter samples with visible keypoints (non-NaN)
samples_with_nose = dataset.match(F("keypoints.keypoints.points")[0][0] != None)
```

### Working with Groups

The dataset uses FiftyOne's grouped samples to link related images:

```python
# Get the default group slice (main_1)
default_view = dataset.select_group_slices()

# View specific slices
main_images = dataset.select_group_slices("main_1")
face_crops = dataset.select_group_slices("face_crop_1") 
aligned_faces = dataset.select_group_slices("face_aligned_1")

# View multiple slices simultaneously
main_and_crop = dataset.select_group_slices(["main_1", "face_crop_1"])

# Iterate through groups
for sample in dataset.select_group_slices("main_1"):
    # Get the group for this sample
    group = sample.subject_group
    print(f"Subject: {sample.subject_id}, Group ID: {group.id}")
    
    # To access other slices for this subject, query by group field
    group_samples = dataset.match(F("subject_group.id") == group.id)
    for s in group_samples:
        print(f"  - {s.slice_name}: {s.filepath}")
```

### Sorting and Limiting

```python
# Sort by age
oldest_first = dataset.sort_by("age", reverse=True)

# Limit number of samples
small_view = dataset.limit(100)

# Skip and limit (pagination)
page_2 = dataset.skip(100).limit(100)

# Random sampling
random_samples = dataset.take(50)

# Sort by multiple fields
sorted_view = dataset.sort_by([("location_country", False), ("age", True)])
```

### Aggregations and Statistics

```python
from fiftyone import ViewField as F

# Count values
pose_counts = dataset.count_values("body_pose.label")
print("Body pose distribution:", pose_counts)

country_counts = dataset.count_values("location_country")
print("Samples by country:", country_counts)

# Get distinct values
all_countries = dataset.distinct("location_country")
all_ancestries = dataset.distinct("ancestry")

# Compute bounds (min/max)
age_bounds = dataset.bounds("age")
print(f"Age range: {age_bounds[0]} to {age_bounds[1]}")

# Aggregate statistics
stats = dataset.aggregate([
    fo.Count(),
    fo.Mean("age"),
    fo.Std("age")
])
print(f"Total samples: {stats[0]}")
print(f"Mean age: {stats[1]:.1f}")
print(f"Age std dev: {stats[2]:.1f}")
```

### Visualizing Keypoints and Segmentations

```python
# View main samples with keypoints
main_samples = dataset.select_group_slices("main_1")
session = fo.launch_app(main_samples)

# The skeleton will automatically be displayed with proper connections
# Use the FiftyOne App sidebar to toggle keypoint and segmentation visibility

# Filter to only samples with keypoints
samples_with_keypoints = main_samples.exists("keypoints")
session = fo.launch_app(samples_with_keypoints)
```

### Fairness Analysis Example

```python
from fiftyone import ViewField as F

# Analyze distribution across demographic groups
print("\n=== Dataset Demographics ===")

# Ancestry distribution
ancestry_counts = dataset.count_values("ancestry")
for ancestry, count in sorted(ancestry_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{ancestry}: {count} samples")

# Age distribution by ranges
age_ranges = {
    "18-25": dataset.match((F("age") >= 18) & (F("age") <= 25)),
    "26-35": dataset.match((F("age") >= 26) & (F("age") <= 35)),
    "36-50": dataset.match((F("age") >= 36) & (F("age") <= 50)),
    "51+": dataset.match(F("age") >= 51)
}

print("\n=== Age Distribution ===")
for range_name, view in age_ranges.items():
    print(f"{range_name}: {len(view)} samples")

# Body pose distribution across age groups
print("\n=== Body Pose by Age Group ===")
for range_name, view in age_ranges.items():
    poses = view.count_values("body_pose.label")
    print(f"\n{range_name}:")
    for pose, count in sorted(poses.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {pose}: {count}")

# Analyze keypoint visibility across demographics
print("\n=== Samples with Full Body Keypoints ===")
main_images = dataset.select_group_slices("main_1")

for country in dataset.distinct("location_country")[:5]:  # Top 5 countries
    country_view = main_images.match(F("location_country") == country)
    with_keypoints = country_view.exists("keypoints")
    print(f"{country}: {len(with_keypoints)}/{len(country_view)} have keypoints")
```

### Creating Custom Views

```python
from fiftyone import ViewField as F

# Create a view for fairness evaluation
eval_view = (
    dataset
    .select_group_slices("main_1")  # Only main images
    .exists("keypoints")             # Must have keypoints
    .match(F("age") >= 18)          # Adults only
    .match(F("head_pose.label") == "Typical")  # Typical head pose
)

print(f"Evaluation set: {len(eval_view)} samples")

# Create views for different evaluation scenarios
outdoor_view = eval_view.match(F("scene.label").contains("Outdoor"))
indoor_view = eval_view.match(F("scene.label").contains("Indoor"))

print(f"Outdoor samples: {len(outdoor_view)}")
print(f"Indoor samples: {len(indoor_view)}")
```

### Selecting Fields

```python
# Select only specific fields for export or analysis
simplified_view = dataset.select_fields([
    "subject_id",
    "age", 
    "ancestry",
    "location_country",
    "keypoints"
])

# Exclude certain fields
minimal_view = dataset.exclude_fields([
    "manufacturer",
    "camera_model",
    "capture_time"
])
```

### Exporting Data

```python
# Export filtered samples
view = dataset.match(F("location_country") == "Kenya")
view.export(
    export_dir="/path/to/export",
    dataset_type=fo.types.ImageDirectory,
)

# Export annotations to JSON
view.export(
    export_dir="/path/to/json",
    dataset_type=fo.types.FiftyOneDataset,
)

# Export specific label fields
main_with_keypoints = dataset.select_group_slices("main_1").exists("keypoints")
main_with_keypoints.export(
    export_dir="/path/to/coco",
    dataset_type=fo.types.COCODetectionDataset,
    label_field="keypoints",
)
```

### Saving Views

```python
# Save a view for later use
eval_view = (
    dataset
    .select_group_slices("main_1")
    .exists("keypoints")
    .match(F("age") >= 18)
)

dataset.save_view("eval_set", eval_view)

# Load a saved view
eval_view = dataset.load_saved_view("eval_set")

# List all saved views
print(dataset.list_saved_views())

# Delete a saved view
dataset.delete_saved_view("eval_set")
```

### Additional Resources

- **FiftyOne Documentation:** https://docs.voxel51.com
- **FiftyOne Cheat Sheets:** https://docs.voxel51.com/cheat_sheets/index.html
  - Filtering: https://docs.voxel51.com/cheat_sheets/filtering_cheat_sheet.html
  - Views: https://docs.voxel51.com/cheat_sheets/views_cheat_sheet.html
- **Groups Guide:** https://docs.voxel51.com/user_guide/groups.html
- **FiftyOne Tutorials:** https://docs.voxel51.com/tutorials/index.html

For questions or issues with the parser, please open an issue in this repository.



## Uses

### Direct Use

FHIBE is specifically designed for:
- **Fairness evaluation** of computer vision models across diverse human populations
- **Bias detection** in facial recognition, keypoint detection, and segmentation systems
- **Benchmarking** model performance across different demographic groups
- **Research** into ethical AI development and consent-based data collection methodologies

The dataset enables researchers to measure and mitigate biases in AI systems while respecting human dignity and data rights.

### Out-of-Scope Use

- **Surveillance or tracking:** The dataset must not be used for any surveillance, tracking, or identification systems
- **Commercial facial recognition:** Not intended for deployment in commercial facial recognition products
- **Unauthorized purposes:** Any use beyond fairness evaluation and bias research requires explicit permission
- **Re-identification:** Attempts to identify or contact participants are strictly prohibited

## Dataset Structure

### FiftyOne Dataset Organization

The dataset is organized using FiftyOne's grouped samples structure, where each subject has multiple image types grouped together:

#### Sample Groups

Each subject has three types of images per capture session:
- **`main_*`**: Full-body or upper-body images with complete annotations
- **`face_crop_*`**: Cropped face regions
- **`face_aligned_*`**: Aligned and cropped face images

All images from the same subject/session are linked via the `subject_group` field, enabling easy navigation between different views of the same person.

#### Fields

Each sample contains the following fields:

**Identity & Organization:**
- `subject_id`: Unique identifier for each participant
- `image_id`: Unique identifier for each capture session
- `slice_name`: Type of image (e.g., "main_1", "face_crop_1")
- `subject_group`: FiftyOne group linking related images

**Demographics (Natural/Permanent Attributes):**
- `age`: Participant's age
- `pronouns`: Self-identified pronouns
- `ancestry`: Geographic ancestry
- `nationality`: Nationality
- `skin_color`: Natural skin color (RGB values)
- `hair_type`: Natural hair type classification
- `hair_color`: Natural hair color (Classifications)
- `eye_color_left`: Left eye color (Classifications)
- `eye_color_right`: Right eye color (Classifications)
- `facial_hairstyle`: Facial hair style (Classifications)
- `facial_hair_color`: Facial hair color (Classifications)
- `facial_marks`: Facial marks or features (Classifications)

**Appearance (Variable Attributes):**
- `hairstyle`: Current hairstyle
- `apparent_hair_type`: Apparent hair type in image
- `apparent_hair_color`: Apparent hair color in image

**Pose & Actions:**
- `body_pose`: Body position (Classification)
- `head_pose`: Head orientation (Classification)
- `interaction_object`: Object interactions (Classifications)
- `interaction_subject`: Subject-to-subject interactions (Classifications)

**Capture Context:**
- `capture_date`: Date of capture (month-year)
- `capture_time`: Time of day
- `location_country`: Country of capture
- `location_region`: Region/state of capture
- `scene`: Scene type (Classification)
- `lighting`: Lighting conditions (Classifications)
- `weather`: Weather conditions (Classifications)
- `camera_position`: Camera position relative to subject
- `camera_distance`: Distance category
- `manufacturer`: Camera manufacturer
- `camera_model`: Camera model

**Annotations (Main Images Only):**
- `face_bbox`: Face bounding box (Detection)
- `keypoints`: 33-point body keypoints (Keypoints) with skeleton
- `segmentations`: Face and body part segmentations (Polylines)

#### Keypoint Schema

The dataset includes 33 keypoints following this canonical order:

```python
0.  Nose
1.  Right eye inner
2.  Right eye
3.  Right eye outer
4.  Left eye inner
5.  Left eye
6.  Left eye outer
7.  Right ear
8.  Left ear
9.  Mouth right
10. Mouth left
11. Right shoulder
12. Left shoulder
13. Right elbow
14. Left elbow
15. Right wrist
16. Left wrist
17. Right pinky knuckle
18. Left pinky knuckle
19. Right index knuckle
20. Left index knuckle
21. Right thumb knuckle
22. Left thumb knuckle
23. Right hip
24. Left hip
25. Right knee
26. Left knee
27. Right ankle
28. Left ankle
29. Right heel
30. Left heel
31. Right foot index
32. Left foot index
```

Invisible or occluded keypoints are marked with `NaN` coordinates. The dataset includes a skeleton connecting these keypoints for pose visualization.

## Dataset Creation

### Curation Rationale

FHIBE was created to address fundamental issues in AI fairness evaluation:

1. **Consent-first approach:** Traditional computer vision datasets often lack proper consent, treating people as data points rather than rights-holders
2. **Fairness evaluation focus:** Most datasets prioritize performance metrics over fairness, leading to systems that work well on average but fail for underrepresented groups
3. **Data dignity:** Recognition that people should retain ownership and control over their personal data
4. **Comprehensive representation:** Intentional effort to include diverse populations often underrepresented in AI datasets

### Source Data

#### Data Collection and Processing

The FHIBE dataset was collected through a rigorous consent-driven process:

- **Informed consent:** All participants provided explicit, informed consent with full understanding of how their data would be used
- **Data ownership:** Participants retain ownership of their data with the right to withdraw at any time
- **Professional collection:** Images were captured by professional photographers in diverse locations
- **Quality control:** Multiple rounds of annotation quality assurance with consistency checks
- **Privacy protection:** All data collection and handling followed strict privacy protocols

#### Who are the source data producers?

The source data consists of images of real people who voluntarily participated in the dataset creation. Sony AI worked with diverse communities globally to ensure broad representation across:
- Geographic regions
- Age groups
- Gender identities
- Ethnic backgrounds
- Physical characteristics

All participants were compensated fairly for their time and contribution.

### Annotations

#### Annotation Process

Annotations include:
- **Keypoints:** 33-point body pose annotations with visibility flags
- **Segmentations:** Face and body part segmentation polygons
- **Bounding boxes:** Face detection boxes
- **Demographics:** Self-reported demographic information
- **Context:** Image capture metadata (location, lighting, weather, camera settings)

Annotations were performed by trained annotators with quality assurance checks and inter-annotator agreement validation.

#### Who are the annotators?

Professional data annotators trained specifically for this dataset, working under strict quality control guidelines with oversight from Sony AI researchers.

#### Personal and Sensitive Information

The dataset contains images of real people along with self-reported demographic information including:
- Age
- Ancestry
- Nationality
- Physical characteristics (skin color, hair type, eye color)

**Privacy measures:**
- All participants provided informed consent specifically for fairness evaluation research
- Participants retain the right to withdraw their data at any time
- No personally identifiable information (names, addresses, contact details) is included
- Images are intended solely for fairness evaluation, not identification

## Bias, Risks, and Limitations

### Limitations

- **Scope:** While diverse, no dataset can represent all human populations comprehensively
- **Context:** Images captured in controlled conditions may not reflect all real-world scenarios
- **Annotation subjectivity:** Some annotations (e.g., apparent characteristics) involve subjective judgments
- **Temporal:** Dataset represents a snapshot in time; fashion, hairstyles, and cultural expressions evolve
- **Consent-scale tradeoff:** Prioritizing consent and data dignity may limit dataset size compared to web-scraped alternatives

### Recommendations

**Users should:**
- **Respect consent boundaries:** Use the dataset only for fairness evaluation and bias research
- **Report limitations:** Acknowledge dataset limitations in any published research
- **Avoid deployment:** Do not use for deployed systems without additional considerations and permissions
- **Consider intersectionality:** Examine fairness across intersections of demographic attributes, not just individual categories
- **Engage with ethics:** Consider the broader ethical implications of fairness evaluation work
- **Honor data dignity:** Treat the data with the same respect you would want for your own images

**Users must not:**
- Use for surveillance or tracking applications
- Attempt to identify or contact participants
- Share or redistribute the dataset without authorization
- Use for purposes beyond fairness evaluation without explicit permission

## Citation

```bibtex
[Citation information to be provided by Sony AI]
```

## More Information

For more information about the dataset, including access requests and usage guidelines:

- **Registration:** https://fairnessbenchmark.ai.sony/register
- **Sony AI Website:** https://ai.sony
- **Contact:** [Contact information to be provided]

## Dataset Card Authors

This FiftyOne dataset card and parsing code created by the community. Original dataset curated by Sony AI.

## Acknowledgments

The FHIBE dataset represents the collaborative effort of researchers, photographers, annotators, and most importantly, the participants who generously contributed their data for fairness research. Special recognition to the Sony AI team for pioneering consent-driven dataset creation in computer vision.