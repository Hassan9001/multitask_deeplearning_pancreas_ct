# dataset_conversion.py
# Reformatting structure of raw data (nnUNetV2 compatible)
# NOW with label sanitization: snaps labels to {0,1,2} and saves as uint8

from pathlib import Path
import re
import json
import shutil
import numpy as np
import nibabel as nib
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

IMG_RE = re.compile(r".*_\d{4}\.nii\.gz$", re.IGNORECASE)
SUBTYPE_RE = re.compile(r"subtype(\d+)$", re.IGNORECASE)

VALID = np.array([0, 1, 2], dtype=np.float32)  # allowed class ids
ATOL = 1e-3  # tolerance if there is tiny float noise

def _sanitize_label_file(label_path: Path) -> None:
    """
    Load a label NIfTI, snap values to the nearest of {0,1,2}, save as uint8.
    """
    img = nib.load(str(label_path))
    data = img.get_fdata().astype(np.float32)

    # Map each voxel to closest valid label id
    idx = np.argmin(np.abs(data[..., None] - VALID), axis=-1)
    fixed = VALID[idx].astype(np.uint8)

    hdr = img.header.copy()
    hdr.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(fixed, img.affine, hdr), str(label_path))

def move_split(src_dir: Path, img_dst: Path, lbl_dst: Path | None, split_name: str,
               mapping: dict, validation_cases: list):
    """
    Move files for a split (train/validation/test).
    - Images match *_0000.nii.gz style (i.e., _\\d{4}.nii.gz) -> go to img_dst.
    - Labels are the remaining .nii.gz files -> go to lbl_dst (if provided).
    Also fills mapping and validation_cases.
    """
    if not src_dir.exists():
        return

    for sub in sorted(p for p in src_dir.iterdir() if p.is_dir()):
        # Infer subtype from folder name (e.g., 'subtype0', 'subtype1', 'subtype2')
        m = SUBTYPE_RE.search(sub.name)
        subtype_id = int(m.group(1)) if m else None

        for f in sub.glob("*.nii.gz"):
            if IMG_RE.match(f.name):
                # image
                shutil.move(str(f), img_dst / f.name)
                # record mapping on images (avoid duplicates from labels)
                case_id = f.name.split("_0000.nii.gz")[0]
                mapping[case_id] = {"subtype": subtype_id, "split": split_name}
                if split_name == "validation":
                    validation_cases.append(case_id)
            elif lbl_dst is not None:
                # label -> move then sanitize
                dst = lbl_dst / f.name
                shutil.move(str(f), dst)
                _sanitize_label_file(dst)

def data_converter(base_folder: str, dest_folder: str):
    base = Path(base_folder)
    train = base / "train"
    val   = base / "validation"
    test  = base / "test"

    dest = Path(dest_folder)
    imagesTr = dest / "imagesTr"
    imagesTs = dest / "imagesTs"
    imagesVa = dest / "imagesVa"
    labelsTr = dest / "labelsTr"
    labelsVa = dest / "labelsVa"

    for d in (imagesTr, imagesTs, imagesVa, labelsTr, labelsVa):
        d.mkdir(parents=True, exist_ok=True)

    # For subtype mapping
    mapping = {}            # case_id -> {"subtype": int|None, "split": "train"|"validation"|"test"}
    validation_cases = []   # list of case_ids in validation

    # Train: move images & labels (labels sanitized)
    move_split(train, imagesTr, labelsTr, "train", mapping, validation_cases)

    # Validation: move images & labels (labels sanitized)
    move_split(val, imagesVa, labelsVa, "validation", mapping, validation_cases)

    # Test: only images (no labels)
    if test.exists():
        for f in test.glob("*.nii.gz"):
            if IMG_RE.match(f.name):
                shutil.move(str(f), imagesTs / f.name)
                case_id = f.name.split("_0000.nii.gz")[0]
                if case_id not in mapping:
                    mapping[case_id] = {"subtype": None, "split": "test"}

    # Write subtype_mapping.json
    with open(dest / "subtype_mapping.json", "w") as fh:
        json.dump(
            {"mapping": mapping, "validation_cases": validation_cases},
            fh,
            indent=2
        )

    return dest

if __name__ == "__main__":
    base_folder = "ML-Quiz-3DMedImg"
    dest_folder = "nnUNet_raw/Dataset777_3DMedImg"

    dest = data_converter(base_folder, dest_folder)

    # Auto-counts
    num_training = len(list((Path(dest) / "labelsTr").glob("*.nii.gz")))
    print(f"Number of training cases: {num_training}")

    num_test = len(list((Path(dest) / "imagesTs").glob("*.nii.gz")))
    print(f"Number of test cases: {num_test}")

    num_validation = len(list((Path(dest) / "imagesVa").glob("*.nii.gz")))
    print(f"Number of validation cases: {num_validation}")

    # NOTE: Using regions for 'pancreas' = union(1,2) is fine. Keep as you had it.
    generate_dataset_json(
        str(dest),
        channel_names={0: "CT"},
        labels={
            "background": 0,
            "pancreas": (1, 2),
            "lesion": 2,
        },
        regions_class_order=(1, 2),
        num_training_cases=num_training,
        file_ending=".nii.gz",
        overwrite_image_reader_writer="SimpleITKIO",
        converted_by="Hassan Al-Hayawi",
    )
