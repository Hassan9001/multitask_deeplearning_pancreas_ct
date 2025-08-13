### Reformating structure of raw data (so that it is nnUNetV2 compatible) ###
#############################################################################

import shutil
import nibabel as nib
import numpy as np
import json
from pathlib import Path
import argparse


def organize_data_for_nnunet(merge_validation_with_training=True):
    base_path = Path('./')
    dataset_name = 'Dataset777_3DMedImg'
    nnunet_raw_data_path = base_path / 'nnUNet_raw' / dataset_name

    # Create necessary directories
    if merge_validation_with_training:
        folders = ['imagesTr', 'labelsTr', 'imagesTs']
    else:
        folders = ['imagesTr', 'labelsTr', 'imagesTs', 'imagesVal', 'labelsVal']
    for folder in folders:
        (nnunet_raw_data_path / folder).mkdir(parents=True, exist_ok=True)

    # Create dataset.json
    dataset_json = {
        "name": "Pancreas",
        "description": "Pancreas Segmentation Dataset",
        "tensorImageSize": "3D",
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "pancreas": 1, "lesion": 2},
        "class_types": {"subtype0": 0, "subtype1": 1, "subtype2": 2},
        "numTraining": 0,
        "numValidation": 0,
        "numTest": 0,
        "file_ending": ".nii.gz",
        "training": [],
        "validation": [],
        "test": []
    }

    subtype_mapping = {}

    # Process training data
    for subtype in ['subtype0', 'subtype1', 'subtype2']:
        train_path = base_path / 'ML-Quiz-3DMedImg' / 'train' / subtype
        if train_path.exists():
            for file in sorted(train_path.glob('*_0000.nii.gz')):
                case_id = file.stem.split('_0000')[0]
                mask_file = train_path / f"{case_id}.nii.gz"

                if not mask_file.exists():
                    print(f"Warning: Mask file not found for training case '{case_id}'. Skipping...")
                    continue

                try:
                    shutil.copy2(file, nnunet_raw_data_path / 'imagesTr' / file.name)

                    mask = nib.load(mask_file)
                    mask_data = np.round(mask.get_fdata()).astype(np.uint8)
                    mask_data = np.clip(mask_data, 0, 2)
                    new_mask = nib.Nifti1Image(mask_data, mask.affine)
                    nib.save(new_mask, nnunet_raw_data_path / 'labelsTr' / f"{case_id}.nii.gz")

                    dataset_json["training"].append({
                        "image": f"./imagesTr/{file.name}",
                        "label": f"./labelsTr/{case_id}.nii.gz"
                    })

                    subtype_mapping[case_id] = {
                        "subtype": int(subtype[-1]),
                        "split": "train"
                    }

                except Exception as e:
                    print(f"Error processing training case '{case_id}': {e}")

    # Process validation data
    validation_cases = []
    for subtype in ['subtype0', 'subtype1', 'subtype2']:
        val_path = base_path / 'ML-Quiz-3DMedImg' / 'validation' / subtype
        if val_path.exists():
            for file in sorted(val_path.glob('*_0000.nii.gz')):
                case_id = file.stem.split('_0000')[0]
                mask_file = val_path / f"{case_id}.nii.gz"

                if not mask_file.exists():
                    print(f"Warning: Mask file not found for validation case '{case_id}'. Skipping...")
                    continue

                try:
                    if merge_validation_with_training:
                        # Merge into training
                        shutil.copy2(file, nnunet_raw_data_path / 'imagesTr' / file.name)

                        mask = nib.load(mask_file)
                        mask_data = np.round(mask.get_fdata()).astype(np.uint8)
                        mask_data = np.clip(mask_data, 0, 2)
                        new_mask = nib.Nifti1Image(mask_data, mask.affine)
                        nib.save(new_mask, nnunet_raw_data_path / 'labelsTr' / f"{case_id}.nii.gz")

                        dataset_json["training"].append({
                            "image": f"./imagesTr/{file.name}",
                            "label": f"./labelsTr/{case_id}.nii.gz"
                        })

                        subtype_mapping[case_id] = {
                            "subtype": int(subtype[-1]),
                            "split": "train"
                        }

                    else:
                        # Keep separate
                        shutil.copy2(file, nnunet_raw_data_path / 'imagesVal' / file.name)

                        mask = nib.load(mask_file)
                        mask_data = np.round(mask.get_fdata()).astype(np.uint8)
                        mask_data = np.clip(mask_data, 0, 2)
                        new_mask = nib.Nifti1Image(mask_data, mask.affine)
                        nib.save(new_mask, nnunet_raw_data_path / 'labelsVal' / f"{case_id}.nii.gz")

                        dataset_json["validation"].append({
                            "image": f"./imagesVal/{file.name}",
                            "label": f"./labelsVal/{case_id}.nii.gz"
                        })

                        subtype_mapping[case_id] = {
                            "subtype": int(subtype[-1]),
                            "split": "validation"
                        }

                    validation_cases.append(case_id)

                except Exception as e:
                    print(f"Error processing validation case '{case_id}': {e}")

    # Process test data
    test_path = base_path / 'ML-Quiz-3DMedImg' / 'test'
    if test_path.exists():
        for file in sorted(test_path.glob('*_0000.nii.gz')):
            try:
                shutil.copy2(file, nnunet_raw_data_path / 'imagesTs' / file.name)
                dataset_json["test"].append({"image": f"./imagesTs/{file.name}"})
            except Exception as e:
                print(f"Error processing test file '{file}': {e}")

    # Update counts
    dataset_json["numTraining"] = len(dataset_json["training"])
    dataset_json["numValidation"] = 0 if merge_validation_with_training else len(dataset_json["validation"])
    dataset_json["numTest"] = len(dataset_json["test"])

    # Save dataset.json
    with open(nnunet_raw_data_path / 'dataset.json', 'w') as f:
        json.dump(dataset_json, f, indent=4)

    # Save subtype mapping
    with open(nnunet_raw_data_path / 'subtype_mapping.json', 'w') as f:
        json.dump({
            "mapping": subtype_mapping,
            "validation_cases": validation_cases
        }, f, indent=4)

    # mode = "MERGED validation with training" if merge_validation_with_training else "KEPT validation separate"
    # print(f"\nMode: {mode}")
    # print(f"Training cases: {dataset_json['numTraining']}")
    # print(f"Validation cases: {dataset_json['numValidation']}")
    # print(f"Test cases: {dataset_json['numTest']}")

    return nnunet_raw_data_path, subtype_mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize data for nnUNet")
    parser.add_argument(
        "--mer", action="store_true",
        help="Merge validation data into training data (default if no flag provided)"
    )
    parser.add_argument(
        "--sep", action="store_true",
        help="Keep validation data separate"
    )
    args = parser.parse_args()

    # Default to merge unless --sep is given
    merge_mode = True if args.mer or not args.sep else False

    nnunet_dataset_path, metadata = organize_data_for_nnunet(merge_mode)

    # # Verify file structure
    # print("\nVerifying file structure:")
    # print(f"Training images: {len(list((nnunet_dataset_path / 'imagesTr').glob('*.nii.gz')))}")
    # print(f"Training labels: {len(list((nnunet_dataset_path / 'labelsTr').glob('*.nii.gz')))}")
    # if (nnunet_dataset_path / 'imagesVal').exists():
    #     print(f"Validation images: {len(list((nnunet_dataset_path / 'imagesVal').glob('*.nii.gz')))}")
    #     print(f"Validation labels: {len(list((nnunet_dataset_path / 'labelsVal').glob('*.nii.gz')))}")
    # print(f"Test images: {len(list((nnunet_dataset_path / 'imagesTs').glob('*.nii.gz')))}")
    print ('Files Created') 