import os
import warnings
from typing import Union, Tuple, List

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from threadpoolctl import threadpool_limits

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd


class nnUNetDataLoader(DataLoader):
    def __init__(self,
                 data: nnUNetBaseDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...]] = None,
                 probabilistic_oversampling: bool = False,
                 transforms=None):
        """
        If we get a 2D patch size, make it pseudo 3D and remember to remove the singleton dimension before
        returning the batch
        """
        super().__init__(data, batch_size, 1, None, True,
                         False, True, sampling_probabilities)

        if len(patch_size) == 2:
            final_patch_size = (1, *patch_size)
            patch_size = (1, *patch_size)
            self.patch_size_was_2d = True
        else:
            self.patch_size_was_2d = False

        # this is used by DataLoader for sampling train cases!
        # self.indices = data.identifiers
        self.indices = getattr(data, "identifiers", None)
        if self.indices is None:
            self.indices = list(data.keys())
        if not self.indices:
            if hasattr(data, "keys"):
                self.indices = list(data.keys())
            else:
                self.indices = list(getattr(data, "_keys", []))

        if not self.indices:
            raise RuntimeError(
                "No training identifiers found. Check that splits_final.json matches the preprocessed cases "
                "and that the dataset exposes identifiers/keys."
            )

        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the images
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if self.patch_size_was_2d:
                pad_sides = (0, *pad_sides)
            for d in range(len(self.need_to_pad)):
                self.need_to_pad[d] += pad_sides[d]
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.sampling_probabilities = sampling_probabilities
        self.annotated_classes_key = tuple([-1] + label_manager.all_labels)
        self.has_ignore = label_manager.has_ignore_label
        self.get_do_oversample = self._oversample_last_XX_percent if not probabilistic_oversampling \
            else self._probabilistic_oversampling
        self.transforms = transforms
    def _load_case_compat(self, ds, case_id):
        """
        Return (data, seg, seg_prev, properties) from ds for case_id,
        working across forks that may not implement .load_case.
        """
        # native API
        if hasattr(ds, "load_case"):
            return ds.load_case(case_id)

        # fallback: dict/tuple interface
        item = None
        if hasattr(ds, "__getitem__"):
            try:
                item = ds[case_id]
            except Exception:
                item = None
        if item is None and hasattr(ds, "get_case"):
            item = ds.get_case(case_id)
        if item is None:
            raise AttributeError("Dataset has neither load_case nor compatible item access")

        # normalize formats
        if isinstance(item, dict):
            data = item.get("data")
            seg = item.get("seg")
            seg_prev = item.get("seg_prev") or item.get("seg_from_prev_stage")
            properties = item.get("properties")
            return data, seg, seg_prev, properties

        if isinstance(item, (list, tuple)):
            if len(item) == 3:
                data, seg, properties = item
                return data, seg, None, properties
            if len(item) == 4:
                return item  # (data, seg, seg_prev, properties)

        raise TypeError("Unknown dataset item format")

    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        """
        determines whether sample sample_idx in a minibatch needs to be guaranteed foreground
        """
        return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def _probabilistic_oversampling(self, sample_idx: int) -> bool:
        # print('YEAH BOIIIIII')
        return np.random.uniform() < self.oversample_foreground_percent

    # def determine_shapes(self):
    #     # load one case
    #     # data, seg, seg_prev, properties = self._data.load_case(self._data.identifiers[0])
    #     # new
    #     case_id = self.indices[0]
    #     if hasattr(self._data, "load_case"):
    #         data, seg, seg_prev, properties = self._data.load_case(case_id)
    #     else:
    #         # fallback: dataset acts like dict
    #         item = self._data[case_id] if hasattr(self._data, "__getitem__") else self._data.get_case(case_id)
    #         if isinstance(item, dict):
    #             data = item.get("data")
    #             seg = item.get("seg")
    #             seg_prev = item.get("seg_prev") or item.get("seg_from_prev_stage")
    #             properties = item.get("properties")
    #         else:  # assume tuple
    #             if len(item) == 3:
    #                 data, seg, properties = item
    #                 seg_prev = None
    #             elif len(item) == 4:
    #                 data, seg, seg_prev, properties = item
    #             else:
    #                 raise TypeError("Unknown dataset item format")
    #     num_color_channels = data.shape[0]

    #     data_shape = (self.batch_size, num_color_channels, *self.patch_size)
    #     channels_seg = seg.shape[0]
    #     if seg_prev is not None:
    #         channels_seg += 1
    #     seg_shape = (self.batch_size, channels_seg, *self.patch_size)
    #     return data_shape, seg_shape
    def determine_shapes(self):
        # load one case
        case_id = self.indices[0]
        data, seg, seg_prev, properties = self._load_case_compat(self._data, case_id)

        num_color_channels = data.shape[0]
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        channels_seg = seg.shape[0]
        if seg_prev is not None:
            channels_seg += 1
        seg_shape = (self.batch_size, channels_seg, *self.patch_size)
        return data_shape, seg_shape

    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None],
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False):
        # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
        # locations for the given slice
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        if not force_fg and not self.has_ignore:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            # print('I want a random location')
        else:
            if not force_fg and self.has_ignore:
                selected_class = self.annotated_classes_key
                if len(class_locations[selected_class]) == 0:
                    # no annotated pixels in this case. Not good. But we can hardly skip it here
                    warnings.warn('Warning! No annotated pixels in image!')
                    selected_class = None
            elif force_fg:
                assert class_locations is not None, 'if force_fg is set class_locations cannot be None'
                if overwrite_class is not None:
                    assert overwrite_class in class_locations.keys(), 'desired class ("overwrite_class") does not ' \
                                                                      'have class_locations (missing key)'
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                # class_locations keys can also be tuple
                eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]

                # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
                # strange formulation needed to circumvent
                # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                if any(tmp):
                    if len(eligible_classes_or_regions) > 1:
                        eligible_classes_or_regions.pop(np.where(tmp)[0][0])

                if len(eligible_classes_or_regions) == 0:
                    # this only happens if some image does not contain foreground voxels at all
                    selected_class = None
                    if verbose:
                        print('case does not contain any foreground classes')
                else:
                    # I hate myself. Future me aint gonna be happy to read this
                    # 2022_11_25: had to read it today. Wasn't too bad
                    selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                        (overwrite_class is None or (overwrite_class not in eligible_classes_or_regions)) else overwrite_class
                # print(f'I want to have foreground, selected class: {selected_class}')
            else:
                raise RuntimeError('lol what!?')

            if selected_class is not None:
                voxels_of_that_class = class_locations[selected_class]
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                # i + 1 because we have first dimension 0!
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                # If the image does not contain any foreground classes, we fall back to random cropping
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs
    
    def generate_train_batch(self):
        import re
        selected_keys = self.get_indices()

        # preallocate
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)

        # helper to parse subtype from key/path
        def _infer_subtype_from_key(k: str) -> int:
            # pattern like quiz_0_041 -> grabs the '0'
            m = re.search(r'quiz_(\d)_\d+', k)
            if m:
                return int(m.group(1))
            # or folder name contains subtype0/1/2
            m = re.search(r'subtype\s*([0-2])', k)
            if m:
                return int(m.group(1))
            return -1  # unknown / ignore

        class_targets_list = []

        for j, case_id in enumerate(selected_keys):
            # oversampling foreground improves stability
            force_fg = self.get_do_oversample(j)

            data, seg, seg_prev, properties = self._load_case_compat(self._data, case_id)
            shape = data.shape[1:]

            # properties['class_locations'] may be missing on some forks
            class_locs = None
            if isinstance(properties, dict):
                class_locs = properties.get('class_locations', None)

            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, class_locs)
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            # crop/pad
            data_all[j] = crop_and_pad_nd(data, bbox, 0)
            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)[None]))
            seg_all[j] = seg_cropped

            # collect classification target (0/1/2)
            class_targets_list.append(_infer_subtype_from_key(str(case_id)))

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        class_targets = torch.as_tensor(class_targets_list, dtype=torch.long)

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images
            return {'data': data_all, 'target': seg_all, 'classTarget': class_targets, 'keys': selected_keys}

        return {'data': torch.from_numpy(data_all).float(),
                'target': torch.from_numpy(seg_all).to(torch.int16),
                'classTarget': class_targets,
                'keys': selected_keys}

    # def generate_train_batch(self):
    #     selected_keys = self.get_indices()
    #     # preallocate memory for data and seg
    #     data_all = np.zeros(self.data_shape, dtype=np.float32)
    #     seg_all = np.zeros(self.seg_shape, dtype=np.int16)

    #     for j, i in enumerate(selected_keys):
    #         # oversampling foreground will improve stability of model training, especially if many patches are empty
    #         # (Lung for example)
    #         force_fg = self.get_do_oversample(j)

    #         # data, seg, seg_prev, properties = self._data.load_case(i)
    #         # new
    #         case_id = self.indices[0]
    #         if hasattr(self._data, "load_case"):
    #             data, seg, seg_prev, properties = self._data.load_case(case_id)
    #         else:
    #             # fallback: dataset acts like dict
    #             item = self._data[case_id] if hasattr(self._data, "__getitem__") else self._data.get_case(case_id)
    #             if isinstance(item, dict):
    #                 data = item.get("data")
    #                 seg = item.get("seg")
    #                 seg_prev = item.get("seg_prev") or item.get("seg_from_prev_stage")
    #                 properties = item.get("properties")
    #             else:  # assume tuple
    #                 if len(item) == 3:
    #                     data, seg, properties = item
    #                     seg_prev = None
    #                 elif len(item) == 4:
    #                     data, seg, seg_prev, properties = item
    #                 else:
    #                     raise TypeError("Unknown dataset item format")

    #         # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
    #         # self._data.load_case(i) (see nnUNetDataset.load_case)
    #         shape = data.shape[1:]

    #         bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
    #         bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

    #         # use ACVL utils for that. Cleaner.
    #         data_all[j] = crop_and_pad_nd(data, bbox, 0)

    #         seg_cropped = crop_and_pad_nd(seg, bbox, -1)
    #         if seg_prev is not None:
    #             seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)[None]))
    #         seg_all[j] = seg_cropped

    #     if self.patch_size_was_2d:
    #         data_all = data_all[:, :, 0]
    #         seg_all = seg_all[:, :, 0]

    #     if self.transforms is not None:
    #         with torch.no_grad():
    #             with threadpool_limits(limits=1, user_api=None):
    #                 data_all = torch.from_numpy(data_all).float()
    #                 seg_all = torch.from_numpy(seg_all).to(torch.int16)
    #                 images = []
    #                 segs = []
    #                 for b in range(self.batch_size):
    #                     tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
    #                     images.append(tmp['image'])
    #                     segs.append(tmp['segmentation'])
    #                 data_all = torch.stack(images)
    #                 if isinstance(segs[0], list):
    #                     seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
    #                 else:
    #                     seg_all = torch.stack(segs)
    #                 del segs, images
    #         return {'data': data_all, 'target': seg_all, 'keys': selected_keys}

    #     return {'data': data_all, 'target': seg_all, 'keys': selected_keys}


if __name__ == '__main__':
    folder = join(nnUNet_preprocessed, 'Dataset002_Heart', 'nnUNetPlans_3d_fullres')
    ds = nnUNetDatasetBlosc2(folder)  # this should not load the properties!
    pm = PlansManager(join(folder, os.pardir, 'nnUNetPlans.json'))
    lm = pm.get_label_manager(load_json(join(folder, os.pardir, 'dataset.json')))
    dl = nnUNetDataLoader(ds, 5, (16, 16, 16), (16, 16, 16), lm,
                          0.33, None, None)
    a = next(dl)
