from typing import Tuple, Union
import warnings

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import (
    nnUNetTrainerDA5,
    nnUNetTrainerDA5ord0,
)
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


class nnUNetDataLoaderBetterIgnoreSampling(nnUNetDataLoader):
    # Port of old nnUNetDataLoaderBaseBetterIgnSampling.get_bbox from
    # nnUNetTrainer_betterIgnoreSampling.py, old lines 16-107.
    #
    # The old file implemented this once on nnUNetDataLoaderBase and then routed
    # both the 2D and 3D dataloaders through it. Current nnU-Net uses the unified
    # nnUNetDataLoader, so overriding this single method covers both cases.
    def get_bbox(
        self,
        data_shape: np.ndarray,
        force_fg: bool,
        class_locations: Union[dict, None],
        overwrite_class = None,
        verbose: bool = False,
    ):
        # Same as old lines 21-33: compute the valid lower/upper crop bounds.
        # need_to_pad may be increased for very small images so a full patch can
        # still be sampled and later padded by crop_and_pad_nd.
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        lbs = [-need_to_pad[i] // 2 for i in range(dim)]
        ubs = [
            data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i]
            for i in range(dim)
        ]

        # Same as old lines 35-40: if there is no forced foreground sampling and
        # no ignore label, choose a fully random crop inside the valid bounds.
        if not force_fg and not self.has_ignore:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
        else:
            # Same role as old lines 41-43, with one added guard for missing
            # class_locations. With ignore labels, non-foreground samples should
            # still be taken from annotated voxels instead of arbitrary image
            # background. The annotated_classes_key is the region containing all
            # non-ignore labels.
            if class_locations is None:
                selected_class = None
            elif not force_fg and self.has_ignore:
                selected_class = self.annotated_classes_key
                if len(class_locations.get(selected_class, ())) == 0:
                    warnings.warn("Warning! No annotated pixels in image!")
                    selected_class = None
            elif force_fg:
                # Same as old lines 44-78: choose a foreground class or region
                # that is present in this case. If the annotated_classes_key is
                # available together with real foreground classes, exclude it so
                # oversampling still targets actual foreground.
                assert class_locations is not None, "if force_fg is set class_locations cannot be None"
                if overwrite_class is not None:
                    assert overwrite_class in class_locations.keys(), (
                        'desired class ("overwrite_class") does not have class_locations (missing key)'
                    )

                eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]
                annotated_key_matches = [
                    i == self.annotated_classes_key if isinstance(i, tuple) else False
                    for i in eligible_classes_or_regions
                ]
                if any(annotated_key_matches) and len(eligible_classes_or_regions) > 1:
                    eligible_classes_or_regions.pop(np.where(annotated_key_matches)[0][0])

                if len(eligible_classes_or_regions) == 0:
                    selected_class = None
                    if verbose:
                        print("case does not contain any foreground classes")
                else:
                    selected_class = (
                        eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))]
                        if overwrite_class is None or overwrite_class not in eligible_classes_or_regions
                        else overwrite_class
                    )
            else:
                raise RuntimeError("unexpected foreground sampling state")

            voxels_of_that_class = (
                class_locations[selected_class]
                if class_locations is not None and selected_class is not None
                else None
            )

            if voxels_of_that_class is not None:
                # Same as old line 79, with .copy() added because the latest
                # implementation mutates selected_voxel below and should not
                # write back into properties["class_locations"].
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))].copy()

                if self.has_ignore and not force_fg:
                    # Same as old lines 80-95: this is the actual "better ignore
                    # sampling" change. Instead of always centering the patch on
                    # the selected annotated voxel, randomly offset that voxel by
                    # up to half a patch in each spatial dimension. This keeps
                    # the sampled patch inside annotated image content while
                    # avoiding repeated crops locked to the same voxel center.
                    allowed_max_neg_offset = [
                        min(s, p // 2) for s, p in zip(selected_voxel[1:], self.patch_size)
                    ]
                    allowed_max_pos_offset = [
                        min(d - s, p // 2)
                        for s, p, d in zip(selected_voxel[1:], self.patch_size, data_shape)
                    ]
                    for d in range(len(self.patch_size)):
                        offset_low = -allowed_max_neg_offset[d]
                        offset_high = allowed_max_pos_offset[d]
                        if offset_low < offset_high:
                            selected_voxel[d + 1] += np.random.randint(offset_low, offset_high)

                # Same as old lines 97-100: use the selected voxel as the patch
                # center and convert it into lower bbox coordinates, clipped to
                # the allowed lower bound. selected_voxel[0] is the class index
                # stored by nnU-Net, so spatial coordinates start at index 1.
                bbox_lbs = [
                    max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2)
                    for i in range(dim)
                ]
            else:
                # Same as old lines 101-103: if there is no valid foreground or
                # annotated voxel, fall back to random cropping.
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        # Same as old lines 105-107: upper bbox coordinates are lower + patch.
        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
        return bbox_lbs, bbox_ubs


class nnUNetTrainer_betterIgnoreSampling(nnUNetTrainer):
    def get_dataloaders(self):
        # Current-version replacement for the old get_plain_dataloaders-only
        # hook from nnUNetTrainer_betterIgnoreSampling.py, old lines 126-160.
        #
        # In newer nnU-Net versions, nnUNetTrainer.get_dataloaders constructs
        # transforms and dataloaders directly. Therefore this trainer mirrors the
        # current base get_dataloaders implementation, but swaps in
        # nnUNetDataLoaderBetterIgnoreSampling below.
        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # Current nnU-Net transform setup. This did not exist in this form in
        # the old trainer override, where transforms were still applied outside
        # the plain dataloaders.
        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )

        # Create the custom dataloaders with the current transform objects.
        # This is the new-version equivalent of old lines 130-160, but without
        # separate 2D/3D dataloader classes because nnUNetDataLoader now handles
        # both. The validation loader uses final patch size directly, same as
        # the old validation branches.
        dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size, tr_transforms, val_transforms)

        # Current nnU-Net augmenter wrapping, copied from the base trainer so
        # this variant differs only in the dataloader class and bbox sampling.
        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val,
                transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )

        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def get_plain_dataloaders(
        self,
        initial_patch_size: Tuple[int, ...],
        tr_transforms: BasicTransform,
        val_transforms: BasicTransform,
    ):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        # Equivalent to old 2D/3D training dataloader construction at old lines
        # 130-137 and 145-152. The important behavioral difference from stock
        # nnU-Net is the custom dataloader class; all current-version arguments
        # such as transforms and probabilistic_oversampling are preserved.
        dl_tr = nnUNetDataLoaderBetterIgnoreSampling(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )
        # Equivalent to old validation dataloader construction at old lines
        # 138-144 and 153-159. Validation uses patch_size for both requested and
        # final patch size because no larger augmentation crop is needed.
        dl_val = nnUNetDataLoaderBetterIgnoreSampling(
            dataset_val,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )
        return dl_tr, dl_val


class nnUNetTrainer_betterIgnoreSampling_noSmooth(nnUNetTrainer_betterIgnoreSampling):
    def _build_loss(self):
        # Port of old noSmooth loss variant from old lines 163-188. Only the
        # Dice smooth parameter is changed to 0; the dataloader behavior comes
        # from nnUNetTrainer_betterIgnoreSampling.
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss(
                {},
                {
                    "batch_dice": self.configuration_manager.batch_dice,
                    "do_bg": True,
                    "smooth": 0,
                    "ddp": self.is_ddp,
                },
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
        else:
            loss = DC_and_CE_loss(
                {
                    "batch_dice": self.configuration_manager.batch_dice,
                    "smooth": 0,
                    "do_bg": False,
                    "ddp": self.is_ddp,
                },
                {},
                weight_ce=1,
                weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss,
            )

        deep_supervision_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
        weights = weights / weights.sum()
        return DeepSupervisionWrapper(loss, weights)


class nnUNetTrainerDA5_betterIgnoreSampling(nnUNetTrainer_betterIgnoreSampling, nnUNetTrainerDA5):
    # Current-version equivalent of old lines 191-225. Multiple inheritance lets
    # DA5 provide the augmentation settings while this class keeps the custom
    # get_dataloaders/dataloader behavior.
    pass


class nnUNetTrainerDA5ord0_betterIgnoreSampling(nnUNetTrainer_betterIgnoreSampling, nnUNetTrainerDA5ord0):
    # Current-version equivalent of old lines 228-262 for the DA5 order-0
    # variant, again combining DA5ord0 augmentation with better-ignore sampling.
    pass


class nnUNetTrainer_betterIgnoreSampling_10epochs(nnUNetTrainer_betterIgnoreSampling):
    # Debug/training-length variant matching old lines 265-270, adapted to the
    # current nnUNetTrainer constructor signature.
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 10

class nnUNetTrainer_betterIgnoreSampling_100epochs(nnUNetTrainer_betterIgnoreSampling):
    # Same as the 10-epoch debug variant, but for a 100-epoch run.
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
