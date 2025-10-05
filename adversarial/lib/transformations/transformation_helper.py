#!/usr/bin/env python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torchvision.transforms as torch_trans
import lib.transformations.transforms as transforms
from lib.datasets.transform_dataset import TransformDataset


try:
    _ResizeTransform = torch_trans.Resize
except AttributeError:  # torchvision < 0.8
    _ResizeTransform = torch_trans.Scale

try:
    _RandomResizedCrop = torch_trans.RandomResizedCrop
except AttributeError:  # torchvision < 0.8
    _RandomResizedCrop = torch_trans.RandomSizedCrop


# Initialize transformations to be applied to dataset
def setup_transformations(args, data_type, defense, crop=None):
    if 'preprocessed_data' in args and args.preprocessed_data:
        assert defense is not None, (
            "If data is already pre processed for defenses then "
            "defenses can't be None")
    if crop:
        assert callable(crop), "crop should be a callable method"

    transform = []
    # setup transformation without adversary
    if 'adversary' not in args or args.adversary is None:
        if (data_type == 'train'):
            if 'preprocessed_data' in args and args.preprocessed_data:
                # Defenses are already applied on randomly cropped images
                transform.append(_ResizeTransform(args.data_params['IMAGE_SIZE']))
            else:
                transform.append(
                    _RandomResizedCrop(args.data_params['IMAGE_SIZE']))

            transform.append(torch_trans.RandomHorizontalFlip())
            transform.append(torch_trans.ToTensor())

        else:  # validation
            # No augmentation for validation
            if 'preprocessed_data' not in args or not args.preprocessed_data:
                transform.append(_ResizeTransform(args.data_params['IMAGE_SCALE_SIZE']))
                transform.append(torch_trans.CenterCrop(
                    args.data_params['IMAGE_SIZE']))
            transform.append(torch_trans.ToTensor())
            if crop:
                transform.append(crop)

            transform.append(transforms.Scale(args.data_params['IMAGE_SIZE']))

        # Apply defenses at runtime (VERY SLOW)
        #  Prefer pre-processing and saving data, and then using it
        if ('preprocessed_data' in args and not args.preprocessed_data and
                defense is not None):
            transform = transform + [defense]

    else:  # Adversarial images
        if crop is not None:
            transform.append(crop)

        transform.append(transforms.Scale(args.data_params['IMAGE_SIZE'],
                                          args.data_params['MEAN_STD']))

        # Apply defenses at runtime (VERY SLOW)
        #  Prefer pre-processing and saving data, and then using it
        if not args.preprocessed_data and defense is not None:
            transform.append(defense)

    if 'normalize' in args and args.normalize:
        transform.append(
            torch_trans.Normalize(mean=args.data_params['MEAN_STD']['MEAN'],
                                    std=args.data_params['MEAN_STD']['STD']))

    if len(transform) == 0:
        transform = None
    else:
        transform = torch_trans.Compose(transform)

    return transform


# Update dataset
def update_dataset_transformation(dataset, args, data_type,
                                    defense, crop):

    # only supported for TransformDataset at the moment
    assert isinstance(dataset, TransformDataset), (
        "updating datase transformation is only supported for TransformDataset"
        "for adversaries")

    assert data_type is not 'train', \
        "updating datase transformation is not supported in training"

    transform = setup_transformations(args, data_type, defense, crop)
    dataset.update_transformation(transform=transform)
