#==========================================================================
# This file is under License LGPL-3.0 (see details in the license file).
# This file is a part of implementation for paper:
# How To Train Your Deep Multi-Object Tracker.
# This contribution is headed by Perception research team, INRIA.
# Contributor(s) : Yihong Xu
# INRIA contact  : yihong.xu@inria.fr
# created on 16th April 2020.
# the code is modified based on:
# https://github.com/phil-bergmann/tracking_wo_bnw/tree/iccv_19
# https://github.com/jwyang/faster-rcnn.pytorch/
#==========================================================================

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import, division, print_function

import numpy as np

from .coco import coco
from .imagenet import imagenet
from .mot import MOT17, MOT19CVPR
from .pascal_voc import pascal_voc
from .vg import vg

__sets = {}


# Set up voc_<year>_<split>
for year in ['2007', '2012', '0712']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2014_cap_<split>
for year in ['2014']:
  for split in ['train', 'val', 'capval', 'valminuscapval', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
  for split in ['test', 'test-dev']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# MOT17 dataset
mot17_splits = ['train', 'frame_val', 'frame_train', 'test', 'all']
# we generate 7 train/val splits for cross validation with single sequence val sets
# ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']
for i in range(1, 8):
  mot17_splits += [f'seq_train_{i}']
  mot17_splits += [f'seq_val_{i}']
for year in ['2017']:
  for split in mot17_splits:
    name = f'mot_{year}_{split}'
    __sets[name] = (lambda split=split, year=year: MOT17(split, year))

# MOT19_CVPR dataset
mot19_cvpr_splits = ['train', 'frame_val', 'frame_train', 'test', 'all']
# we generate 4 train/val splits for cross validation with single sequence val sets
# ['CVPR-01', 'CVPR-02', 'CVPR-03', 'CVPR-05']
for i in range(1, 5):
  mot19_cvpr_splits += [f'seq_train_{i}']
  mot19_cvpr_splits += [f'seq_val_{i}']
for split in mot19_cvpr_splits:
  name = f'mot19_cvpr_{split}'
  __sets[name] = (lambda split=split: MOT19CVPR(split))

# Set up vg_<split>
# for version in ['1600-400-20']:
#     for split in ['minitrain', 'train', 'minival', 'val', 'test']:
#         name = 'vg_{}_{}'.format(version,split)
#         __sets[name] = (lambda split=split, version=version: vg(version, split))
for version in ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']:
    for split in ['minitrain', 'smalltrain', 'train', 'minival', 'smallval', 'val', 'test']:
        name = 'vg_{}_{}'.format(version,split)
        __sets[name] = (lambda split=split, version=version: vg(version, split))

# set up image net.
for split in ['train', 'val', 'val1', 'val2', 'test']:
    name = 'imagenet_{}'.format(split)
    devkit_path = 'data/imagenet/ILSVRC/devkit'
    data_path = 'data/imagenet/ILSVRC'
    __sets[name] = (lambda split=split, devkit_path=devkit_path, data_path=data_path: imagenet(split,devkit_path,data_path))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
