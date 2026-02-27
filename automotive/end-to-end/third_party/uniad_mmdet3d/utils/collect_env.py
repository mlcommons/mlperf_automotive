# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_git_hash

import mmdet
# import mmseg


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMDetection'] = mmdet.__version__
    env_info['MMSegmentation'] = 'do not import mmseg'
    env_info['MMDetection3D'] = 'my_custom_mmdet3d' + '+' + get_git_hash()[:7]

    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
