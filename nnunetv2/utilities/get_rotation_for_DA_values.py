import numpy as np

from nnunetv2.configuration import ANISO_THRESHOLD


def get_rotation_for_dummyDA_values(patch_size: int):
    dim = len(patch_size)

    # todo rotation should be defined dynamically based on patch size (more isotropic patch sizes = more rotation)
    if dim == 2:
        do_dummy_2d_data_aug = False
        # todo revisit this parametrization
        if max(patch_size) / min(patch_size) > 1.5:
            rotation_for_DA = {
                "x": (-15.0 / 360 * 2.0 * np.pi, 15.0 / 360 * 2.0 * np.pi),
                "y": (0, 0),
                "z": (0, 0),
            }
        else:
            rotation_for_DA = {
                "x": (-180.0 / 360 * 2.0 * np.pi, 180.0 / 360 * 2.0 * np.pi),
                "y": (0, 0),
                "z": (0, 0),
            }
        mirror_axes = (0, 1)
    elif dim == 3:
        # todo this is not ideal. We could also have patch_size (64, 16, 128) in which case a full 180deg 2d rot would be bad
        # order of the axes is determined by spacing, not image size
        do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
        if do_dummy_2d_data_aug:
            # why do we rotate 180 deg here all the time? We should also restrict it
            rotation_for_DA = {
                "x": (-180.0 / 360 * 2.0 * np.pi, 180.0 / 360 * 2.0 * np.pi),
                "y": (0, 0),
                "z": (0, 0),
            }
        else:
            rotation_for_DA = {
                "x": (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
                "y": (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
                "z": (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi),
            }
        mirror_axes = (0, 1, 2)
    else:
        raise RuntimeError()

    return rotation_for_DA, do_dummy_2d_data_aug, mirror_axes
