import numpy as np
import nibabel as nib
import subprocess

from .path import remove_ext
from concurrent.futures.thread import ThreadPoolExecutor


def run_bash(cmd, v=True):
    if v:
        subprocess.check_call(['bash', '-c', cmd])
    else:
        subprocess.check_call(['bash', '-c', cmd], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def parallel_run(func, args, num_threads):
    """Runs func in parallel with the given args and returns an ordered list with the returned values."""
    # Assert list of lists to comply with variadic positional arguments (i.e. the * in fn(*args))
    assert all([isinstance(arg, list) for arg in args]), 'Function arguments must be given as list'
    assert callable(func), 'func must be a callable function'
    # Define output variable and load function wrapper to maintain correct list order
    results = [None] * len(args)
    def _run_load_func(n_, args_):
        results[n_] = func(*args_)
    # Parallel run func and store the results in the right place
    pool = ThreadPoolExecutor(max_workers=num_threads)
    future_tasks = [pool.submit(_run_load_func, n, args) for n, args in enumerate(args)]
    # Check if any exceptions occured during execution
    [future_task.result() for future_task in future_tasks]
    pool.shutdown(wait=True)
    return results


def save_nifti(filepath, arr, dtype=None, reference=None, channel_handling='none'):
    """Saves the given volume array as a Nifti1Image using nibabel.

    :param str filepath: filename where the nifti will be saved
    :param numpy.ndarray arr: the array with shape (X, Y, Z) or (CH, X, Y, Z) to save in a nifti image
    :param dtype: (optional) data type for the stored image (default: same dtype as `image`)
    :param nibabel.Nifti1Image reference: (optional) reference nifti from where to take the affine transform and header
    :param str channel_handling: (default: ``'none'``) One of ``'none'``, ``'last'`` or ``'split'``.
        If ``none``, the array is stored in the nifti as given. If  ``'last'`` the channel dimension is put last, this
        is useful to visualize images as multi-component data in *ITK-SNAP*. If ``'split'``, then the image channels
        are each stored in a different nifti file.
    """

    # Multichannel image handling
    assert channel_handling in {'none', 'last', 'split'}
    if len(arr.shape) == 4 and channel_handling != 'none':
        if channel_handling == 'last':
            arr = np.transpose(arr, axes=(1, 2, 3, 0))
        elif channel_handling == 'split':
            for n, channel in enumerate(arr):
                savename = '{}_ch{}.nii.gz'.format(remove_ext(filepath), n)
                save_nifti(savename, channel, dtype=dtype, reference=reference)
            return

    if dtype is not None:
        arr = arr.astype(dtype)

    if reference is None:
        nifti = nib.Nifti1Image(arr, np.eye(4))
    else:
        nifti = nib.Nifti1Image(arr, reference.affine, reference.header)

    print("Saving nifti: {}".format(filepath))
    nifti.to_filename(filepath)


def load_nifti(filepath, arr, dtype=None, reference=None, channel_handling='none'):
    """Saves the given volume array as a Nifti1Image using nibabel.

    :param str filepath: filename where the nifti will be saved
    :param numpy.ndarray arr: the array with shape (X, Y, Z) or (CH, X, Y, Z) to save in a nifti image
    :param dtype: (optional) data type for the stored image (default: same dtype as `image`)
    :param nibabel.Nifti1Image reference: (optional) reference nifti from where to take the affine transform and header
    :param str channel_handling: (default: ``'none'``) One of ``'none'``, ``'last'`` or ``'split'``.
        If ``none``, the array is stored in the nifti as given. If  ``'last'`` the channel dimension is put last, this
        is useful to visualize images as multi-component data in *ITK-SNAP*. If ``'split'``, then the image channels
        are each stored in a different nifti file.
    """

    raise NotImplementedError

    # Multichannel image handling
    assert channel_handling in {'none', 'last', 'split'}
    if len(arr.shape) == 4 and channel_handling != 'none':
        if channel_handling == 'last':
            arr = np.transpose(arr, axes=(1, 2, 3, 0))
        elif channel_handling == 'split':
            for n, channel in enumerate(arr):
                savename = '{}_ch{}.nii.gz'.format(remove_ext(filepath), n)
                save_nifti(savename, channel, dtype=dtype, reference=reference)
            return

    if dtype is not None:
        arr = arr.astype(dtype)

    if reference is None:
        nifti = nib.Nifti1Image(arr, np.eye(4))
    else:
        nifti = nib.Nifti1Image(arr, reference.affine, reference.header)

    print("Saving nifti: {}".format(filepath))
    nifti.to_filename(filepath)

if __name__ == '__main__':
    print(parallel_run(lambda x: x/2, [[1], [2], [3]], num_threads=3))