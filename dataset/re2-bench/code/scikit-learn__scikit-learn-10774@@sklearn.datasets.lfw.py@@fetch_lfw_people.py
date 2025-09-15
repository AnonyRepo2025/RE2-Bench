from os import listdir, makedirs, remove
from os.path import join, exists, isdir
import logging
import numpy as np
from .base import get_data_home, _fetch_remote, RemoteFileMetadata
from ..utils import Bunch
from ..externals.joblib import Memory
from ..externals.six import b
from ..externals._pilutil import imread, imresize
import tarfile

logger = logging.getLogger(__name__)
ARCHIVE = RemoteFileMetadata(
    filename='lfw.tgz',
    url='https://ndownloader.figshare.com/files/5976018',
    checksum=('055f7d9c632d7370e6fb4afc7468d40f'
              '970c34a80d4c6f50ffec63f5a8d536c0'))
FUNNELED_ARCHIVE = RemoteFileMetadata(
    filename='lfw-funneled.tgz',
    url='https://ndownloader.figshare.com/files/5976015',
    checksum=('b47c8422c8cded889dc5a13418c4bc2a'
              'bbda121092b3533a83306f90d900100a'))
TARGETS = (
    RemoteFileMetadata(
        filename='pairsDevTrain.txt',
        url='https://ndownloader.figshare.com/files/5976012',
        checksum=('1d454dada7dfeca0e7eab6f65dc4e97a'
                  '6312d44cf142207be28d688be92aabfa')),

    RemoteFileMetadata(
        filename='pairsDevTest.txt',
        url='https://ndownloader.figshare.com/files/5976009',
        checksum=('7cb06600ea8b2814ac26e946201cdb30'
                  '4296262aad67d046a16a7ec85d0ff87c')),

    RemoteFileMetadata(
        filename='pairs.txt',
        url='https://ndownloader.figshare.com/files/5976006',
        checksum=('ea42330c62c92989f9d7c03237ed5d59'
                  '1365e89b3e649747777b70e692dc1592')),
)

def fetch_lfw_people(data_home=None, funneled=True, resize=0.5,
                     min_faces_per_person=0, color=False,
                     slice_=(slice(70, 195), slice(78, 172)),
                     download_if_missing=True, return_X_y=False):
    """Loader for the Labeled Faces in the Wild (LFW) people dataset

    This dataset is a collection of JPEG pictures of famous people
    collected on the internet, all details are available on the
    official website:

        http://vis-www.cs.umass.edu/lfw/

    Each picture is centered on a single face. Each pixel of each channel
    (color in RGB) is encoded by a float in range 0.0 - 1.0.

    The task is called Face Recognition (or Identification): given the
    picture of a face, find the name of the person given a training set
    (gallery).

    The original images are 250 x 250 pixels, but the default slice and resize
    arguments reduce them to 62 x 47.

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    funneled : boolean, optional, default: True
        Download and use the funneled variant of the dataset.

    resize : float, optional, default 0.5
        Ratio used to resize the each face picture.

    min_faces_per_person : int, optional, default None
        The extracted dataset will only retain pictures of people that have at
        least `min_faces_per_person` different pictures.

    color : boolean, optional, default False
        Keep the 3 RGB channels instead of averaging them to a single
        gray level channel. If color is True the shape of the data has
        one more dimension than the shape with color = False.

    slice_ : optional
        Provide a custom 2D slice (height, width) to extract the
        'interesting' part of the jpeg files and avoid use statistical
        correlation from the background

    download_if_missing : optional, True by default
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    return_X_y : boolean, default=False. If True, returns ``(dataset.data,
    dataset.target)`` instead of a Bunch object. See below for more
    information about the `dataset.data` and `dataset.target` object.

        .. versionadded:: 0.20

    Returns
    -------
    dataset : dict-like object with the following attributes:

    dataset.data : numpy array of shape (13233, 2914)
        Each row corresponds to a ravelled face image of original size 62 x 47
        pixels. Changing the ``slice_`` or resize parameters will change the
        shape of the output.

    dataset.images : numpy array of shape (13233, 62, 47)
        Each row is a face image corresponding to one of the 5749 people in
        the dataset. Changing the ``slice_`` or resize parameters will change
        the shape of the output.

    dataset.target : numpy array of shape (13233,)
        Labels associated to each face image. Those labels range from 0-5748
        and correspond to the person IDs.

    dataset.DESCR : string
        Description of the Labeled Faces in the Wild (LFW) dataset.

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.20

    """
    lfw_home, data_folder_path = check_fetch_lfw(
        data_home=data_home, funneled=funneled,
        download_if_missing=download_if_missing)
    logger.debug('Loading LFW people faces from %s', lfw_home)

    # wrap the loader in a memoizing function that will return memmaped data
    # arrays for optimal memory usage
    m = Memory(cachedir=lfw_home, compress=6, verbose=0)
    load_func = m.cache(_fetch_lfw_people)

    # load and memoize the pairs as np arrays
    faces, target, target_names = load_func(
        data_folder_path, resize=resize,
        min_faces_per_person=min_faces_per_person, color=color, slice_=slice_)

    X = faces.reshape(len(faces), -1)

    if return_X_y:
        return X, target

    # pack the results as a Bunch instance
    return Bunch(data=X, images=faces,
                 target=target, target_names=target_names,
                 DESCR="LFW faces dataset")
