from multiprocessing import Pool, TimeoutError
import time
import os
import re
import numpy as np
import dicom
from scipy.misc import imresize
from collections import namedtuple
from tqdm import tqdm
from skimage.restoration import denoise_tv_chambolle


DATA_DIR = '../'
Meta = namedtuple('Meta', ['age', 'sex', 'px', 'py', 'w', 'h', 'st'])
IMG_SHAPE = (64, 64)


def load_images(files):
    def crop_resize(img):
        if img.shape[0] < img.shape[1]:
            img = img.T
         # we crop image from center
        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
        img = crop_img
        # be careful! imresize will return UINT8 array
        img = imresize(img, IMG_SHAPE)
        return img

    images = []

    dc = dicom.read_file(files[0])
    age = getattr(dc, 'PatientAge', '000Y')
    sex = getattr(dc, 'PatientSex', 'M')
    px, py = getattr(dc, 'PixelSpacing', (1.5, 1.5))
    px, py = float(px), float(py)
    w, h = dc.pixel_array.shape
    st = int(getattr(dc, 'SliceThickness', 8))

    meta = Meta(age, sex, px, py, w, h, st)

    for f in files:
        try:
            dc = dicom.read_file(f)
            image = dc.pixel_array.astype(np.float32, copy=False)
            image = image / np.max(image)
            resized = crop_resize(image)
            resized = resized.astype(np.float32, copy=False)
            resized /= np.max(resized)
            denoised = denoise_tv_chambolle(resized, weight=0.1, multichannel=False)
            images.append(denoised)
        except Exception as err:
            print('ERROR', err, f)
            continue

    return (meta, images)


def read_sax_folder(path):
    files = []
    for x in os.listdir(path):
        r = re.search('(\d{4,})-(\d{4})\.dcm', x)
        if r is None:
            continue
        m = int(r.group(2))
        files.append((m, x))

    files = [t[1] for t in sorted(files)]
    return files


def read_data_folder(path):
    all_studies = []
    all_saxes = []

    for x in os.listdir(path):
        r = re.match('\d+', x)
        if r is None:
            continue
        else:
            all_studies.append((int(r.group()), x))

    all_studies.sort()
    for (study_id, study_dir) in all_studies:
        p = os.path.join(path, study_dir, 'study')
        study_saxes_paths = []
        for x in os.listdir(p):
            r = re.match('sax_(\d+)', x)
            if r is None:
                continue
            m = int(r.group(1))
            study_saxes_paths.append((m, x))
        study_saxes_paths = [(study_id, os.path.join(p, t[1])) for t in sorted(study_saxes_paths)]
        all_saxes.extend(study_saxes_paths)

    chunks = []
    for (study_id, sax_path) in all_saxes:
        lst = []
        sax_files = read_sax_folder(sax_path)
        if len(sax_files) == 30:
            lst.append(sax_files)
        if len(sax_files) < 30:
            lst.append(sax_files + sax_files[:30 - len(sax_files)])
        if len(sax_files) > 30:
            lst.extend(zip(*[iter(sax_files)] * 30))
        for chunk in lst:
            chunks.append((study_id, sax_path, chunk))

    return chunks


def process_the_chunk(chunk):
    study_id, root, files = chunk
    data, imgs = load_images([os.path.join(root, f) for f in files])
    return (study_id, data, np.array(imgs).astype(np.float32, copy=False))



def write_train_npy(prefix):
    """
        Prepare the train data and store at NPY
    """
    print('-'*50)
    print('Prepare train data')
    print('-'*50)

    print('Read train.csv')
    train_table = {}
    with open(os.path.join(DATA_DIR, 'train.csv'), 'r') as fin:
        for line in fin.readlines()[1:]:
            s = line.replace('\n', '').split(',')
            train_table[int(s[0])] = (float(s[1]), float(s[2]))
    print('Done.')
    print('-'*50)
    print('List folders')
    chunks = read_data_folder(os.path.join(DATA_DIR, 'train'))
    print('Done')

    print('*' * 50)
    print('Process the chunks:...')
    t0 = time.time()
    pool = Pool(processes=4)
    it = pool.imap_unordered(process_the_chunk, chunks)
    work = list(tqdm(it))
    pool.close()
    pool.join()
    t1 = time.time()
    print('Done for {}s'.format(t1 - t0))

    X = []
    y = []
    print ('*'*50)
    print('Post-process data')
    for (study_id, meta, images) in tqdm(work):
        systolic, diastolic = train_table[study_id]
        mm2 = (meta.px * min([meta.w, meta.h]) / IMG_SHAPE[0]) ** 2
        mm3 = mm2 * meta.st
        X.append(images)
        y.append((systolic, diastolic, systolic / mm2, diastolic / mm2, systolic / mm3, diastolic / mm3))
    y = np.array(y)
    X = np.array(X)
    X *= 255.0
    X = np.array(X).astype(np.uint8, copy=False)

    np.save(prefix + 'X-train.npy', X)
    print(prefix + 'X-train.npy contains preprocessed array with shape [N, 30, 64, 64] at np.uint8')
    np.save(prefix + 'y-train.npy', y)
    print(prefix + 'y-train.npy contains [sys, dia, sys / mm2, dia / mm2, sys / mm3, dia / mm3] at np.float32')
    print('Done.')

def write_validate_npy(prefix):
    """
        Prepare the validate data and store at NPY
    """
    print('-'*50)
    print('Prepare train data')
    print('-'*50)


    print('List folders')
    chunks = read_data_folder(os.path.join(DATA_DIR, 'validate'))
    print('Done')

    print('*' * 50)
    print('Process the chunks:...')
    t0 = time.time()
    pool = Pool(processes=4)
    it = pool.imap_unordered(process_the_chunk, chunks)
    work = list(tqdm(it))
    pool.close()
    pool.join()
    t1 = time.time()
    print('Done for {}s'.format(t1 - t0))

    X = []
    M = []
    print ('*'*50)
    print('Post-process data')
    for (study_id, meta, images) in tqdm(work):
        mm2 = (meta.px * min([meta.w, meta.h]) / IMG_SHAPE[0]) ** 2
        mm3 = mm2 * meta.st
        X.append(images)
        M.append((study_id, mm2, mm3))

    M = np.array(M)
    X = np.array(X)
    X *= 255.0
    X = np.array(X).astype(np.uint8, copy=False)

    np.save(prefix + 'X-validate.npy', X)
    print(prefix + 'X-validate.npy contains preprocessed array with shape [N, 30, 64, 64] at np.uint8')
    np.save(prefix + 'm-validate.npy', M)
    print(prefix + 'm-train.npy contains [study_id, mm2, mm3] at np.float32')
    print('Done.')


if __name__ == '__main__':
    prefix = 'dry-run/pre-'

    write_train_npy(prefix)
    write_validate_npy(prefix)