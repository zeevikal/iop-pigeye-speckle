from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from tensorflow import keras
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.io
import cv2
import os

from utils import run_video_avi, Frames2SoundAviya, compare_images, run_video
from eval import interp1d

def plot_sample(video_path):
    '''
    plot data sample
    :param video_path:
    :return:
    '''
    # video_path = '../data/0001 (1).avi'

    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frame_count, frame_height, frame_width), np.dtype('uint8'))

    fc = 0
    ret = True

    frame_count = 10
    while (fc < frame_count and ret):
        ret, frame = cap.read()
        if ret:
            buf[fc] = frame[:, :, 0]
            plt.imshow(buf[fc], cmap='inferno')
            plt.savefig('../models/raw_data_sample/sample_{}.png'.format(fc), dpi=900)
        #         plt.show()
        fc += 1

    cap.release()

def plot_vid(video_path):
    '''
    process and plot video
    :param video_path:
    :return:
    '''
    # video_path = '../data/0001 (1).avi'
    m_norm_l, z_norm_l, dist_euclidean_l, dist_manhattan_l, dist_ncc_l = run_video_avi(video_path)
    plt.plot(m_norm_l / max(m_norm_l))
    plt.plot(m_norm_l/max(m_norm_l))
    plt.show()

def plot_frame2sound(video_path, fs=1000):
    '''
    process and plot video using fft
    :param video_path:
    :param fs:
    :return:
    '''
    pos, L = Frames2SoundAviya(video_path)
    dt = 1 / fs
    lineX = pos[0, :]
    lineY = pos[1, :]
    t = np.linspace(0, len(lineX), len(lineX)) * dt
    plt.plot(t, lineX)
    plt.show()

def prep_dists(video_path):
    '''
    prep morm data
    :param video_path:
    :return:
    '''
    # video_path = '../raw_data/390Hz/Eye16/10mm/VideoFile_fps1000_0157.mat'
    mat = scipy.io.loadmat(video_path)

    m_norm_l = list()
    z_norm_l = list()
    dist_euclidean_l = list()
    dist_manhattan_l = list()
    dist_ncc_l = list()

    for k, v in mat.items():
        if 'Video' in k:
            for i in range(0, v.shape[2] - 1):
                m_norm, z_norm, dist_euclidean, dist_manhattan, dist_ncc = compare_images(v[:, :, i + 1], v[:, :, i])
                m_norm_l.append(m_norm)
                z_norm_l.append(z_norm)
                dist_euclidean_l.append(dist_euclidean)
                dist_manhattan_l.append(dist_manhattan)
                dist_ncc_l.append(dist_ncc)
    return m_norm_l, z_norm_l, dist_euclidean_l, dist_manhattan_l, dist_ncc_l

def plot_dists(video_path):
    '''
    plot norm data
    :param video_path:
    :return:
    '''
    m_norm_l, z_norm_l, dist_euclidean_l, dist_manhattan_l, dist_ncc_l = prep_dists(video_path)
    plt.plot(m_norm_l / max(m_norm_l))
    plt.plot(z_norm_l / max(z_norm_l))
    plt.plot(dist_euclidean_l / max(dist_euclidean_l))
    plt.plot(dist_manhattan_l / max(dist_manhattan_l))
    plt.plot(dist_ncc_l / max(dist_ncc_l))
    plt.show()

def plot_dist_jump_frames(m_norm_l, jump_by=1000):
    '''
    plot data while skipping frames
    :param m_norm_l:
    :param jump_by:
    :return:
    '''
    jump_by = 1000  # frames
    for i in range(0, len(m_norm_l), jump_by):
        print('num frames: ', len(m_norm_l[i:i + jump_by]))
        plt.plot(m_norm_l[i:i + jump_by] / max(m_norm_l))
        plt.ylabel('Manhattan norm')
        plt.show()

def plot_eye_data(eye_path):
    '''
    plot single eye's data
    :param eye_path:
    :return:
    '''
    # eye_path = '../raw_data/390Hz/Eye15/'
    for path, subdirs, files in tqdm(os.walk(eye_path)):
        for name in files:
            m_norm_l, z_norm_l, dist_euclidean_l, dist_manhattan_l, dist_ncc_l = run_video(os.path.join(path, name))
            plt.plot(m_norm_l / max(m_norm_l), label=path.split(sep=os.sep)[-1])
    plt.legend(loc="best")
    plt.show()

def plot_normal_high_iop_sample(x_train, y_train, y_test, save_fig_path):
    '''
    plot normal high iop sample
    :param x_train:
    :param y_train:
    :param y_test:
    :param save_fig_path:
    :return:
    '''
    classes = np.unique(np.concatenate((y_train, y_test), axis=0))

    plt.figure()
    for c in classes:
        c_x_train = x_train[y_train == c]
        plt.plot(c_x_train[0][:100], label="IOP " + str(c), alpha=0.5)
    plt.legend(loc="best")
    plt.savefig(save_fig_path, dpi=900)
    plt.show()

def plot_model_architecture(model):
    '''
    plot model architecture
    :param model:
    :return:
    '''
    keras.utils.plot_model(model, show_shapes=True)

def plot_train_stats(history, save_fig_path):
    '''
    plot training process statistics
    :param history:
    :param save_fig_path:
    :return:
    '''
    metric = "sparse_categorical_accuracy"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.savefig(save_fig_path, dpi=900)
    plt.show()

def cm_analysis(y_true, y_pred, labels, save_fig_path, ymap=None, figsize=(10,10)):
    '''
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    '''
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
#     cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm = pd.DataFrame(cm, index=['Normal IOP', 'High IOP'], columns=['Normal IOP', 'High IOP'])
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.savefig(save_fig_path, dpi=900)
    plt.show()

def plot_model_filters_on_sample(test_sample, feature_maps):
    '''
    plot trained model filters on sample
    :param test_sample:
    :param feature_maps:
    :return:
    '''
    plt.plot(test_sample / test_sample.max(), alpha=0.5)
    for i, f in enumerate(feature_maps):
        for ff in f[:20]:
            ff = ff.clip(min=0)
            ff = ff / ff.max()
            ff = interp1d(ff, test_sample.shape[0])
            plt.plot(ff, alpha=0.5)

def plot_model_filters_on_sample_with_threshold(test_sample, feature_maps, threshold=0.8):
    '''
    plot trained model filters on sample with threshold
    :param test_sample:
    :param feature_maps:
    :param threshold:
    :return:
    '''
    plt.plot(test_sample / test_sample.max(), alpha=0.5)
    for i, f in enumerate(feature_maps):
        for ff in f[:20]:
            ff = ff.clip(min=0)
            ff = ff / ff.max()
            ff[ff < threshold] = 0
            ff = interp1d(ff, test_sample.shape[0])
            plt.plot(ff, alpha=0.5)
