from tqdm.notebook import tqdm
from scipy.linalg import norm
import numpy as np
import scipy.io
import cv2
import os


def normalize(arr):
    '''
    norm array
    :param arr:
    :return:
    '''
    rng = arr.max() - arr.min()
    amin = arr.min()
    return (arr - amin) * 255 / rng


def compare_images(img1, img2):
    '''
    compare 2 images
    :param img1:
    :param img2:
    :return:
    '''
    # normalize to compensate for exposure difference, this may be unnecessary
    # consider disabling it
    #     img1 = normalize(img1)
    #     img2 = normalize(img2)
    # calculate the difference and its norms
    #     diff = signal.correlate2d(img1, img2, boundary='symm', mode='same')
    diff = img1 - img2  # elementwise for scipy arrays
    m_norm = np.sum(np.abs(diff))  # Manhattan norm
    z_norm = norm(diff.ravel(), 0)  # Zero norm
    #     print("Manhattan norm:", n_m, "/ per pixel:", n_m/img1.size)
    #     print("Zero norm:", n_0, "/ per pixel:", n_0*1.0/img1.size)

    # Two popular and relatively simple methods are:
    # (a) the Euclidean distance already suggested
    # (b) normalized cross-correlation. Normalized cross-correlation tends to be noticeably more
    # robust to lighting changes than simple cross-correlation.
    # Wikipedia gives a formula for the normalized cross-correlation.
    # More sophisticated methods exist too, but they require quite a bit more work.
    dist_euclidean = np.sqrt(np.sum((img1 - img2) ^ 2)) / img1.size
    dist_manhattan = np.sum(np.abs(img1 - img2)) / img1.size
    dist_ncc = np.sum((img1 - np.mean(img1)) * (img2 - np.mean(img2))) / ((img1.size - 1) * np.std(img1) * np.std(img2))
    return (m_norm, z_norm, dist_euclidean, dist_manhattan, dist_ncc)


def run_video(video_path):
    '''
    run video sample
    :param video_path:
    :return:
    '''
    #     video_path = '../raw_data/390Hz/Eye16/10mm/VideoFile_fps1000_0157.mat'
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
    return (m_norm_l, z_norm_l, dist_euclidean_l, dist_manhattan_l, dist_ncc_l)


def run_video_avi(video_path):
    '''
    run AVI formated video sample
    :param video_path:
    :return:
    '''
    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frame_count, frame_height, frame_width), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frame_count and ret):
        ret, frame = cap.read()
        if ret:
            buf[fc] = frame[:, :, 0]
        fc += 1

    cap.release()

    m_norm_l = list()
    z_norm_l = list()
    dist_euclidean_l = list()
    dist_manhattan_l = list()
    dist_ncc_l = list()

    for i in range(0, buf.shape[0] - 1):
        m_norm, z_norm, dist_euclidean, dist_manhattan, dist_ncc = compare_images(buf[i + 1], buf[i])
        m_norm_l.append(m_norm)
        z_norm_l.append(z_norm)
        dist_euclidean_l.append(dist_euclidean)
        dist_manhattan_l.append(dist_manhattan)
        dist_ncc_l.append(dist_ncc)
    return (m_norm_l, z_norm_l, dist_euclidean_l, dist_manhattan_l, dist_ncc_l)


def run_video_avi_fft(video_path):
    '''
    run fft on video sample
    :param video_path:
    :return:
    '''
    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frame_count, frame_height, frame_width), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frame_count and ret):
        ret, frame = cap.read()
        if ret:
            buf[fc] = frame[:, :, 0]
        fc += 1

    cap.release()

    total_abs_l = list()
    P_real_l = list()
    P_imag_l = list()
    P_complex_l = list()
    P_inverse_l = list()

    for i in range(0, buf.shape[0] - 1):
        image1 = buf[i]
        image2 = buf[i + 1]

        f1 = cv2.dft(image1.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
        f2 = cv2.dft(image2.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)

        f1_shf = np.fft.fftshift(f1)
        f2_shf = np.fft.fftshift(f2)

        f1_shf_cplx = f1_shf[:, :, 0] * 1j + 1 * f1_shf[:, :, 1]
        f2_shf_cplx = f2_shf[:, :, 0] * 1j + 1 * f2_shf[:, :, 1]

        f1_shf_abs = np.abs(f1_shf_cplx)
        f2_shf_abs = np.abs(f2_shf_cplx)
        total_abs = f1_shf_abs * f2_shf_abs

        P_real = (np.real(f1_shf_cplx) * np.real(f2_shf_cplx) +
                  np.imag(f1_shf_cplx) * np.imag(f2_shf_cplx)) / total_abs
        P_imag = (np.imag(f1_shf_cplx) * np.real(f2_shf_cplx) +
                  np.real(f1_shf_cplx) * np.imag(f2_shf_cplx)) / total_abs
        P_complex = P_real + 1j * P_imag

        P_inverse = np.abs(np.fft.ifft2(P_complex))

        total_abs_l.append(np.mean(total_abs))
        P_real_l.append(np.mean(P_real))
        P_imag_l.append(np.mean(P_imag))
        P_complex_l.append(np.mean(P_complex))
        P_inverse_l.append(np.mean(P_inverse))

    return (total_abs_l, P_real_l, P_imag_l, P_complex_l, P_inverse_l)


def Frames2SoundAviya(vid):
    '''
    prep fft
    :param vid:
    :return:
    '''
    CORRFIRST = 0

    cam = cv2.VideoCapture(vid)
    vidImages = []
    while cam.isOpened():
        ret, img = cam.read()

        if ret == True:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = img.astype(np.float32)
            vidImages.append(img)
        else:
            print('ended')
            break
    cam.release()

    vidImages = np.array(vidImages)
    vidSize = vidImages.shape

    s1 = vidImages.shape[1]
    s2 = vidImages.shape[2]
    L = vidImages.shape[0]

    meanImage = vidImages[0, :, :]

    meanImage = meanImage - np.mean(np.mean(meanImage))

    meanImage = (meanImage) / (np.sqrt(np.sum(np.sum(meanImage ** 2)))
                               )  # normaliztion
    meanImageFt = np.fft.fftshift(np.fft.fft2(
        np.fft.fftshift(meanImage)))  # ft of the mean ref image
    filMean = np.conj(meanImageFt)
    # currentImage = np.zeros(s1)
    currentImage = np.zeros((128, 128))

    texto = ''
    # peakValue = np.zeros(L)
    # pos = np.zeros(100)
    pos = np.empty(shape=(2, L))

    for g in tqdm(range(0, L)):
        # running display info
        #         if np.mod(g, 500) == 0:
        # for pp in np.arange(1,len(texto)+1).reshape(-1):
        #             for pp in range(len(texto)):
        #                 print('\b')
        #             texto = str(np.round(g / L * 100))
        #             print(texto)

        currentImage = vidImages[g, :, :]
        currentImage = currentImage - np.mean(np.mean(currentImage))
        currentImage = (currentImage) / (np.sqrt(
            np.sum(np.sum(currentImage ** 2))))  # normaliztion of each image
        currentImageft = np.fft.fftshift(
            np.fft.fft2(np.fft.fftshift(currentImage)))
        corr = np.real(
            np.fft.fftshift(
                np.fft.ifft2(np.fft.fftshift(currentImageft * filMean))))
        if CORRFIRST == 0:  # first corr in order to initializing with first frame
            filMean = np.conj(currentImageft)

        C = np.amax(corr, axis=0)
        I = np.argmax(corr, axis=1)  # np.where(corr = np.amax(corr, axis=0))

        P = np.max(C)
        I1 = np.argmax(C)  # np.where(C == np.amax(C)) #int(np.max(C))

        p1 = I[I1]
        p2 = I1

        dp1 = 0
        dp2 = 0
        if (p1 > 1) and (p1 < s1 - 1):
            # Parabolic max point (1st derivative)
            # print(p1)
            y1 = corr[p1 - 1, p2]
            y2 = corr[p1, p2]
            y3 = corr[p1 + 1, p2]
            dp1 = (y3 - y1) / 2 / (2 * y2 - y1 - y3)

        if (p2 > 1) and (p2 < s2 - 1):
            # Parabolic max point (1st derivative)
            y1 = corr[p1, p2 - 1]
            y2 = corr[p1, p2]
            y3 = corr[p1, p2 + 1]
            dp2 = (y3 - y1) / 2 / (2 * y2 - y1 - y3)
        cen1 = np.floor(s1 / 2) + 1
        cen2 = np.floor(s2 / 2) + 1
        p1 = p1 - cen1 + dp1
        p2 = p2 - cen2 + dp2

        # print(pos)
        pos[0, g] = p1
        pos[1, g] = p2

    posX = pos[0, :]
    posY = pos[1, :]
    pos[0, :] = pos[0, :] + 1

    posR = np.sqrt((pos[0, :] ** 2) + (pos[1, :] ** 2))

    locL = np.arange(L)  # [0,L]
    locLlength = len(locL)
    # corrFft=abs(np.fft.fftshift(np.fft.fft(np.fft.fftshift(peakValue))))
    corrX = abs(np.fft.fftshift(np.fft.fft(np.fft.fftshift(pos[0, :]))))
    corrY = abs(np.fft.fftshift(np.fft.fft(np.fft.fftshift(pos[1, :]))))
    corrR = abs(np.fft.fftshift(np.fft.fft(np.fft.fftshift(posR))))

    return pos, L


def prep_data(root):
    '''
    prep data
    :param root:
    :return:
    '''
    # root = '../raw_data/256px-390Hz/'
    eyes_data = dict()
    for path, subdirs, files in tqdm(os.walk(root)):
        for name in files:
            video_path = os.path.join(path, name)
            #         if 'Eye21' not in video_path: continue
            eye = path.split(sep=os.sep)[-2]
            label = path.split(sep=os.sep)[-1]
            pos, _ = Frames2SoundAviya(video_path)
            lineX = pos[0, :]
            l = (video_path, label, lineX.clip(min=0))
            eyes_data[eye] = eyes_data.get(eye, list()) + [l]
    return eyes_data


def process_data(data):
    '''
    extract relevant data from dataset
    :param data:
    :return:
    '''
    x = np.array(data[2])
    x = np.append(x, 0)
    y = int(data[1].split(sep='mm')[0])
    return x, y


def filter_high_iop_eyes(eyes_data):
    '''
    filter out eyes without high IOP measurements
    :param eyes_data:
    :return:
    '''
    higheyes = set()
    for k, v in eyes_data.items():
        #     print(v[0][0], v[0][1].split(sep='mm')[0])
        for vv in v:
            if int(vv[1].split(sep='mm')[0]) > 30:
                higheyes.add(k)
    return list(higheyes)


def set_diff_eyes_normal_high_iop(higheyes):
    '''
    split different eyes for train-test
    :param higheyes:
    :return:
    '''
    train_eyes = list(higheyes)[:4]
    test_eyes = list(higheyes)[4:]
    return train_eyes, test_eyes


def prep_normal_high_iop(eyes_data):
    '''
    prep training data for normal/high iop task
    :param eyes_data:
    :return:
    '''
    x_train, y_train = list(), list()
    x_test, y_test = list(), list()

    train_size = 0.5
    high_iop_threshold = 22

    higheyes = filter_high_iop_eyes(eyes_data)
    for eye, data in tqdm(eyes_data.items()):
        if eye in list(higheyes):
            for d in data:
                x, y = process_data(d)
                x_train.append(x[:int(x.shape[0] * train_size)])
                x_test.append(x[int(x.shape[0] * train_size):])
                if y > high_iop_threshold:
                    y_train.append(1)
                    y_test.append(1)
                else:
                    y_train.append(0)
                    y_test.append(0)

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    num_classes = len(np.unique(y_train))
    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]
    return x_train, y_train, x_test, y_test


def prep_min_diff_iop_single_eye(eyes_data, higheyes):
    '''
    prep data for multiple IOP groups task
    :param eyes_data:
    :param higheyes:
    :return:
    '''
    x_train, y_train = list(), list()
    x_test, y_test = list(), list()

    train_size = 0.5

    for eye, data in tqdm(eyes_data.items()):
        if eye in list(higheyes):
            for d in data:
                x, y = process_data(d)
                x_train.append(x[:int(x.shape[0] * train_size)])
                x_test.append(x[int(x.shape[0] * train_size):])
                if y <= 14:  # 10-14mmHg - class 0
                    y_train.append(0)
                    y_test.append(0)
                elif 15 <= y <= 19:  # 15-19mmHg - class 1
                    y_train.append(1)
                    y_test.append(1)
                elif 20 <= y <= 22:  # 20-22mmHg - class 2
                    y_train.append(2)
                    y_test.append(2)
                elif 22 < y <= 28:  # 24-28mmHg - class 3
                    y_train.append(3)
                    y_test.append(3)
                elif y > 28:  # 32-40mmHg - class 4
                    y_train.append(4)
                    y_test.append(4)

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    num_classes = len(np.unique(y_train))
    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]
    return x_train, y_train, x_test, y_test


def prep_1mm_diff_single_eye(eyes_data):
    '''
    prep data for 1mm difference IOP task
    :param eyes_data:
    :return:
    '''
    x_train, y_train = list(), list()
    x_test, y_test = list(), list()

    train_size = 0.5

    for eye, data in tqdm(eyes_data.items()):
        # if not eye == 'Eye1': continue
        for d in data:
            x, y = process_data(d)
            for i in np.array_split(x[:int(x.shape[0] * train_size)], 10):
                if i.shape[0] < 8700:
                    i = np.pad(i, (0, 1))
                x_train.append(i)
                y_train.append(y - 12)
            for i in np.array_split(x[int(x.shape[0] * train_size):], 10):
                if i.shape[0] < 8700:
                    i = np.pad(i, (0, 1))
                x_test.append(i)
                y_test.append(y - 12)

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    num_classes = len(np.unique(y_train))
    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]
    return x_train, y_train, x_test, y_test
