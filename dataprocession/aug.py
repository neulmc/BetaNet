import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.io as scio
from functools import reduce
import math
import pandas as pd
import time
import multiprocessing.dummy as mp

arg_dir = '../BetaBSDS/' # output_dir
root_dir_images = '../rawbsds/images/train_val' # rawbsds_dir
root_dir_gts = '../rawbsds/groundTruth/train_val'

img_interpret = cv2.INTER_LINEAR
fuz_interpret = cv2.INTER_AREA
gt_interpret_05 = cv2.INTER_AREA
gt_interpret_15 = cv2.INTER_AREA
voting = {'aug_rawgt_scale_0.5(0.5_ar)': 3, 'aug_rawgt': 5, 'aug_rawgt_scale_1.5(1.5_ar)': 7}
pos_num_15 = 1.5
pos_num_05 = 0.5
muhu = 2

def make_lst():
    title = '(beta)'
    gt_filenames = []
    image_filenames = []
    for dir in os.listdir(arg_dir.replace('train/', 'train')):
        if ('aug_gt' not in dir) or (title not in dir):
            continue
        for sondir in os.listdir(arg_dir + '/' + dir):
            for filename in os.listdir(arg_dir + '/' + dir + '/' + sondir):
                if '_N1' in filename:
                    continue
                gt_filenames.append('train/' + dir + '/' + sondir + '/' + filename)
                image_filenames.append(
                    'train/' + dir.replace('gt', 'data').replace(title, '') + '/' + sondir + '/' + filename.split('_')[
                        0] + '.png')

    filenames_len = len(image_filenames)

    if filenames_len:
        dft = pd.DataFrame(np.arange(filenames_len * 2).reshape(filenames_len, 2), columns=['Addr', 'Addg'])
        dft.Addr = image_filenames
        dft.Addg = gt_filenames
    dft.to_csv(arg_dir.replace('train/', '') + 'train_pair' + title + '.lst', columns=['Addr', 'Addg'], index=False,
               header=False)


def multi():
    # this one!
    def vote_method40_fast(gts, gts_expande, pixel_tole, total):
        #gts: 原始gt， gts_expande: 膨胀后gt
        def gassu(pixel_tole):
            kernel_size = pixel_tole * 2 + 1
            mask = np.zeros([kernel_size, kernel_size])
            mask[pixel_tole, pixel_tole] = 1
            mask_kernel = np.array([0, 1, 0, 1, 1, 1, 0, 1, 0]).reshape([3, 3]).astype(np.uint8)
            mask = cv2.dilate(mask, mask_kernel, iterations=pixel_tole)

            kernel_size_e = pixel_tole * 4 + 1
            mask_e = np.zeros([kernel_size_e, kernel_size_e])
            mask_e[pixel_tole * 2, pixel_tole * 2] = 1
            mask_kernel_e = np.array([0, 1, 0, 1, 1, 1, 0, 1, 0]).reshape([3, 3]).astype(np.uint8)
            mask_e = cv2.dilate(mask, mask_kernel_e, iterations=pixel_tole * 2)

            sigma = 1
            X = np.linspace(-sigma, sigma, kernel_size)
            Y = np.linspace(-sigma, sigma, kernel_size)
            x, y = np.meshgrid(X, Y)
            gauss_1 = 1 / (2 * np.pi * sigma ** 2) * np.exp(- (x ** 2 + y ** 2) / (2 * sigma ** 2))
            sigma0 = 0.2
            #sigma0 = 1
            gauss_0 = 1 / (2 * np.pi * sigma0 ** 2) * np.exp(- (x ** 2 + y ** 2) / (2 * sigma0 ** 2))

            gauss_1[mask == 0] = 0
            gauss_0[mask == 0] = 0
            return gauss_0, gauss_1, mask, mask_e

        nump = 0
        for dsa in gts:
            nump += np.sum(dsa)

        h, w = gts[0].shape
        h = h + pixel_tole * 2
        w = w + pixel_tole * 2
        gts_final = np.zeros([h, w])
        gts = list(map(lambda x: cv2.copyMakeBorder(x, pixel_tole, pixel_tole, pixel_tole, pixel_tole, cv2.BORDER_CONSTANT, value=(0)), gts))
        gts_expande = list(map(lambda x: cv2.copyMakeBorder(x, pixel_tole, pixel_tole, pixel_tole, pixel_tole, cv2.BORDER_CONSTANT, value=(0)), gts_expande))
        gts_expande_sum = reduce(lambda x, y: x + y, gts_expande)
        gaussfi_tight, gassufi, mask, mask_e = gassu(pixel_tole)
        gts_inter_sum = list(map(lambda x: np.around(cv2.filter2D(x, -1, kernel=mask)).astype(np.float), gts))

        for now_vote in range(total, 0, -1):
            now_vote_map = np.zeros(gts_expande_sum.shape)
            now_vote_map[np.where(gts_expande_sum == now_vote)] = 1
            if np.sum(now_vote_map) == 0:
                continue
            now_vote_map_cross = np.around(cv2.filter2D(now_vote_map, -1, kernel=mask_e)).astype(np.float)
            indenp_locate = np.where((now_vote_map_cross == 1)&(now_vote_map == 1))
            for indep_idx in range(len(indenp_locate[0])):
                h_, w_ = indenp_locate[0][indep_idx], indenp_locate[1][indep_idx]
                gts_final[h_, w_] = gts_expande_sum[h_, w_]
                #idx_tmp = np.zeros([h, w])
                #idx_tmp[h_ - pixel_tole: h_ + pixel_tole + 1, w_ - pixel_tole: w_ + pixel_tole + 1] = gassufi
                for gts_idx, gts_each in enumerate(gts):
                    #if np.max(gts_each + idx_tmp) <= 1:
                    if np.max(gts_each[h_ - pixel_tole: h_ + pixel_tole + 1, w_ - pixel_tole: w_ + pixel_tole + 1] + gassufi) <= 1:
                        continue
                    #idx_each = np.argmax(gts_each + idx_tmp)
                    idx_each = np.argmax(gts_each[h_ - pixel_tole: h_ + pixel_tole + 1, w_ - pixel_tole: w_ + pixel_tole + 1] + gassufi)
                    #h_i, w_i = math.floor(idx_each / w), idx_each % w
                    h_i, w_i = math.floor(idx_each / (2 * pixel_tole + 1)) + h_ - pixel_tole, idx_each % (2 * pixel_tole + 1) + w_ - pixel_tole
                    gts_each[h_i, w_i] = 0
                    gts_inter_sum_tmp = gts_inter_sum[gts_idx][h_i - pixel_tole: h_i + pixel_tole + 1,
                                        w_i - pixel_tole: w_i + pixel_tole + 1].copy()
                    gts_inter_sum_new = gts_inter_sum[gts_idx][h_i - pixel_tole: h_i + pixel_tole + 1,
                                        w_i - pixel_tole: w_i + pixel_tole + 1]
                    gts_inter_sum_new -= mask
                    changes = np.where((gts_inter_sum_tmp != 0) & (gts_inter_sum_new == 0))
                    if len(changes[0]) > 0:
                        gts_expande_sum_tmp = gts_expande_sum[h_i - pixel_tole: h_i + pixel_tole + 1,
                                              w_i - pixel_tole: w_i + pixel_tole + 1]
                        gts_expande_sum_tmp[changes] -= 1
                gts_expande_sum[h_, w_] = 0
                now_vote_map[h_, w_] = 0

                #sddd += now_vote

            gts_inter_sum_dist = list(map(lambda x: cv2.filter2D(x, -1, kernel=gaussfi_tight), gts))
            gts_inter_sum_dist = reduce(lambda x, y: x + y, gts_inter_sum_dist)
            corss_locate = np.where(now_vote_map == 1)
            dist_advance = np.argsort(-gts_inter_sum_dist[corss_locate])
            for cross_idx in range(len(corss_locate[0])):
                h_, w_ = corss_locate[0][dist_advance[cross_idx]], corss_locate[1][dist_advance[cross_idx]]
                if now_vote_map[h_, w_] != 1:
                    continue
                gts_final[h_, w_] = gts_expande_sum[h_, w_]

                for gts_idx, gts_each in enumerate(gts):
                    if np.max(gts_each[h_ - pixel_tole: h_ + pixel_tole + 1, w_ - pixel_tole: w_ + pixel_tole + 1] + gassufi) <= 1:
                        continue
                    idx_each = np.argmax(gts_each[h_ - pixel_tole: h_ + pixel_tole + 1, w_ - pixel_tole: w_ + pixel_tole + 1] + gassufi)
                    h_i, w_i = math.floor(idx_each / (2 * pixel_tole + 1)) + h_ - pixel_tole, idx_each % (2 * pixel_tole + 1) + w_ - pixel_tole
                    gts_each[h_i, w_i] = 0
                    gts_inter_sum_tmp = gts_inter_sum[gts_idx][h_i - pixel_tole: h_i + pixel_tole + 1,
                                        w_i - pixel_tole: w_i + pixel_tole + 1].copy()
                    gts_inter_sum_new = gts_inter_sum[gts_idx][h_i - pixel_tole: h_i + pixel_tole + 1,
                                        w_i - pixel_tole: w_i + pixel_tole + 1]
                    gts_inter_sum_new -= mask
                    changes = np.where((gts_inter_sum_tmp != 0) & (gts_inter_sum_new == 0))
                    if len(changes[0]) > 0:
                        gts_expande_sum_tmp = gts_expande_sum[h_i - pixel_tole: h_i + pixel_tole + 1, w_i - pixel_tole: w_i + pixel_tole + 1]
                        gts_expande_sum_tmp[changes] -= 1
                        now_vote_map_tmp = now_vote_map[h_i - pixel_tole: h_i + pixel_tole + 1, w_i - pixel_tole: w_i + pixel_tole + 1]
                        now_vote_map_tmp[changes] -= 1
                gts_expande_sum[h_, w_] = 0
                now_vote_map[h_, w_] = 0

        return gts_final[pixel_tole: h - pixel_tole, pixel_tole: w - pixel_tole]

    def merge_aug_gt():
        gt_files_tasks = []
        skey_tasks = []
        son_dir_tasks = []
        scales_tasks = []
        scales = {'aug_rawgt_scale_0.5(0.5_ar)': 'aug_gt_scale_0.5(beta)',
                  'aug_rawgt(1_ar)': 'aug_gt(beta)',
                  'aug_rawgt_scale_1.5(1.5_ar)': 'aug_gt_scale_1.5(beta)',
                  }
        for skey in scales.keys():
            print(skey)
            if not os.path.exists(arg_dir + scales[skey]):
                os.makedirs(arg_dir + scales[skey])
            for son_dir in os.listdir(arg_dir + skey):
                print(son_dir)
                if not os.path.exists(arg_dir + scales[skey] + '/' + son_dir):
                    os.makedirs(arg_dir + scales[skey] + '/' + son_dir)
                gt_files = os.listdir(arg_dir + skey + '/' + son_dir)
                gt_files_tasks.append(gt_files)
                skey_tasks.append(skey)
                son_dir_tasks.append(son_dir)
                scales_tasks.append(scales)
        return gt_files_tasks, skey_tasks, son_dir_tasks, scales_tasks

    def do_task(args):
        def make_fuz(lb, lb_file, total_num):
            muhu_n = muhu
            if 'scale_1.5' in lb_file:
                muhu_n = round(muhu_n * 1.5)
            elif 'scale_0.5' in lb_file:
                muhu_n = round(muhu_n * 0.5)

            lb_fuz = lb.copy()
            lb_fuz[lb_fuz == total_num + 1] = 0
            lb_fuz[lb_fuz > 0] = 1
            kernel = np.array([0, 1, 0, 1, 1, 1, 0, 1, 0]).reshape([3, 3]).astype(np.uint8)
            lb_fuz = cv2.dilate(lb_fuz, kernel, iterations=muhu_n)
            lb[(lb == 0) & (lb_fuz > 0)] = total_num + 1
            return lb

        gt_files, skey, son_dir, scales = args
        files_num = dict()
        for file in gt_files:
            each_name = file.split('_')[0]
            if each_name not in files_num.keys():
                files_num[each_name] = 1
            else:
                files_num[each_name] += 1
        for file in files_num.keys():
            file_num = files_num[file]
            # file_num = 1
            if os.path.exists(arg_dir + scales[skey] + '/' + son_dir + '/' + file + '_N' + str(file_num) + '.png') \
                    and os.path.exists(arg_dir + scales[skey] + '/' + son_dir + '/' + file + '_N1' + str(file_num) + '.png'):
                continue
            gts_ = list(
                map(lambda x: cv2.imread(arg_dir + skey + '/' + son_dir + '/' + file + '_' + str(x) + '.png', 0),
                    range(file_num)))
            print(arg_dir + skey + '/' + son_dir + '/' + file + '_' + str(0) + '.png')
            gts = []
            gts_fuz = np.zeros(gts_[0].shape)
            for gt_ in gts_:
                gts_fuz[np.where(gt_ == 128)] = file_num + 1
                gt_[np.where(gt_ == 128)] = 0
                gts.append(gt_ / 255)
            for gt_each in gts:
                gts_fuz[np.where(gt_each == 1)] = file_num + 1
            pixel_tole = int((voting[skey] - 1) / 2)
            if pixel_tole == 0:
                gts = reduce(lambda x, y: x + y, list(gts))
            else:
                kernel = np.array([0, 1, 0, 1, 1, 1, 0, 1, 0]).reshape([3, 3]).astype(np.uint8)
                gts_expande = list(map(lambda x: cv2.dilate(x, kernel, iterations=pixel_tole), gts))
                gts = vote_method40_fast(gts, gts_expande, pixel_tole, file_num)
            # count_density(gt_raw = gts_raw, gt_out = gts, total = file_num)
            gts_raw = gts.copy()
            gts[np.where((gts == 0) & (gts_fuz == file_num + 1))] = file_num + 1
            gts = make_fuz(gts, scales[skey], file_num)
            cv2.imwrite(arg_dir + scales[skey] + '/' + son_dir + '/' + file + '_N' + str(file_num) + '.png', gts)
            # plt.figure()
            plt.imsave(arg_dir + scales[skey] + '/' + son_dir + '/' + file + '_N1' + str(file_num) + '.png', gts_raw)

    gt_files_tasks, skey_tasks, son_dir_tasks, scales_tasks = merge_aug_gt()
    args = zip(gt_files_tasks, skey_tasks, son_dir_tasks, scales_tasks)
    p = mp.Pool(4)
    p.map(do_task, args)
    p.close()
    p.join()

def make_aug_images():

    def auto_thrshold(trans_image, num_pos):

        resolute = 10
        threshold_list = range(resolute, 255, resolute)
        num_hold = np.array(list(map(lambda x:len(np.where(trans_image>x)[0]), threshold_list)))

        num_diff = num_hold - num_pos
        num_diff[np.where(num_diff>0)] = -1e6
        real_threshold = threshold_list[np.argmax(num_diff)]
        trans_image[np.where(trans_image>real_threshold)] = 255
        trans_image[np.where(trans_image == 0)] = 254
        trans_image[np.where(trans_image <= real_threshold)] = 128
        trans_image[np.where(trans_image == 254)] = 0

        return trans_image

    def centercut(image, size):
        h = image.shape[0]
        w = image.shape[1]
        h_init = int((h - size[1]) / 2)
        w_init = int((w - size[0]) / 2)
        return image[h_init:h_init+size[1], w_init:w_init+size[0]]

    def rotate_hw(result, w = 384, h = 544, degree = 0, scale = 1):
        raw_size = [22.5, 202.5, 157.5, 337.5, 15.0, 165.0, 195.0, 345.0]
        rev_size = [67.5, 112.5, 247.5, 292.5, 75.0, 105.0, 255.0, 285.0]
        if degree % 180 == 0:
            return centercut(result, (w, h))
        elif degree % 90 == 0:
            return centercut(result, (h, w))
        elif degree % 45 == 0:
            return centercut(result, (math.ceil(scale * 227), math.ceil(scale * 227)))
        elif degree % 22.5 == 0:
            if degree in raw_size:
                if w > h:
                    return centercut(result, (math.ceil(scale * 454), math.ceil(scale * 159)))
                else:
                    return centercut(result, (math.ceil(scale * 159), math.ceil(scale * 454)))
            elif degree in rev_size:
                if h > w:
                    return centercut(result, (math.ceil(scale * 454), math.ceil(scale * 159)))
                else:
                    return centercut(result, (math.ceil(scale * 159), math.ceil(scale * 454)))
        elif degree % 15.0 == 0:
            if degree in raw_size:
                if w > h:
                    return centercut(result, (math.ceil(scale * 440), math.ceil(scale * 214)))
                else:
                    return centercut(result, (math.ceil(scale * 214), math.ceil(scale * 440)))
            elif degree in rev_size:
                if h > w:
                    return centercut(result, (math.ceil(scale * 440), math.ceil(scale * 214)))
                else:
                    return centercut(result, (math.ceil(scale * 214), math.ceil(scale * 440)))

    def rotate_image(img, degree = 0, scale = 1.0, fordata = True, num_pos = 0):

        h = img.shape[0]
        w = img.shape[1]
        # 第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
        M = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 1)
        # 自适应图片边框大小
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        M[0, 2] += (new_w - w) * 0.5
        M[1, 2] += (new_h - h) * 0.5
        new_w = int(np.round(new_w))
        new_h = int(np.round(new_h))
        # 执行旋转, 任意角度旋转
        if not fordata:
            result = cv2.warpAffine(img, M, (new_w, new_h), flags = cv2.INTER_LINEAR)
            #result_s = result[:,:,0]
            if scale == 1.5:
                #result = auto_thrshold(result, num_pos * pos_num_15 * pos_num_15)
                result = auto_thrshold(result, num_pos * pos_num_15)
            elif scale == 1:
                result = auto_thrshold(result, num_pos * scale * scale)
            else:
                result = auto_thrshold(result, num_pos * pos_num_05)
                #result = auto_thrshold(result, num_pos * scale)
        else:
            result = cv2.warpAffine(img, M, (new_w, new_h))
        # crop central
        result = rotate_hw(result, w, h, degree, scale)

        return result

    def build_aug(source_dir, fordata = True):
        scales = {'aug_data': 1, 'aug_data_scale_0.5': 0.5, 'aug_data_scale_1.5': 1.5}
        if not fordata:
            scales = {'aug_rawgt(1_ar)': 1, 'aug_rawgt_scale_0.5(0.5_ar)': 0.5, 'aug_rawgt_scale_1.5(1.5_ar)': 1.5}

        degrees = {'0.0_': 0.0, '22.5_': 22.5, '15.0_': 15.0, '45.0_': 45.0, '75.0_': 75.0, '67.5_': 67.5,
                   '90.0_': 90.0, '112.5_': 112.5, '105.0_': 105.0, '135.0_': 135.0, '165.0_': 165.0, '157.5_': 157.5,
                   '180.0_': 180.0, '202.5_': 202.5, '195.0_': 195.0, '225.0_': 225.0, '255.0_': 255.0, '247.5_': 247.5,
                   '270.0_': 270.0, '292.5_': 292.5, '285.0_': 285.0, '315.0_': 315.0, '345.0_': 345.0, '337.5_': 337.5}
        flips = {'1_0': 0, '1_1': 1}
        now_procession = 0
        total_procession = len(os.listdir(source_dir))
        for file in os.listdir(source_dir):
            print('[%(tn)d]/[%(nn)d]:'%{'tn':total_procession, 'nn':now_procession} + file)
            now_procession += 1
            for skey in scales.keys():
                if not os.path.exists(arg_dir + skey):
                    os.makedirs(arg_dir + skey)
                dir_scale = arg_dir + skey + '/'
                img = cv2.imread(source_dir + '/' + file)
                if scales[skey] == 1:
                    scale_img = img.copy()
                    num_pos = len(np.where(scale_img == 255)[0])
                else:
                    if not fordata:
                        if scales[skey] == 0.5:
                            gt_interpret = gt_interpret_05
                        elif scales[skey] == 1.5:
                            gt_interpret = gt_interpret_15
                        scale_img = cv2.resize(img.copy(),
                                                (math.ceil(img.shape[1] * scales[skey]),
                                                math.ceil(img.shape[0] * scales[skey])),
                                                interpolation=gt_interpret) # cv2.INTER_CUBIC
                    else:
                        scale_img = cv2.resize(img.copy(),
                                               (math.ceil(img.shape[1] * scales[skey]),
                                                math.ceil(img.shape[0] * scales[skey])),
                                               interpolation=img_interpret)
                for fkey in flips.keys():
                    if fkey == '1_1':
                        scale_flip_img = cv2.flip(scale_img.copy(), 1)
                    else:
                        scale_flip_img = scale_img.copy()
                    for dkey in degrees.keys():
                        if not os.path.exists(dir_scale + dkey + fkey):
                            os.makedirs(dir_scale + dkey + fkey)
                        dir_scale_degree = dir_scale + dkey + fkey + '/'
                        if os.path.exists(dir_scale_degree + file.replace('.jpg', '.png')):
                            sdd = cv2.imread(dir_scale_degree + file.replace('.jpg', '.png'))
                            sdd[np.where(sdd == 0)] = 0
                            continue
                        scale_flip_rot_img = rotate_image(scale_flip_img.copy(), degrees[dkey], scales[skey], fordata, num_pos)
                        cv2.imwrite(dir_scale_degree + file.replace('.jpg','.png'), scale_flip_rot_img, )


    build_aug(root_dir_images) # make augment for images
    #build_aug(root_dir_gts, fordata = False)  # make augment for annotations

if __name__ == '__main__':
    make_aug_images() # build aug
    #multi() # merge annotations
    #make_lst()