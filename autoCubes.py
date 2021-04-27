# -*- coding: utf-8 -*-
from PIL import Image
import subprocess
import time
import re
import glob
import os
import math
from sys import argv
from sys import exit

import cv2
import imutils
import numpy as np
# import rawpy
from pyimagesearch.shapedetector import ShapeDetector

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
camera_folder = '/storage/self/primary/DCIM/Camera'
checker = 'spydercheckr24'
if '--xrite' in argv:
    checker = 'x-rite'
x_multipliers = [0.157, 0.366, 0.572, 0.786]
y_multipliers = [0.114, 0.267, 0.4275, 0.583, 0.735, 0.884]

pipetka = 30

spydercheckr24_colors = [[249, 242, 238], [202, 198, 195], [161, 157, 154], [122, 118, 116], [80, 80, 78], [43, 41, 43],
                         [0, 127, 159], [192, 75, 145], [245, 205, 0], [186, 26, 51], [57, 146, 64], [25, 55, 135],
                         [222, 118, 32], [58, 88, 159], [195, 79, 95], [83, 58, 106], [157, 188, 54], [238, 158, 25],
                         [98, 187, 166], [126, 125, 174], [82, 106, 60], [87, 120, 155], [197, 145, 125], [112, 76, 60]]

xrite_colors = [[115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67], [133, 128, 177], [103, 189, 170],
                [214, 126, 44], [80, 91, 166], [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
                [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31], [187, 86, 149], [8, 133, 161],
                [243, 243, 242], [200, 200, 200], [160, 160, 160], [122, 122, 121], [85, 85, 85], [52, 52, 52]]

xrite_remodeled = [[243, 243, 242], [200, 200, 200], [160, 160, 160], [122, 122, 121], [85, 85, 85], [52, 52, 52],
                   [8, 133, 161], [187, 86, 149], [231, 199, 31], [175, 54, 60], [70, 148, 73], [56, 61, 150],
                   [214, 126, 44], [80, 91, 166], [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
                   [112, 76, 60], [197, 145, 125], [87, 120, 155], [82, 106, 60], [126, 125, 174], [98, 187, 166]]


def adb_command(command):
    try:
        response = subprocess.check_output(os.path.join(__location__, "adb/adb") + " " + command, timeout=5)
    except subprocess.CalledProcessError:
        print('Произошла ошибка adb. Проверьте, подключен ли телефон.')
        exit()
        response = 'Ошибка!'
    except subprocess.TimeoutExpired:
        print('Процесс adb превысил время ожидания и был закрыт. Попробуем еще раз')
        response = adb_command(command)
    return response


def unwarp(img, src, dst):
    h, w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped


def start_photoncamera():
    adb_command("shell am start -n com.particlesdevs.photoncamera/com.particlesdevs.photoncamera.ui.SplashActivity")
    time.sleep(3)
    # print(res)
    return


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def get_screen_size():
    res = adb_command("shell wm size")
    size = re.findall(r"\d{3,4}", str(res))
    return size


def tap_shutter():
    # scr_size = get_screen_size()
    # adb_command(f"shell input tap {int(int(scr_size[0])/2)} {int(int(scr_size[1])*0.8)}")
    adb_command('shell input keyevent 25')


def pull_last_photo(filename):
    adb_command(f'pull {filename} last_photo.jpg')


def get_last_modified_file(folder, local=False):
    if local:
        list_of_files = glob.glob(folder)  # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        return folder + "/" + latest_file
    res = str(adb_command(f'shell "ls -lt {folder} | head"'))
    res2 = [r for r in res.split(r'\r\n')]
    res3 = res2[1].split()[-1]
    return folder + "/" + res3


def wait_for_new_photo(folder, local=False):
    print("Жду появления новой фотографии...")
    was = get_last_modified_file(folder, local)
    for i in range(20):
        now = get_last_modified_file(folder, local)
        if now != was:
            print("Новое фото найдено.")
            if '--gcam' in argv:
                print('Жду окончания обработки фото...')
                for j in range(10):
                    was = now
                    now = get_last_modified_file(folder, local)
                    if now != was:
                        return now
                    time.sleep(2)

            return now
        else:
            was = now
            time.sleep(3)
    print("Не могу найти новое фото...")
    exit()


def get_average_color(img, point, pixel_square=pipetka):
    r_sum = 0
    g_sum = 0
    b_sum = 0
    for i in range(pixel_square):
        for j in range(pixel_square):
            pixel = img.getpixel((point[0] + i, point[1] + j))
            r_sum += pixel[0]
            g_sum += pixel[1]
            b_sum += pixel[2]
    r_avg = int(r_sum / pixel_square ** 2)
    g_avg = int(g_sum / pixel_square ** 2)
    b_avg = int(b_sum / pixel_square ** 2)
    return r_avg, g_avg, b_avg


def opencv_show_image(points, img):
    if img.shape[0] < img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    for p in points:
        cv2.rectangle(img, p, (p[0]+pipetka, p[1]+pipetka), (0, 0, 0), 3)
    resized = cv2.resize(img, (600, 800))
    if '--showpoints' in argv:
        cv2.imshow('points', resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    pass


def get_photo_colors(img, points_list):
    average_colors = [get_average_color(img, point) for point in points_list]
    return average_colors


def show_result():
    with open(os.path.join(__location__, 'customCCT_autoCubes.txt')) as fp:
        print('Результат калибровки:\r\n', fp.read())


def opencv_find_etalon(image_filename):
    # img = cv2.imread(os.path.join(__location__, image_filename).replace('\\', '/'))
    try:
        img = Image.open(os.path.join(__location__, image_filename))
    except FileNotFoundError:
        print('Не могу найти файл фото!')
        exit()
        return
    pil_image = img.convert('RGB')
    open_cv_image = np.array(pil_image)
    img = open_cv_image[:, :, ::-1].copy()
    # print(img.shape)
    if img.shape[0] < img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    resized = cv2.resize(img, (600, 800))
    # ratio = img.shape[0] / float(resized.shape[0])
    blurred = cv2.GaussianBlur(resized, (9, 9), 0)
    img_transf = cv2.cvtColor(blurred, cv2.COLOR_BGR2YUV)
    # img_transf[:, :, 0] = cv2.equalizeHist(img_transf[:, :, 0])
    img4 = cv2.cvtColor(img_transf, cv2.COLOR_YUV2BGR)
    gray = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 80, 200, cv2.THRESH_BINARY)[1]
    if '--showpoints' in argv:
        cv2.imshow('thresh', thresh)
        cv2.waitKey(0)

    # Lines
    edges = cv2.Canny(gray, 100, 150, apertureSize=3)
    if '--showpoints' in argv:
        cv2.imshow('edges', edges)
    lines = cv2.HoughLines(edges, 1, np.pi / 75, 60)

    def angle(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dX = x2 - x1
        dY = y2 - y1
        rads = math.atan2(-dY, dX)  # wrong for finding angle/declination?
        return math.degrees(rads)

    min_x = 600
    min_y = 800
    max_x = 0
    max_y = 0

    angles = []
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # x1 stores the rounded off value of (r * cos(theta) - 1000 * sin(theta))
        x1 = int(x0 + 800 * -b)
        # y1 stores the rounded off value of (r * sin(theta)+ 1000 * cos(theta))
        y1 = int(y0 + 800 * a)
        # x2 stores the rounded off value of (r * cos(theta)+ 1000 * sin(theta))
        x2 = int(x0 - 800 * -b)
        # y2 stores the rounded off value of (r * sin(theta)- 1000 * cos(theta))
        y2 = int(y0 - 800 * a)

        angles.append(angle((x1, y1), (x2, y2)))
        if 45 < abs(angle((x1, y1), (x2, y2))) < 135:
            cv2.line(resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
    if '--showpoints' in argv:
        cv2.imshow('Lines', resized)
        cv2.waitKey(0)
    angles = np.array(angles)

    vertical = np.absolute(angles[np.argwhere(np.bitwise_and(np.absolute(angles) > 45, np.absolute(angles) < 135))])
    horisontal = np.absolute(angles[np.argwhere(np.bitwise_or(np.absolute(angles) < 45, np.absolute(angles) > 135))])

    if '--debug' in argv:
        print('Vertical: \r\n', vertical.sum() / len(vertical), '\r\nHorisontal: \r\n', horisontal.sum() / len(horisontal))
        print('Degrees to rotate: ', round((90 - vertical.sum() / len(vertical) + horisontal.sum() / len(horisontal)) / 2))
    # find contours in the thresholded image and initialize the
    # shape detector
    degrees = round((90 - vertical.sum() / len(vertical) + horisontal.sum() / len(horisontal)) / 2)
    if abs(degrees) > 2:
        rotated = rotate_bound(img,  degrees)
    else:
        rotated = img
    resized = cv2.resize(rotated, (600, 800))
    x_ratio = rotated.shape[1] / float(resized.shape[1])
    y_ratio = rotated.shape[0] / float(resized.shape[0])
    blurred = cv2.GaussianBlur(resized, (9, 9), 0)
    '''img_transf = cv2.cvtColor(blurred, cv2.COLOR_BGR2YUV)
    # img_transf[:, :, 0] = cv2.equalizeHist(img_transf[:, :, 0])
    img4 = cv2.cvtColor(img_transf, cv2.COLOR_YUV2BGR)'''
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 150, apertureSize=3)
    thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

    if '--showpoints' in argv:
        cv2.imshow('thresh', thresh)
        cv2.waitKey(0)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    thresh2 = cv2.threshold(gray, 100, 200, cv2.THRESH_BINARY)[1]

    if '--showpoints' in argv:
        cv2.imshow('thresh2', thresh2)
        cv2.waitKey(0)

    cnts2 = cv2.findContours(thresh2.copy(), cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)

    thresh3 = cv2.threshold(gray, 0, 100, cv2.THRESH_BINARY)[1]

    if '--showpoints' in argv:
        cv2.imshow('thresh3', thresh3)
        cv2.waitKey(0)

    cnts3 = cv2.findContours(thresh3.copy(), cv2.RETR_TREE,
                             cv2.CHAIN_APPROX_SIMPLE)

    cnts4 = cv2.findContours(edges.copy(), cv2.RETR_TREE,
                             cv2.CHAIN_APPROX_SIMPLE)

    cnts += imutils.grab_contours(cnts2)
    cnts += imutils.grab_contours(cnts3)
    cnts += imutils.grab_contours(cnts4)

    # print(len(cnts))
    sd = ShapeDetector()

    # print(cnts)
    for c in cnts:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        epsilon = 0.1 * cv2.arcLength(c, True)
        c = cv2.approxPolyDP(c, epsilon, True)
        M = cv2.moments(c)
        cX = int((M["m10"] / (M["m00"] + 0.0000001)))
        cY = int((M["m01"] / (M["m00"] + 0.0000001)))
        shape = sd.detect(c)
        x = ''
        y = ''
        w = ''
        h = ''
        if shape == 'square' or shape == 'rectangle':
            x, y, w, h = cv2.boundingRect(c)
            if 60 < w < 150 and 60 < h < 150:
                if x < min_x:
                    min_x = x
                if x+w > max_x:
                    max_x = x+w
                if y < min_y:
                    min_y = y
                if y+h > max_y:
                    max_y = y+h

        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        # c = c.astype("float")
        # c *= ratio
                c = c.astype("int")

                cv2.drawContours(resized, [c], -1, (0, 255, 0), 2)

        # show the output image
    max_min = tuple((np.array([min_x*x_ratio, min_y*y_ratio, max_x*x_ratio, max_y*y_ratio])).astype('int'))
    cv2.rectangle(resized, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)
    if '--showpoints' in argv:
        cv2.imshow("Image", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if '--debug' in argv:
        print("Cropped area: ", max_min)
        print('Rotated shape', rotated.shape)
    return rotated[max_min[1]:max_min[3], max_min[0]:max_min[2]]


def get_colors_from_test_photo(image='', photo_file='last_photo.jpg'):
    global im
    custom_image_flag = False
    custom_image_filename = ''

    for arg in argv:
        found_jpeg = re.search(r'[a-zA-Z0-9\-_]*\.jpg', arg)
        found_raw = re.search(r'[a-zA-Z0-9\-_]*\.dng', arg)
        if isinstance(found_jpeg, re.Match):
            custom_image_filename = found_jpeg.group()
            custom_image_flag = True
            break
        if isinstance(found_raw, re.Match):
            custom_image_filename = found_raw.group()
            custom_image_flag = True
            break

    if custom_image_flag:
        photo_file = custom_image_filename
    # print(os.path.join(__location__, photo_file))
    for i in range(5):
        try:
            if not image:
                im_cv2 = opencv_find_etalon(photo_file)
                print(im_cv2.shape)
                img = cv2.cvtColor(im_cv2, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(img)
                '''crop = opencv_find_etalon(photo_file)
                im = im.crop(crop)
                if im.width > im.height:
                    im = im.rotate(270, expand=True)'''
                '''pil_image = im.convert('RGB')
                open_cv_image = np.array(pil_image)
                open_cv_image = open_cv_image[:, :, ::-1].copy()'''
                '''for x in x_multipliers:
                    for y in y_multipliers:
                        curr_point = (int(x * im.width), int(y * im.height))
                        points.append(curr_point)'''
                points = []
                for j in range(1, 5):
                    for k in range(1, 7):
                        points.append((int(j * im.width/4 - im.width/8), int(k * im.height/6 - im.height/12)))
                # print(im.size, points)

                if '--showpoints' in argv:
                    opencv_show_image(points, im_cv2)
                    # pg_show_image(points, photo_file)
                photo_colors = get_photo_colors(im, points)
                return photo_colors
            else:
                points = []
                im = image
                for x in x_multipliers:
                    for y in y_multipliers:
                        curr_point = (int(x * im.width), int(y * im.height))
                        points.append(curr_point)
                photo_colors = get_photo_colors(im, points)
                return photo_colors
        except OSError:
            print('Не удалось прочитать файл изображения, пробую {} раз'.format(i + 2))
            if '--nophone' not in argv:
                pull_last_photo(get_last_modified_file(camera_folder))
            time.sleep(3)
    print('Ничего не вышло :( Попробуйте еще раз')


def apply_matrix(points, m):
    out = []
    for p in points:
        p = np.array(p)
        # print('Point: ', p, '\r\nMatrix: ', m)
        out.append((p[0] * m[0][0] + p[1] * m[0][1] + p[2] * m[0][2],
                    p[0] * m[1][0] + p[1] * m[1][1] + p[2] * m[1][2],
                    p[0] * m[2][0] + p[1] * m[2][1] + p[2] * m[2][2]))
    return np.array(out)


def normalize_matrix(m: np.array):
    v_sum = m.sum() / 3
    for i in range(3):
        m[:, i] += (v_sum - m[:, i].sum()) / 3
    # print(m)
    return m


def find_matrix_changer(etalon: np.array, sample: np.array):
    a = etalon
    b = sample
    m = np.array([(((a[i] - ((a[i].sum() - b[i].sum()) / 3 + b[i])) / a[i].sum()) * np.amax(a[i]) / a[i].sum())
                 for i in range(len(sample))]).transpose()
    return m


def get_etalon_colors():
    with Image.open(os.path.join(__location__, "etalon_crop.jpg")) as im:
        points = []
        for x in x_multipliers:
            for y in y_multipliers:
                curr_point = (int(x * im.width), int(y * im.height))
                points.append(curr_point)
                # print(curr_point)

        etalon_colors = get_photo_colors(im, points)
    return etalon_colors


def combine_matrices(matrices, pix_sum):
    # pixel = np.array(pixel)
    # pix_sum = np.sum(pixel)
    if pix_sum < 256:
        return matrices['shadows']
    elif pix_sum < 512:
        return matrices['midtones']
    else:
        return matrices['lights']
    # return matrices['shadows'] + (matrices['midtones'] - matrices['shadows']) / 382 * pix_sum if pix_sum < 383 else \
    #       matrices['midtones'] + (matrices['lights'] - matrices['midtones']) / 383 * (pix_sum - 383)


def find_matrix_from_points(before, after, c_sum):  # Most fun function
    a = before[0][0]
    b = before[0][1]
    c = before[0][2]
    d = before[1][0]
    e = before[1][1]
    f = before[1][2]

    x = after[0]
    n = after[1] - x

    # The fun begins now
    try:
        mult3 = (x * (d - a - e + b) * (e - b - d + a) - c_sum * (
                    (e - b) * (d - a - e + b) * a + (d - a) * (e - b - d + a) * b) - n * (
                             -(d - a - e + b) * a - (e - b - d + a) * b)) / \
                (-(e - b - f + c) * (d - a - e + b) * a - (d - a - f + c) * (e - b - d + a) * b + (d - a - e + b) * (
                            e - b - d + a) * c)
        mult2 = (c_sum * (d - a) - mult3 * (d - a - f + c) - n) / (d - a - e + b)

        mult1 = (c_sum * (e - b) - mult3 * (e - b - f + c) - n) / (e - b - d + a)
    except RuntimeWarning:
        print('Что-то не то в множителях. Справлюсь, не переживай!)')
        return 0, 0, 0

    return mult1, mult2, mult3


def engage_cct_matrix(cubes, image, temperature='warm'):
    # matrices = [cubes[temperature]['shadows'], cubes[temperature]['midtones'], cubes[temperature]['lights']]
    img_arr = np.asarray(image)
    img_array = np.copy(img_arr)
    # Pre - computing matrices for each possible sum of rgb in pixel
    c_ms = np.array([combine_matrices(cubes[temperature], i) for i in range(766)])
    sums = np.sum(img_arr, axis=2)  # getting r+g+b sum of each pixel of the image
    m_fs = c_ms[sums]  #

    print('Pixel: ', '100x120')
    print('From img_arr: ', img_arr[100, 120])
    print('Sum from sums: ', sums[100, 120])
    # print('Pre-computed matrix for sum:\r\n', c_ms[sums[100, 100]])
    m = m_fs[100, 120]
    print('Matrix from m_fs:\r\n', m)

    img_array[:, :] = np.array([np.sum(m_fs[:, :, 0] * img_arr[:, :], axis=2),
                                np.sum(m_fs[:, :, 1] * img_arr[:, :], axis=2),
                                np.sum(m_fs[:, :, 2] * img_arr[:, :], axis=2)]).transpose(1, 2, 0)
    print('Result of correction: ', img_array[100, 120])

    # print(img_arr[800, 800], img_array[800, 800])
    return Image.fromarray(img_array)


def get_cct_matrix_file_from_phone():
    print(adb_command("pull /storage/self/primary/DCIM/PhotonCamera/customCCT.txt customCCT.txt"))


def parse_cct_matrix_from_file(filename='customCCT.txt'):  # making cubes dict with ndarrays of matrices from file
    try:
        with open(os.path.join(__location__, filename)) as fp:
            file_contents = fp.readlines()

    except FileNotFoundError:
        print('Не могу открыть файл :( То-ли его нет, то-ли доступ запрещен...')
    except PermissionError:
        print('Не могу открыть файл :( То-ли его нет, то-ли доступ запрещен...')
    finally:
        if not file_contents:
            pass
        else:
            label = file_contents[0].strip()
            if label == 'MATRIX':
                file_contents = file_contents[2:]  # Getting rid of 'MATRIX' and empty line
                matrix = []
                for i in range(3):
                    row = file_contents[i].split(',')
                    matrix.append([])
                    for j in range(3):
                        matrix[i].append(float(row[j].strip()))
                matrix = np.array(matrix)
                cubes = {
                    'warm': {
                        'shadows': matrix,
                        'midtones': matrix,
                        'lights': matrix
                    },
                    'cool': {
                        'shadows': matrix,
                        'midtones': matrix,
                        'lights': matrix
                    }
                }
                return cubes
            elif label == 'MATRIXES':
                warm_point = file_contents[2]
                cool_point = file_contents[8]

                matrix1 = file_contents[4:7]  # Getting rid of 'MATRIXES'
                matrix = []
                for i in range(3):
                    row = matrix1[i].split(',')
                    matrix.append([])
                    for j in range(3):
                        matrix[i].append(float(row[j].strip()))
                warm_matrix = np.array(matrix)

                matrix2 = file_contents[10:]
                matrix = []
                for i in range(3):
                    row = matrix2[i].split(',')
                    matrix.append([])
                    for j in range(3):
                        matrix[i].append(float(row[j].strip()))
                cool_matrix = np.array(matrix)

                cubes = {
                    'warm': {
                        'shadows': warm_matrix,
                        'midtones': warm_matrix,
                        'lights': warm_matrix
                    },
                    'cool': {
                        'shadows': cool_matrix,
                        'midtones': cool_matrix,
                        'lights': cool_matrix
                    },
                    'warm_point': warm_point,
                    'cool_point': cool_point
                }
                return cubes
            elif label == 'CUBE':
                shadows_matrix = file_contents[2:5]  # Getting rid of 'CUBE' and empty line
                matrix = []
                for i in range(3):
                    row = shadows_matrix[i].split(',')
                    matrix.append([])
                    for j in range(3):
                        matrix[i].append(float(row[j].strip()))
                shadows_matrix = np.array(matrix)

                midtones_matrix = file_contents[6:9]
                matrix = []
                for i in range(3):
                    row = midtones_matrix[i].split(',')
                    matrix.append([])
                    for j in range(3):
                        matrix[i].append(float(row[j].strip()))
                midtones_matrix = np.array(matrix)

                lights_matrix = file_contents[10:]
                matrix = []
                for i in range(3):
                    row = lights_matrix[i].split(',')
                    matrix.append([])
                    for j in range(3):
                        matrix[i].append(float(row[j].strip()))
                lights_matrix = np.array(matrix)

                cubes = {
                    'warm': {
                        'shadows': shadows_matrix,
                        'midtones': midtones_matrix,
                        'lights': lights_matrix
                    },
                    'cool': {
                        'shadows': shadows_matrix,
                        'midtones': midtones_matrix,
                        'lights': lights_matrix
                    }
                }
                return cubes
            elif label == 'CUBES':
                warm_point = file_contents[2]
                cool_point = file_contents[16]
                # print(*[str(i)+" "+file_contents[i] for i in range(len(file_contents))])
                warm_shadows_matrix = file_contents[4:7]  # Getting rid of 'CUBES'
                matrix = []
                for i in range(3):
                    row = warm_shadows_matrix[i].split(',')
                    matrix.append([])
                    for j in range(3):
                        matrix[i].append(float(row[j].strip()))
                warm_shadows_matrix = np.array(matrix)

                warm_midtones_matrix = file_contents[8:11]
                matrix = []
                for i in range(3):
                    row = warm_midtones_matrix[i].split(',')
                    matrix.append([])
                    for j in range(3):
                        matrix[i].append(float(row[j].strip()))
                warm_midtones_matrix = np.array(matrix)

                warm_lights_matrix = file_contents[12:15]
                matrix = []
                for i in range(3):
                    row = warm_lights_matrix[i].split(',')
                    matrix.append([])
                    for j in range(3):
                        matrix[i].append(float(row[j].strip()))
                warm_lights_matrix = np.array(matrix)

                cool_shadows_matrix = file_contents[18:21]  # Getting rid of 'CUBES'
                matrix = []
                for i in range(3):
                    row = cool_shadows_matrix[i].split(',')
                    matrix.append([])
                    for j in range(3):
                        matrix[i].append(float(row[j].strip()))
                cool_shadows_matrix = np.array(matrix)

                cool_midtones_matrix = file_contents[22:25]
                matrix = []
                for i in range(3):
                    row = cool_midtones_matrix[i].split(',')
                    matrix.append([])
                    for j in range(3):
                        matrix[i].append(float(row[j].strip()))
                cool_midtones_matrix = np.array(matrix)

                cool_lights_matrix = file_contents[26:]
                matrix = []
                for i in range(3):
                    row = cool_lights_matrix[i].split(',')
                    matrix.append([])
                    for j in range(3):
                        matrix[i].append(float(row[j].strip()))
                cool_lights_matrix = np.array(matrix)

                cubes = {
                    'warm': {
                        'shadows': np.array(warm_shadows_matrix),
                        'midtones': np.array(warm_midtones_matrix),
                        'lights': np.array(warm_lights_matrix)
                    },
                    'cool': {
                        'shadows': np.array(cool_shadows_matrix),
                        'midtones': np.array(cool_midtones_matrix),
                        'lights': np.array(cool_lights_matrix)
                    },
                    'warm_point': warm_point,
                    'cool_point': cool_point
                }
                return cubes


def save_cubes_to_local_file(cubes: dict, custom_format=False, temp='warm', filename='customCCT_autoCubes.txt'):
    def matrix_to_text(matrix):
        text = ""
        for i in range(3):
            for j in range(3):
                text += str(round(matrix[i][j], 4))
                if i < 2 or j < 2:
                    text += ','
            if i < 2:
                text += '\n'
        return text

    if '--cubes' in argv:   # Cubes
        w_p = cubes.get('warm_point')
        if w_p is None:
            w_p = '0.7,1.0,0.4\n'
        c_p = cubes.get('cool_point')
        if c_p is None:
            c_p = '0.5,1.0,0.72\n'
        file_text = f"""CUBES\n\n{
        w_p}\n{
        matrix_to_text(normalize_matrix(cubes['warm']['shadows']))}\n\n{
        matrix_to_text(normalize_matrix(cubes['warm']['midtones']))}\n\n{
        matrix_to_text(normalize_matrix(cubes['warm']['lights']))}\n\n{
        c_p}\n{
        matrix_to_text(normalize_matrix(cubes['cool']['shadows']))}\n\n{
        matrix_to_text(normalize_matrix(cubes['cool']['midtones']))}\n\n{
        matrix_to_text(normalize_matrix(cubes['cool']['lights']))}"""
    elif '--matrixes' in argv:   # Matrices
        w_p = cubes.get('warm_point')
        if w_p is None:
            w_p = '0.7,1.0,0.4\n'
        c_p = cubes.get('cool_point')
        if c_p is None:
            c_p = '0.5,1.0,0.72\n'
        file_text = f"""MATRIXES\n\n{
        w_p}\n{
        matrix_to_text(normalize_matrix(cubes['warm']['midtones']))}\n\n{
        c_p}\n{
        matrix_to_text(normalize_matrix(cubes['cool']['midtones']))}\n\n"""
    elif '--cube' in argv:
        file_text = f"""CUBE\n\n{matrix_to_text(normalize_matrix(cubes[temp]['shadows']))}\n\n{
        matrix_to_text(normalize_matrix(cubes[temp]['midtones']))}\n\n{
        matrix_to_text(normalize_matrix(cubes[temp]['lights']))}"""
    else:   # Matrix
        file_text = f"""MATRIX\n\n{matrix_to_text(normalize_matrix(cubes['warm']['midtones']))}"""
    if '--debug' in argv:
        print("В телефон будет записана матрица: \r\n", file_text)

    try:
        with open(os.path.join(__location__, filename), 'w') as fp:
            fp.write(file_text)
    except PermissionError:
        print('Не могу записать в файл...')


def save_cubes_to_phone(cubes, single_cube=False, temp='warm'):
    save_cubes_to_local_file(cubes, single_cube, temp)
    adb_command('push customCCT_autoCubes.txt /storage/self/primary/DCIM/PhotonCamera/customCCT.txt')
    print('Сохранено на телефоне. Проверяйте калибровку.')
    show_result()


def help_message():
    text = """
    Перед началом калибровки убедитесь, что вы снимаете мишень в необходимых условиях баланса белого.
    Если вы будете использовать одну матрицу(gcam), снимайте в условиях нейтрального освещения
    (дневной свет, лампа дневного света)
    Снимайте цветовую мишень так, чтобы белый квадратик мишени был слева. 
    При этом не важно, горизонтально или вертикально.
    Для Gcam добавьте ключ --gcam.
    Для использования с чекером x-rite  добавьте ключ --xrite.
    Если калибруете две матрицы/кубы для теплой или холодной температур, по умолчанию калибруется теплая матрица.
    Для калибровки холодной добавьте ключ --cool
    Для двух матриц добавьте --matrixes
    (правильно matrices, но у некоторых талантливых разрабов проблемы с английским)
    Для куба добавьте --cube
    Для кубов --cubes
    
    Если вы снимаете на PhotonCamera и телефон подключен по adb, 
    то при калибровке автоматически выставится матрица для калибровки.
    Если вы снимаете на Gcam с функцией cct, 
    перед калибровкой отключите настройки цвета и выставьте матрицу следующим образом:
    Rr: 1  Rg: 0  Rb: 0
    Gr: 0  Gg: 1  Gb: 0
    Br: 0  Bg: 0  Bb: 1
    """
    return text


def check_calibration(cubes, temp='warm', np_sample_was=np.array([]), np_etalon_was=np.array([]), backup=dict({})):
    print('Проверим калибровку.\r\n')
    time.sleep(1)
    print('Сделайте снимок цветового эталона.')
    if '--nophone' not in argv:
        pull_last_photo(wait_for_new_photo(camera_folder))
        time.sleep(1)
    col1 = get_colors_from_test_photo()
    if checker == 'x-rite':
        col1 = col1[0:6] + col1[11:5:-1] + col1[12:18] + col1[23:17:-1]
        np_etalon = np.array(xrite_remodeled)
    else:
        np_etalon = np.array(spydercheckr24_colors)
    np_sample = np.array(col1)

    if '--debug' in argv:
        print('Was:\r\n', np_sample_was[9:12])
        print('Computed with matrix correction:\r\n',
              apply_matrix(np_sample_was[9:12], normalize_matrix(cubes[temp]['midtones'])).astype('int'))
        print('Real correction:\r\n', np_sample[9:12])
        print(apply_matrix(np_sample_was[9:12],
                           normalize_matrix(cubes[temp]['midtones'])).astype('int') - np_sample[9:12])
    # print(col1)
    # print(spydercheckr24_colors)
    # shadows = np.argwhere(np_sample.sum(axis=1) < 256)
    cubes_was = parse_cct_matrix_from_file('customCCT_autoCubes.txt')
    # print(shadows)
    temperature = temp

    print('Улучшим калибровку.')
    for i in range(1):  # Colors fixing

        # Initializing expected colors via checker colors comparison
        def init_exp_by_levels():
            exp = [[0, 0, 0] for k in range(24)]
            if checker == 'spydercheckr24':
                # Red
                exp[9][0] = round(
                    ((np_sample[1][0] - np_sample[2][0]) / 2 + np_sample[2][0]) * 1.025)  # Derived from etalon
                gb_sum_exp = np_sample[4][0]
                b_div_g_exp = 1.96
                exp[9][2] = round(b_div_g_exp * gb_sum_exp / (b_div_g_exp + 1))
                exp[9][1] = round(np_sample[5][1] * 0.634)
                # More red
                exp[14][0] = np_sample[1][2]
                exp[14][1] = round((np_sample[5][1] + np_sample[5][2]) / 2)
                exp[14][2] = round(np_sample[4][1] * 1.1875)
                # Even more red
                exp[8][0] = round(np_sample[0][0] * 0.984)
                exp[8][1] = round(np_sample[1][1] * 1.035)
                exp[8][2] = np_sample[5][2] - 43
                if exp[8][2] < 0:
                    exp[8][2] = 0
                exp[7][0] = round(np_sample[1][0] * 0.95)
                exp[7][1] = round(np_sample[4][1] * 0.9375)
                exp[7][2] = round(np_sample[2][2] * 0.941)

                # Green
                exp[10][1] = round(
                    ((np_sample[2][1] - np_sample[3][1]) / 2 + np_sample[3][1]) * 1.06)  # Derived from etalon
                rb_sum_exp = np_sample[3][1]
                b_div_r_exp = 1.123
                exp[10][2] = round(rb_sum_exp * b_div_r_exp / (b_div_r_exp + 1))
                exp[10][0] = round(np_sample[5][0] * 1.326)
                # More green
                exp[16][0] = np_sample[2][1]
                exp[16][2] = round(np_sample[2][0] / 3)
                exp[16][1] = round(0.96 * np_sample[1][2])
                exp[20][0] = round(np_sample[4][0] * 1.025)
                exp[20][1] = round(np_sample[3][1] * 0.899)
                exp[20][2] = round(((np_sample[4][2] - np_sample[5][2]) / 2) + np_sample[5][2])

                # Blue
                exp[11][2] = round(((np_sample[2][2] - np_sample[3][2]) / 2) + np_sample[3][2])
                rg_sum_exp = np_sample[4][2]
                g_div_r_exp = 2.2
                exp[11][0] = round(np_sample[5][0] * 0.581)
                # exp[11][1] = rg_sum_exp - exp[11][0]
                exp[11][1] = round(1.341 * np_sample[5][1])
                # More blue
                exp[6][0] = np_sample[5][0] - 43
                if exp[6][0] < 0:
                    exp[6][0] = 0
                exp[6][1] = round(np_sample[3][1] * 1.076)
                exp[6][2] = round(np_sample[2][2] * 1.032467532)

            elif checker == 'x-rite':
                exp[9][0] = round(
                    ((np_sample[1][0] - np_sample[2][0]) / 2 + np_sample[2][0]) * 0.97222222)
                gb_sum_exp = round(np_sample[3][0] * 0.9344262295081968)
                b_div_g_exp = 1.1111111111111112
                exp[9][2] = round(b_div_g_exp * gb_sum_exp / (b_div_g_exp + 1))
                exp[9][1] = round(np_sample[5][1] * 1.0384615384615385)
                # More red
                exp[14][0] = round(np_sample[1][2] * 0.965)
                exp[14][1] = round(np_sample[4][1] * 1.0588235294117647)
                exp[14][2] = round(np_sample[4][2] * 1.1647058823529413)
                # Even more red
                exp[8][0] = round(np_sample[0][0] * 0.9506172839506173)
                exp[8][1] = round(np_sample[1][1] * 0.995)
                exp[8][2] = round(np_sample[5][2] * 0.5961538461538461)

                exp[7][0] = round(np_sample[1][0] * 0.935)
                exp[7][1] = round(np_sample[4][1] * 1.011764705882353)
                exp[7][2] = round(np_sample[2][2] * 0.93125)

                # Green
                exp[10][1] = round(np_sample[2][1] * 0.925)  # Derived from etalon colors
                rb_sum_exp = round(((np_sample[2][1] - np_sample[3][1]) / 2 + np_sample[3][1]) * 1.0141843971631206)
                b_div_r_exp = 1.042857142857143
                exp[10][2] = round(rb_sum_exp * b_div_r_exp / (b_div_r_exp + 1))
                exp[10][0] = round(np_sample[5][0] * 1.3461538461538463)
                # More green
                exp[16][0] = round(np_sample[2][1] * 0.98125)
                exp[16][2] = round(np_sample[5][2] * 1.2307692307692308)
                exp[16][1] = round(0.94 * np_sample[1][2])

                exp[20][0] = round(np_sample[4][0] * 1.0235294117647058)
                exp[20][1] = round(np_sample[3][1] * 0.9836065573770492)
                exp[20][2] = round(np_sample[2][2] * 0.9836065573770492)

                # Blue
                exp[11][2] = round(np_sample[2][2] * 0.9375)  # Derived from etalon colors
                exp[11][0] = round(np_sample[5][0] * 1.0769230769230769)
                exp[11][1] = round(1.1730769230769231 * np_sample[5][1])
                # More blue
                exp[6][0] = round(np_sample[5][0] / 6.5)
                exp[6][1] = round(np_sample[3][1] * 1.0901639344262295)
                exp[6][2] = round(np_sample[2][2] * 1.00625)

            return exp

        def init_exp_by_etalon():
            exp = (np_etalon - np.sum(np_etalon, axis=1)[:, None] / 3 + np.sum(np_sample, axis=1)[:, None] / 3).astype(
                'int')
            return exp

        exp = init_exp_by_levels()

        # exp = init_exp_by_etalon().tolist()

        def making_matrix(grade, debug=False):
            # Getting rows of the matrix
            matrix_R = []
            matrix_G = []
            matrix_B = []

            s = np_sample_was

            r_sum = np.sum(cubes[temperature][grade][0])
            g_sum = np.sum(cubes[temperature][grade][1])
            b_sum = np.sum(cubes[temperature][grade][2])
            c_sums = (r_sum, g_sum, b_sum)
            matrices = (matrix_R, matrix_G, matrix_B)
            exp_copy = exp.copy()
            # print(exp_copy)
            m = 0
            n = 0
            for o in range(3):
                for m in range(len(exp_copy)):
                    for n in range(m + 1, len(exp_copy)):
                        if exp_copy[m] != [0, 0, 0]:
                            res = find_matrix_from_points([s[m], s[n]], [exp_copy[m][o], exp_copy[n][o]], c_sums[o])
                            if min(res) < -2.5 or max(res) > 2.5:
                                continue
                            else:
                                matrices[o].append(res)

            '''matrix_R.append(find_matrix_from_points([s[8], s[7]], [exp[8][0], exp[7][0]], r_sum))
            matrix_R.append(find_matrix_from_points([s[7], np_sample[5]], [exp[7][0], np_sample[5][0]], r_sum))
            matrix_R.append(find_matrix_from_points([s[7], s[16]], [exp[7][0], exp[16][0]], r_sum))
            # matrix_R.append(find_matrix_from_points([s[8], np_sample[5]], [exp[8][0], np_sample[5][0]], r_sum))
            matrix_R.append(find_matrix_from_points([s[16], s[11]], [exp[16][0], exp[11][0]], r_sum))
            matrix_R.append(find_matrix_from_points([s[11], s[6]], [exp[11][0], exp[6][0]], r_sum))
            #matrix_R.append(find_matrix_from_points([s[8], s[16]], [exp[8][0], exp[16][0]], r_sum))
            # matrix_R.append(find_matrix_from_points([s[6], s[16]], [exp[6][0], exp[16][0]], r_sum))

            matrix_G.append(find_matrix_from_points(s[9:11], [exp[9][1], exp[10][1]], g_sum))
            # matrix_G.append(find_matrix_from_points([s[16], s[10]], [exp[16][1], exp[10][1]], g_sum))
            matrix_G.append(find_matrix_from_points([s[8], s[7]], [exp[8][1], exp[7][1]], g_sum))
            matrix_G.append(find_matrix_from_points([s[9], s[16]], [exp[9][1], exp[16][1]], g_sum))
            matrix_G.append(find_matrix_from_points([s[9], s[7]], [exp[9][1], exp[7][1]], g_sum))
            # matrix_G.append(find_matrix_from_points([s[8], s[6]], [exp[8][1], exp[6][1]], g_sum))
            matrix_G.append(find_matrix_from_points([s[16], s[6]], [exp[16][1], exp[6][1]], g_sum))
            matrix_G.append(find_matrix_from_points([s[10], s[6]], [exp[10][1], exp[6][1]], g_sum))

            matrix_B.append(find_matrix_from_points([s[10], s[9]], [exp[10][2], exp[9][2]], b_sum))
            matrix_B.append(find_matrix_from_points([s[16], s[9]], [exp[16][2], exp[9][2]], b_sum))
            matrix_B.append(find_matrix_from_points([s[16], s[10]], [exp[16][2], exp[10][2]], b_sum))
            matrix_B.append(find_matrix_from_points([s[6], s[10]], [exp[6][2], exp[10][2]], b_sum))
            matrix_B.append(find_matrix_from_points([s[6], s[9]], [exp[6][2], exp[9][2]], b_sum))
            matrix_B.append(find_matrix_from_points([s[6], s[7]], [exp[6][2], exp[7][2]], b_sum))
            matrix_B.append(find_matrix_from_points([s[6], s[16]], [exp[6][2], exp[16][2]], b_sum))
            matrix_B.append(find_matrix_from_points([s[8], s[10]], [exp[8][2], exp[10][2]], b_sum))
            matrix_B.append(find_matrix_from_points([s[8], s[7]], [exp[8][2], exp[7][2]], b_sum))'''

            matrix_R = np.array(matrix_R)
            matrix_G = np.array(matrix_G)
            matrix_B = np.array(matrix_B)

            matrix_R = np.delete(matrix_R, np.argwhere(matrix_R[:, 0] < 0), axis=0)
            matrix_G = np.delete(matrix_G, np.argwhere(matrix_G[:, 1] < 0), axis=0)
            matrix_B = np.delete(matrix_B, np.argwhere(matrix_B[:, 2] < 0), axis=0)

            matrix_R = matrix_R[np.logical_not(np.isnan(matrix_R).any(axis=1))]
            matrix_G = matrix_G[np.logical_not(np.isnan(matrix_G).any(axis=1))]
            matrix_B = matrix_B[np.logical_not(np.isnan(matrix_B).any(axis=1))]

            '''matrix_R = np.delete(matrix_R, np.argwhere(matrix_R[:, 0] < 0.9), 0)
            matrix_R = np.delete(matrix_R, np.argwhere(np.absolute(matrix_R[:, 1]) > 1), 0)
            matrix_R = np.delete(matrix_R, np.argwhere(np.absolute(matrix_R[:, 2]) > 1), 0)

            matrix_G = np.delete(matrix_G, np.argwhere(matrix_G[:, 1] < 0.9), 0)
            matrix_G = np.delete(matrix_G, np.argwhere(np.absolute(matrix_G[:, 0]) > 1), 0)
            matrix_G = np.delete(matrix_G, np.argwhere(np.absolute(matrix_G[:, 2]) > 1), 0)

            matrix_B = np.delete(matrix_B, np.argwhere(matrix_B[:, 2] < 0.9), 0)
            matrix_B = np.delete(matrix_B, np.argwhere(np.absolute(matrix_B[:, 1]) > 1), 0)
            matrix_B = np.delete(matrix_B, np.argwhere(np.absolute(matrix_B[:, 0]) > 1), 0)'''

            '''if checker == 'spydercheckr24':
                matrix_R = np.delete(matrix_R,
                                     [5, 7, 14, 15, 17, 19, 25, 29, 30, 31, 32,
                                      34, 36, 38, 42, 45, 47, 49, 51, 52, 53], 0)
                matrix_G = np.delete(matrix_G,
                                     [5, 6, 13, 14, 15, 17, 19, 20, 24, 29, 30,
                                      31, 33, 36, 37, 38, 39, 41, 43, 44, 45, 46,
                                      49, 50, 51, 52, 53, 55, 56, 57], 0)
                matrix_B = np.delete(matrix_B,
                                     [5, 8, 14, 21, 24, 25, 27, 30, 33, 36, 40,
                                      41, 42, 45, 46, 49, 51, 52, 54, 55, 56, 57], 0)
            elif checker == 'x-rite':
                matrix_R = np.delete(matrix_R,
                                     [2, 5, 6, 7, 12, 14, 15, 18, 19, 20, 23,
                                      26, 30, 31, 32, 33, 35, 38, 43, 44, 45, 46, 48, 50], 0)
                matrix_G = np.delete(matrix_G,
                                     [2, 5, 6, 8, 13, 14, 16, 18, 19, 20,
                                      21, 24, 27, 30, 31, 32, 34, 36, 37,
                                      40, 41, 42, 43, 44, 47, 48, 50, 51, 52, 53], 0)
                matrix_B = np.delete(matrix_B,
                                     [5, 8, 13, 16, 20, 23, 24, 27, 28, 30,
                                      32, 35, 36, 37, 40, 41, 42, 44, 45, 46, 47], 0)'''

            matrix = np.array(
                [matrix_R.sum(axis=0) / len(matrix_R), matrix_G.sum(axis=0) / len(matrix_G),
                 matrix_B.sum(axis=0) / len(matrix_B)]
            )

            '''if checker == 'x-rite':
                matrix -= np.array([[0.07383913, -0.01645874, -0.0573804],
                                    [-0.12457132,  0.2080515,  -0.08348018],
                                    [-0.07460538, -0.04046158,  0.10687024]])
            elif checker == 'spydercheckr24':
                matrix -= np.array([[0.08126999,  0.03904493, -0.12031491],
                                    [-0.15891245,  0.16444426, -0.0055318],
                                    [-0.123535,    0.02758223,  0.10464843]])
            '''
            if debug:
                print(np.around(np.array(matrix - np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])), 8))
                print(grade.title(), '\r\n')
                # print('R:\r\n', matrix_R, '\r\nG:\r\n', matrix_G, '\r\nB:\r\n', matrix_B)
                print('Matrix:\r\n', np.around(normalize_matrix(matrix), 2), '\r\nMatrix sum:', matrix.sum())
                # print('Etalon colors:\r\n', np.array(spydercheckr24_colors[6:9]))
                print('From sample:\r\n', np_sample[6:12])
                print('Expected colors:\r\n', exp[6:12])
                # print('Applied matrix:\r\n', apply_matrix(np_sample[6:9], matrix).astype('int'))
            # print(red_exp, green_exp, blue_exp)
            return matrix

        # Shadows
        cubes[temperature]['shadows'] = making_matrix('shadows')

        # Midtones
        cubes[temperature]['midtones'] = making_matrix('midtones')

        # Lights
        cubes[temperature]['lights'] = making_matrix('lights')

        # Step 2. Trying to enhance results. Normalizing saturation and trying to get better color balance

        if '--debug' in argv:
            print('Sample colors 9-11:\r\n', np_sample_was[9:12])
            print('Expected 9-11 colors:\r\n', exp[9:12])
            print('Matrix after 1st step:\r\n', normalize_matrix(cubes[temperature]['midtones']))
            debug_values = apply_matrix(np_sample_was[9:12], cubes[temperature]['midtones'])
            print('With current matrix  after 1st step applied:\r\n', debug_values.astype('int'))

        if '--gcam' in argv:
            r = 30
        else:
            r = 20
        for x in range(r):
            mod = find_matrix_changer(np_etalon[9:12],
                                      apply_matrix(np_sample_was[9:12], normalize_matrix(cubes[temperature]['midtones'])))
            # mod = find_matrix_changer(np_etalon[9:12], np_sample[9:12])

            # print(cubes[temperature]['midtones'], '\r\n', mod)

            # Shadows
            r_sum = cubes[temperature]['shadows'][0].sum()
            g_sum = cubes[temperature]['shadows'][1].sum()
            b_sum = cubes[temperature]['shadows'][2].sum()

            new_red = mod[0] + cubes[temperature]['shadows'][0]
            new_green = mod[1] + cubes[temperature]['shadows'][1]
            new_blue = mod[2] + cubes[temperature]['shadows'][2]

            new_matrix = np.array([new_red * r_sum / new_red.sum(),
                                   new_green * g_sum / new_green.sum(),
                                   new_blue * b_sum / new_blue.sum()])

            cubes[temperature]['shadows'] = new_matrix

            # Midtones
            r_sum = cubes[temperature]['midtones'][0].sum()
            g_sum = cubes[temperature]['midtones'][1].sum()
            b_sum = cubes[temperature]['midtones'][2].sum()

            new_red = mod[0] + cubes[temperature]['midtones'][0]
            new_green = mod[1] + cubes[temperature]['midtones'][1]
            new_blue = mod[2] + cubes[temperature]['midtones'][2]

            new_matrix = np.array([new_red * r_sum / new_red.sum(),
                                   new_green * g_sum / new_green.sum(),
                                   new_blue * b_sum / new_blue.sum()])

            cubes[temperature]['midtones'] = new_matrix

            # print(temperature, '\r\n', normalize_matrix(cubes[temperature]['lights']))

            # Lights
            r_sum = cubes[temperature]['lights'][0].sum()
            g_sum = cubes[temperature]['lights'][1].sum()
            b_sum = cubes[temperature]['lights'][2].sum()

            new_red = mod[0] + cubes[temperature]['lights'][0]
            new_green = mod[1] + cubes[temperature]['lights'][1]
            new_blue = mod[2] + cubes[temperature]['lights'][2]

            new_matrix = np.array([new_red * r_sum / new_red.sum(),
                                   new_green * g_sum / new_green.sum(),
                                   new_blue * b_sum / new_blue.sum()])

            cubes[temperature]['lights'] = new_matrix

    if '--nophone' not in argv and '--gcam' not in argv:
        save_cubes_to_phone(cubes, True)
    else:
        print('Текущая матрица:\r\n', normalize_matrix(cubes[temperature]['midtones']))
        save_cubes_to_local_file(cubes)
    # show_result()
    
    if '--noinputs' not in argv:
        once_more = input('Хотите еще раз проверить и уточнить калибровку? y/n: ')
    else:
        once_more = 'n'

    if once_more.lower() == 'y':
        check_calibration(cubes, np_sample_was=np_sample_was, np_etalon_was=np_etalon, temp=temp, backup=backup)

    if '--nophone' not in argv and '--gcam' not in argv:
        if '--cube' not in argv:
            if len(backup) > 0:
                backup[temperature] = cubes[temperature]
                save_cubes_to_phone(backup, False, temperature)
            else:
                save_cubes_to_phone(cubes, False, temperature)
        else:
            save_cubes_to_phone(cubes, True, temperature)
    else:
        if '--cube' not in argv:
            save_cubes_to_local_file(cubes, False, temperature)
        else:
            save_cubes_to_local_file(cubes, True, temperature)


def do_the_calibration(cubes, number_of_times, temperature='warm', backup=dict({})):
    print(help_message())
    print("Выполняется настройка баланса белого/черного.")
    # before_calib = get_colors_from_test_photo()

    if '--nophone' in argv:
        number_of_times = 1
    cubes_arr = []

    print('Сделайте снимок цветового эталона.')

    # Automatic shutter is not robust, so better keep out
    '''time.sleep(3)
    try:
        start_photoncamera()
        tap_shutter()
        pull_last_photo(wait_for_new_photo())
    except:
        print('Кажется, телефон не подключен. Возможно, другая проблема...')'''
    if '--nophone' not in argv:
        pull_last_photo(wait_for_new_photo(camera_folder))
        # wait_for_new_photo('.', True)
        time.sleep(1)

    col1 = get_colors_from_test_photo()

    if checker == 'x-rite':
        col1 = col1[0:6] + col1[11:5:-1] + col1[12:18] + col1[23:17:-1]
        np_etalon = np.array(xrite_remodeled)
    else:
        col2 = spydercheckr24_colors
        np_etalon = np.array(col2)
    np_sample = np.array(col1)

    for n in range(number_of_times):
        if '--nowb' not in argv:
            for i in range(1):  # Neutral colors (White balance fixing)

                # print(col1)
                # col2 = get_etalon_colors()

                # Finding contrast level
                '''contrast_level = np.sum(np_sample[0] - np_sample[5])
                contrast_etalon = np.sum(np_etalon[0] - np_etalon[5])
                if contrast_etalon / contrast_level > 1.5:
                    print('Контраст слишком низкий. Стоит попробовать понизить тени.')
                    low_contrast = True
                    high_contrast = False
                elif contrast_etalon / contrast_level < 0.75:
                    print('Контраст слишком высокий. Попробуйте поднять тени в настройках.')
                    high_contrast = True
                    low_contrast = False
                else:
                    low_contrast = False
                    high_contrast = False
                print('Контраст: ', contrast_level)
                print('Должен быть: ', contrast_etalon)
                '''
                # Computing multipliers for matrices
                diff = np_etalon - np_sample
                average_diff = np.array(diff.sum(axis=1) / 3, dtype="int").transpose()

                abs_diff = np.array([diff[:, 0] - average_diff, diff[:, 1] - average_diff, diff[:, 2]
                                     - average_diff], dtype='float').transpose()

                matrix_multipliers = np.divide(abs_diff, np_sample, out=np.zeros_like(abs_diff), where=np_sample != 0)

                lights_mult = matrix_multipliers[0]
                midtones_mult = matrix_multipliers[3]
                shadows_mult = matrix_multipliers[5]

                cubes[temperature]['shadows'] += cubes[temperature]['shadows'] * \
                    np.array([shadows_mult, shadows_mult, shadows_mult]).transpose()
                cubes[temperature]['midtones'] += cubes[temperature]['midtones'] * \
                    np.array([midtones_mult, midtones_mult, midtones_mult]).transpose()
                cubes[temperature]['lights'] += cubes[temperature]['lights'] * \
                    np.array([lights_mult, lights_mult, lights_mult]).transpose()
                # new_image = engage_cct_matrix(cubes, im, temperature)   # Applying computed matrix to the image
                # ImageShow.show(new_image)
                '''
                if '--nophone' not in argv and '--gcam' not in argv:
                    save_cubes_to_phone(cubes, True, temperature)
                if i == number_of_times - 1:
                    after_wb = col1  # Backuping colors after wb correction'''

        if '--nowb' not in argv:
            print('Настройка баланса белого/черного завершена.')
            # print('Было:\r\n', before_calib, '\r\n', 'Стало:\r\n', col1)

        print('Приступаем к калибровке цветов.')
        for i in range(1):  # Colors fixing

            # Initializing expected colors via checker colors comparison
            def init_exp_by_levels():
                exp = [[0, 0, 0] for k in range(24)]
                if checker == 'spydercheckr24':
                    # Red
                    exp[9][0] = round(
                        ((np_sample[1][0] - np_sample[2][0]) / 2 + np_sample[2][0]) * 1.025)  # Derived from etalon
                    gb_sum_exp = np_sample[4][0]
                    b_div_g_exp = 1.96
                    exp[9][2] = round(b_div_g_exp * gb_sum_exp / (b_div_g_exp + 1))
                    exp[9][1] = round(np_sample[5][1] * 0.634)
                    # More red
                    exp[14][0] = np_sample[1][2]
                    exp[14][1] = round((np_sample[5][1] + np_sample[5][2]) / 2)
                    exp[14][2] = round(np_sample[4][1] * 1.1875)
                    # Even more red
                    exp[8][0] = round(np_sample[0][0] * 0.984)
                    exp[8][1] = round(np_sample[1][1] * 1.035)
                    exp[8][2] = np_sample[5][2] - 43
                    if exp[8][2] < 0:
                        exp[8][2] = 0
                    exp[7][0] = round(np_sample[1][0] * 0.95)
                    exp[7][1] = round(np_sample[4][1] * 0.9375)
                    exp[7][2] = round(np_sample[2][2] * 0.941)

                    # Green
                    exp[10][1] = round(
                        ((np_sample[2][1] - np_sample[3][1]) / 2 + np_sample[3][1]) * 1.06)  # Derived from etalon
                    rb_sum_exp = np_sample[3][1]
                    b_div_r_exp = 1.123
                    exp[10][2] = round(rb_sum_exp * b_div_r_exp / (b_div_r_exp + 1))
                    exp[10][0] = round(np_sample[5][0] * 1.326)
                    # More green
                    exp[16][0] = np_sample[2][1]
                    exp[16][2] = round(np_sample[2][0] / 3)
                    exp[16][1] = round(0.96 * np_sample[1][2])
                    exp[20][0] = round(np_sample[4][0] * 1.025)
                    exp[20][1] = round(np_sample[3][1] * 0.899)
                    exp[20][2] = round(((np_sample[4][2] - np_sample[5][2]) / 2) + np_sample[5][2])

                    # Blue
                    exp[11][2] = round(((np_sample[2][2] - np_sample[3][2]) / 2) + np_sample[3][2])
                    rg_sum_exp = np_sample[4][2]
                    g_div_r_exp = 2.2
                    exp[11][0] = round(np_sample[5][0] * 0.581)
                    # exp[11][1] = rg_sum_exp - exp[11][0]
                    exp[11][1] = round(1.341 * np_sample[5][1])
                    # More blue
                    exp[6][0] = np_sample[5][0] - 43
                    if exp[6][0] < 0:
                        exp[6][0] = 0
                    exp[6][1] = round(np_sample[3][1] * 1.076)
                    exp[6][2] = round(np_sample[2][2] * 1.032467532)

                elif checker == 'x-rite':
                    exp[9][0] = round(
                        ((np_sample[1][0] - np_sample[2][0]) / 2 + np_sample[2][0]) * 0.97222222)
                    gb_sum_exp = round(np_sample[3][0] * 0.9344262295081968)
                    b_div_g_exp = 1.1111111111111112
                    exp[9][2] = round(b_div_g_exp * gb_sum_exp / (b_div_g_exp + 1))
                    exp[9][1] = round(np_sample[5][1] * 1.0384615384615385)
                    # More red
                    exp[14][0] = round(np_sample[1][2] * 0.965)
                    exp[14][1] = round(np_sample[4][1] * 1.0588235294117647)
                    exp[14][2] = round(np_sample[4][2] * 1.1647058823529413)
                    # Even more red
                    exp[8][0] = round(np_sample[0][0] * 0.9506172839506173)
                    exp[8][1] = round(np_sample[1][1] * 0.995)
                    exp[8][2] = round(np_sample[5][2] * 0.5961538461538461)

                    exp[7][0] = round(np_sample[1][0] * 0.935)
                    exp[7][1] = round(np_sample[4][1] * 1.011764705882353)
                    exp[7][2] = round(np_sample[2][2] * 0.93125)

                    # Green
                    exp[10][1] = round(np_sample[2][1] * 0.925)  # Derived from etalon colors
                    rb_sum_exp = round(((np_sample[2][1] - np_sample[3][1]) / 2 + np_sample[3][1]) * 1.0141843971631206)
                    b_div_r_exp = 1.042857142857143
                    exp[10][2] = round(rb_sum_exp * b_div_r_exp / (b_div_r_exp + 1))
                    exp[10][0] = round(np_sample[5][0] * 1.3461538461538463)
                    # More green
                    exp[16][0] = round(np_sample[2][1] * 0.98125)
                    exp[16][2] = round(np_sample[5][2] * 1.2307692307692308)
                    exp[16][1] = round(0.94 * np_sample[1][2])

                    exp[20][0] = round(np_sample[4][0] * 1.0235294117647058)
                    exp[20][1] = round(np_sample[3][1] * 0.9836065573770492)
                    exp[20][2] = round(np_sample[2][2] * 0.9836065573770492)

                    # Blue
                    exp[11][2] = round(np_sample[2][2] * 0.9375)  # Derived from etalon colors
                    exp[11][0] = round(np_sample[5][0] * 1.0769230769230769)
                    exp[11][1] = round(1.1730769230769231 * np_sample[5][1])
                    # More blue
                    exp[6][0] = round(np_sample[5][0] / 6.5)
                    exp[6][1] = round(np_sample[3][1] * 1.0901639344262295)
                    exp[6][2] = round(np_sample[2][2] * 1.00625)

                return exp

            def init_exp_by_etalon():
                exp = (np_etalon - np.sum(np_etalon, axis=1)[:, None]/3 + np.sum(np_sample, axis=1)[:, None]/3).astype('int')
                return exp

            exp = init_exp_by_levels()
            # exp = init_exp_by_etalon().tolist()

            def making_matrix(grade, debug=False):
                # Getting rows of the matrix
                matrix_R = []
                matrix_G = []
                matrix_B = []

                s = np_sample

                r_sum = np.sum(cubes[temperature][grade][0])
                g_sum = np.sum(cubes[temperature][grade][1])
                b_sum = np.sum(cubes[temperature][grade][2])
                c_sums = (r_sum, g_sum, b_sum)
                matrices = (matrix_R, matrix_G, matrix_B)
                exp_copy = exp.copy()
                # print(exp_copy)
                m = 0
                n = 0
                for o in range(3):
                    for m in range(len(exp_copy)):
                        for n in range(m + 1, len(exp_copy)):
                            if exp_copy[m] != [0, 0, 0]:
                                res = find_matrix_from_points([s[m], s[n]], [exp_copy[m][o], exp_copy[n][o]], c_sums[o])
                                if min(res) < -2.5 or max(res) > 2.5:
                                    continue
                                else:
                                    matrices[o].append(res)

                '''matrix_R.append(find_matrix_from_points([s[8], s[7]], [exp[8][0], exp[7][0]], r_sum))
                matrix_R.append(find_matrix_from_points([s[7], np_sample[5]], [exp[7][0], np_sample[5][0]], r_sum))
                matrix_R.append(find_matrix_from_points([s[7], s[16]], [exp[7][0], exp[16][0]], r_sum))
                # matrix_R.append(find_matrix_from_points([s[8], np_sample[5]], [exp[8][0], np_sample[5][0]], r_sum))
                matrix_R.append(find_matrix_from_points([s[16], s[11]], [exp[16][0], exp[11][0]], r_sum))
                matrix_R.append(find_matrix_from_points([s[11], s[6]], [exp[11][0], exp[6][0]], r_sum))
                #matrix_R.append(find_matrix_from_points([s[8], s[16]], [exp[8][0], exp[16][0]], r_sum))
                # matrix_R.append(find_matrix_from_points([s[6], s[16]], [exp[6][0], exp[16][0]], r_sum))
    
                matrix_G.append(find_matrix_from_points(s[9:11], [exp[9][1], exp[10][1]], g_sum))
                # matrix_G.append(find_matrix_from_points([s[16], s[10]], [exp[16][1], exp[10][1]], g_sum))
                matrix_G.append(find_matrix_from_points([s[8], s[7]], [exp[8][1], exp[7][1]], g_sum))
                matrix_G.append(find_matrix_from_points([s[9], s[16]], [exp[9][1], exp[16][1]], g_sum))
                matrix_G.append(find_matrix_from_points([s[9], s[7]], [exp[9][1], exp[7][1]], g_sum))
                # matrix_G.append(find_matrix_from_points([s[8], s[6]], [exp[8][1], exp[6][1]], g_sum))
                matrix_G.append(find_matrix_from_points([s[16], s[6]], [exp[16][1], exp[6][1]], g_sum))
                matrix_G.append(find_matrix_from_points([s[10], s[6]], [exp[10][1], exp[6][1]], g_sum))
    
                matrix_B.append(find_matrix_from_points([s[10], s[9]], [exp[10][2], exp[9][2]], b_sum))
                matrix_B.append(find_matrix_from_points([s[16], s[9]], [exp[16][2], exp[9][2]], b_sum))
                matrix_B.append(find_matrix_from_points([s[16], s[10]], [exp[16][2], exp[10][2]], b_sum))
                matrix_B.append(find_matrix_from_points([s[6], s[10]], [exp[6][2], exp[10][2]], b_sum))
                matrix_B.append(find_matrix_from_points([s[6], s[9]], [exp[6][2], exp[9][2]], b_sum))
                matrix_B.append(find_matrix_from_points([s[6], s[7]], [exp[6][2], exp[7][2]], b_sum))
                matrix_B.append(find_matrix_from_points([s[6], s[16]], [exp[6][2], exp[16][2]], b_sum))
                matrix_B.append(find_matrix_from_points([s[8], s[10]], [exp[8][2], exp[10][2]], b_sum))
                matrix_B.append(find_matrix_from_points([s[8], s[7]], [exp[8][2], exp[7][2]], b_sum))'''

                matrix_R = np.array(matrix_R)
                matrix_G = np.array(matrix_G)
                matrix_B = np.array(matrix_B)

                matrix_R = np.delete(matrix_R, np.argwhere(matrix_R[:, 0] < 0), axis=0)
                matrix_G = np.delete(matrix_G, np.argwhere(matrix_G[:, 1] < 0), axis=0)
                matrix_B = np.delete(matrix_B, np.argwhere(matrix_B[:, 2] < 0), axis=0)

                matrix_R = matrix_R[np.logical_not(np.isnan(matrix_R).any(axis=1))]
                matrix_G = matrix_G[np.logical_not(np.isnan(matrix_G).any(axis=1))]
                matrix_B = matrix_B[np.logical_not(np.isnan(matrix_B).any(axis=1))]

                '''matrix_R = np.delete(matrix_R, np.argwhere(matrix_R[:, 0] < 0.9), 0)
                matrix_R = np.delete(matrix_R, np.argwhere(np.absolute(matrix_R[:, 1]) > 1), 0)
                matrix_R = np.delete(matrix_R, np.argwhere(np.absolute(matrix_R[:, 2]) > 1), 0)

                matrix_G = np.delete(matrix_G, np.argwhere(matrix_G[:, 1] < 0.9), 0)
                matrix_G = np.delete(matrix_G, np.argwhere(np.absolute(matrix_G[:, 0]) > 1), 0)
                matrix_G = np.delete(matrix_G, np.argwhere(np.absolute(matrix_G[:, 2]) > 1), 0)

                matrix_B = np.delete(matrix_B, np.argwhere(matrix_B[:, 2] < 0.9), 0)
                matrix_B = np.delete(matrix_B, np.argwhere(np.absolute(matrix_B[:, 1]) > 1), 0)
                matrix_B = np.delete(matrix_B, np.argwhere(np.absolute(matrix_B[:, 0]) > 1), 0)'''

                '''if checker == 'spydercheckr24':
                    matrix_R = np.delete(matrix_R,
                                         [5, 7, 14, 15, 17, 19, 25, 29, 30, 31, 32,
                                          34, 36, 38, 42, 45, 47, 49, 51, 52, 53], 0)
                    matrix_G = np.delete(matrix_G,
                                         [5, 6, 13, 14, 15, 17, 19, 20, 24, 29, 30,
                                          31, 33, 36, 37, 38, 39, 41, 43, 44, 45, 46,
                                          49, 50, 51, 52, 53, 55, 56, 57], 0)
                    matrix_B = np.delete(matrix_B,
                                         [5, 8, 14, 21, 24, 25, 27, 30, 33, 36, 40,
                                          41, 42, 45, 46, 49, 51, 52, 54, 55, 56, 57], 0)
                elif checker == 'x-rite':
                    matrix_R = np.delete(matrix_R,
                                         [2, 5, 6, 7, 12, 14, 15, 18, 19, 20, 23,
                                          26, 30, 31, 32, 33, 35, 38, 43, 44, 45, 46, 48, 50], 0)
                    matrix_G = np.delete(matrix_G,
                                         [2, 5, 6, 8, 13, 14, 16, 18, 19, 20,
                                          21, 24, 27, 30, 31, 32, 34, 36, 37,
                                          40, 41, 42, 43, 44, 47, 48, 50, 51, 52, 53], 0)
                    matrix_B = np.delete(matrix_B,
                                         [5, 8, 13, 16, 20, 23, 24, 27, 28, 30,
                                          32, 35, 36, 37, 40, 41, 42, 44, 45, 46, 47], 0)'''

                matrix = np.array(
                    [matrix_R.sum(axis=0) / len(matrix_R), matrix_G.sum(axis=0) / len(matrix_G),
                     matrix_B.sum(axis=0) / len(matrix_B)]
                )

                '''if checker == 'x-rite':
                    matrix -= np.array([[0.07383913, -0.01645874, -0.0573804],
                                        [-0.12457132,  0.2080515,  -0.08348018],
                                        [-0.07460538, -0.04046158,  0.10687024]])
                elif checker == 'spydercheckr24':
                    matrix -= np.array([[0.08126999,  0.03904493, -0.12031491],
                                        [-0.15891245,  0.16444426, -0.0055318],
                                        [-0.123535,    0.02758223,  0.10464843]])
                '''
                if debug:
                    print(np.around(np.array(matrix - np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])), 8))
                    print(grade.title(), '\r\n')
                    # print('R:\r\n', matrix_R, '\r\nG:\r\n', matrix_G, '\r\nB:\r\n', matrix_B)
                    print('Matrix:\r\n', np.around(normalize_matrix(matrix), 2), '\r\nMatrix sum:', matrix.sum())
                    # print('Etalon colors:\r\n', np.array(spydercheckr24_colors[6:9]))
                    print('From sample:\r\n', np_sample[6:12])
                    print('Expected colors:\r\n', exp[6:12])
                    # print('Applied matrix:\r\n', apply_matrix(np_sample[6:9], matrix).astype('int'))
                # print(red_exp, green_exp, blue_exp)
                return matrix

            # Shadows
            cubes[temperature]['shadows'] = making_matrix('shadows')

            # Midtones
            cubes[temperature]['midtones'] = making_matrix('midtones')

            # Lights
            cubes[temperature]['lights'] = making_matrix('lights')

            # Step 2. Trying to enhance results. Normalizing saturation and trying to get better color balance

            if '--debug' in argv:
                print('Sample colors 9-11:\r\n', np_sample[9:12])
                print('Expected 9-11 colors:\r\n', exp[9:12])
                print('Matrix after 1st step:\r\n', normalize_matrix(cubes[temperature]['midtones']))
                debug_values = apply_matrix(np_sample[9:12], cubes[temperature]['midtones'])
                print('With current matrix  after 1st step applied:\r\n', debug_values.astype('int'))

            if '--gcam' in argv:
                r = 30
            else:
                r = 20
            for x in range(r):
                mod = find_matrix_changer(np_etalon[9:12],
                                          apply_matrix(np_sample[9:12], normalize_matrix(cubes[temperature]['midtones'])))
                # mod = find_matrix_changer(np_etalon[9:12], np_sample[9:12])

                # print(cubes[temperature]['midtones'], '\r\n', mod)

                # Shadows
                r_sum = cubes[temperature]['shadows'][0].sum()
                g_sum = cubes[temperature]['shadows'][1].sum()
                b_sum = cubes[temperature]['shadows'][2].sum()

                new_red = mod[0] + cubes[temperature]['shadows'][0]
                new_green = mod[1] + cubes[temperature]['shadows'][1]
                new_blue = mod[2] + cubes[temperature]['shadows'][2]

                new_matrix = np.array([new_red * r_sum / new_red.sum(),
                                       new_green * g_sum / new_green.sum(),
                                       new_blue * b_sum / new_blue.sum()])

                cubes[temperature]['shadows'] = new_matrix

                # Midtones
                r_sum = cubes[temperature]['midtones'][0].sum()
                g_sum = cubes[temperature]['midtones'][1].sum()
                b_sum = cubes[temperature]['midtones'][2].sum()

                new_red = mod[0] + cubes[temperature]['midtones'][0]
                new_green = mod[1] + cubes[temperature]['midtones'][1]
                new_blue = mod[2] + cubes[temperature]['midtones'][2]

                new_matrix = np.array([new_red * r_sum / new_red.sum(),
                                       new_green * g_sum / new_green.sum(),
                                       new_blue * b_sum / new_blue.sum()])

                cubes[temperature]['midtones'] = new_matrix

                # print(temperature, '\r\n', normalize_matrix(cubes[temperature]['lights']))

                # Lights
                r_sum = cubes[temperature]['lights'][0].sum()
                g_sum = cubes[temperature]['lights'][1].sum()
                b_sum = cubes[temperature]['lights'][2].sum()

                new_red = mod[0] + cubes[temperature]['lights'][0]
                new_green = mod[1] + cubes[temperature]['lights'][1]
                new_blue = mod[2] + cubes[temperature]['lights'][2]

                new_matrix = np.array([new_red * r_sum / new_red.sum(),
                                       new_green * g_sum / new_green.sum(),
                                       new_blue * b_sum / new_blue.sum()])

                cubes[temperature]['lights'] = new_matrix

        if number_of_times > 1 and n < number_of_times - 1:
            cubes_arr.append(cubes)
    if number_of_times > 1:
        new_cubes = {
                    'warm': {
                        'shadows': np.array([[], [], []]),
                        'midtones': np.array([[], [], []]),
                        'lights': np.array([[], [], []])
                    },
                    'cool': {
                        'shadows': np.array([[], [], []]),
                        'midtones': np.array([[], [], []]),
                        'lights': np.array([[], [], []])
                    },
                    'warm_point': cubes['warm_point'],
                    'cool_point': cubes['cool_point']
                }
        for cube in cubes_arr:
            new_cubes[temperature]['shadows'] += cube[temperature]['shadows']
            new_cubes[temperature]['midtones'] += cube[temperature]['midtones']
            new_cubes[temperature]['lights'] += cube[temperature]['lights']
        new_cubes[temperature]['shadows'] /= number_of_times
        new_cubes[temperature]['midtones'] /= number_of_times
        new_cubes[temperature]['lights'] /= number_of_times

        cubes = new_cubes
    '''if checker == 'spydercheckr24':
        cubes[temperature]['midtones'] += np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) - \
                                          np.array([[0.9821, 0.0124, 0.0055],
                                                    [0.0161, 1.1759, -0.192],
                                                    [0.0046, -0.1854, 1.1894]])
    elif checker == 'x-rite':
        cubes[temperature]['midtones'] += np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) - \
                                          np.array([[1.0107, 0.0003, -0.0111],
                                                    [0.0061, 1.0145, -0.0206],
                                                    [-0.0196, -0.0175, 1.0289]])'''

    print('Калибровка завершена, удачной фотоохоты!')

    if '--nophone' not in argv and '--gcam' not in argv:
        if '--cube' not in argv:
            if len(backup) > 0:
                backup[temperature] = cubes[temperature]
                save_cubes_to_phone(backup, False, temperature)
            else:
                save_cubes_to_phone(cubes, False, temperature)
        else:
            save_cubes_to_phone(cubes, True, temperature)
    else:
        if '--cube' not in argv:
            save_cubes_to_local_file(cubes, False, temperature)
        else:
            save_cubes_to_local_file(cubes, True, temperature)

    # show_result()

    if '--noinputs' not in argv:
        if input('Проверить/уточнить калибровку? y/n: ').lower() == 'y':
            check_calibration(cubes, np_sample_was=np_sample, np_etalon_was=np_etalon, backup=backup)


backup_cubes = {}

if '--onlycheck' not in argv:
    '''if '--from0' not in argv:
        if '--nophone' not in argv:
            get_cct_matrix_file_from_phone()
        if '--fromprevcalib' in argv:
            cct_cubes = parse_cct_matrix_from_file('customCCT_autoCubes.txt')
        else:
            cct_cubes = parse_cct_matrix_from_file()
        backup_cubes = cct_cubes.copy()
    else:'''
    if '--nophone' not in argv and '--gcam' not in argv:
        if '--backupcct' in argv:
            get_cct_matrix_file_from_phone()
            backup_cubes = parse_cct_matrix_from_file()
            # print(backup_cubes)
            save_cubes_to_local_file(backup_cubes, filename='CCTbackup.txt')
            # exit()
    cct_cubes = parse_cct_matrix_from_file('customCCT_all1.txt')
    if '--nophone' not in argv and '--gcam' not in argv:
        save_cubes_to_phone(cct_cubes, True)

    if '--cool' in argv:
        do_the_calibration(cct_cubes, 1, 'cool', backup_cubes)
    else:
        do_the_calibration(cct_cubes, 1, 'warm', backup_cubes)
else:
    check_calibration(parse_cct_matrix_from_file('customCCT_autoCubes.txt'), backup=parse_cct_matrix_from_file('CCTbackup.txt'))
show_result()

# save_cubes_to_local_file(cct_cubes)
