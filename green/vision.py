import cv2 as cv
import numpy as np


def morph_open(img_in, ksize):
    if ksize == 0:
        img_out = img_in
    else:
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (ksize, ksize))
        img_out = cv.morphologyEx(img_in, cv.MORPH_OPEN, kernel)
    return img_out


def morph_close(img_in, ksize):
    if ksize == 0:
        img_out = img_in
    else:
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (ksize, ksize))
        img_out = cv.morphologyEx(img_in, cv.MORPH_CLOSE, kernel)
    return img_out


def morph_erode(img_in, ksize):
    if ksize == 0:
        img_out = img_in
    else:
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (ksize, ksize))
        img_out = cv.erode(img_in, kernel)
    return img_out


def morph_dilate(img_in, ksize):
    if ksize == 0:
        img_out = img_in
    else:
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (ksize, ksize))
        img_out = cv.dilate(img_in, kernel)
    return img_out


def morph(img_in, erode_ksize, dilate_ksize, open_ksize, close_ksize):
    img_erode = morph_erode(img_in, erode_ksize)
    img_dilate = morph_dilate(img_erode, dilate_ksize)
    img_open = morph_open(img_dilate, open_ksize)
    img_out = morph_close(img_open, close_ksize)
    return img_out


def find_red_pt(img_in):
    blur_ksize = 1
    img_blur = cv.GaussianBlur(img_in, (blur_ksize, blur_ksize), 0)

    img_hsv = cv.cvtColor(img_blur, cv.COLOR_BGR2HSV)
    lower_range_red = np.array([140, 0, 220])
    upper_range_red = np.array([255, 70, 255])

    img_binary = cv.inRange(img_hsv, lower_range_red, upper_range_red)
    img_morph = morph(img_binary, 1, 5, 1, 3)
    cnts, hier = cv.findContours(img_morph, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    obj_x = 0
    obj_y = 0
    if cnts:
        ret = True
        cnt = cnts[0]
        moment_cnt = cv.moments(cnt)
        obj_x = int(moment_cnt['m10'] / moment_cnt['m00'])
        obj_y = int(moment_cnt['m01'] / moment_cnt['m00'])
    else:
        ret = False

    return ret, obj_x, obj_y


def find_green_pt(img_in):
    blur_ksize = 1
    img_blur = cv.GaussianBlur(img_in, (blur_ksize, blur_ksize), 0)

    img_hsv = cv.cvtColor(img_blur, cv.COLOR_BGR2HSV)
    lower_range_green = np.array([35, 43, 46])
    upper_range_green = np.array([80, 255, 255])

    img_binary = cv.inRange(img_hsv, lower_range_green, upper_range_green)
    img_morph = morph(img_binary, 1, 5, 1, 3)
    cnts, hier = cv.findContours(img_morph, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    obj_x = 0
    obj_y = 0
    if cnts:
        ret = True
        cnt = cnts[0]
        moment_cnt = cv.moments(cnt)
        obj_x = int(moment_cnt['m10'] / moment_cnt['m00'])
        obj_y = int(moment_cnt['m01'] / moment_cnt['m00'])
    else:
        ret = False

    return ret, obj_x, obj_y


def find_screen_pts(img_in):
    img_gray = cv.cvtColor(img_in, cv.COLOR_BGR2GRAY)

    clahe = cv.createCLAHE(3, (7, 7))
    img_equalize = clahe.apply(img_gray)

    ctr_y, ctr_x = int(img_gray.shape[0] / 2), int(img_gray.shape[1] / 2)

    corners = cv.goodFeaturesToTrack(img_equalize, 8, 0.03, 80)
    corner_pts = []

    if len(corners) < 4:
        ret = False
    else:
        ret = True
        for corner in corners:
            corner_x, corner_y = corner.ravel()
            distance = np.sqrt((corner_x - ctr_x)**2 + (corner_y - ctr_y)**2)
            corner_pts.append((int(corner_x), int(corner_y)))

        corner_pts = sorted(corner_pts, key=lambda p: np.sqrt((p[0] - ctr_x)**2 + (p[1] - ctr_y)**2), reverse=True)
        corner_pts = corner_pts[:4]
    return ret, corner_pts


def find_box_pts(img_in):
    blur_ksize = 3
    img_blur = cv.GaussianBlur(img_in, (blur_ksize, blur_ksize), 0)
    img_gray = cv.cvtColor(img_blur, cv.COLOR_BGR2GRAY)
    ret, img_binary = cv.threshold(img_gray, 100, 255, cv.THRESH_BINARY_INV)
    img_morph = morph(img_binary, 3, 1, 1, 2)

    min_area = 9000
    max_area = 15000

    box_pts = []

    ret = False
    cnts, hier = cv.findContours(img_morph, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    if cnts:
        for idx, cnt in enumerate(cnts):
            current_hierarchy = hier[0][idx]
            next_idx, prev_idx, child_idx, parent_idx = current_hierarchy

            if min_area < cv.contourArea(cnt) < max_area and parent_idx != -1:
                ret = True
                inner_cnt = cnt

        if ret:
            rect = cv.minAreaRect(inner_cnt)
            coordinate = np.int0(cv.boxPoints(rect))
            for pt in coordinate:
                box_pts.append(tuple(pt))
            # cv.polylines(img_in, [coordinate], )
    return ret, box_pts


def get_distance(p1, p2):
    distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return distance


def get_trace(pts, step):
    trace_pts = []
    for i in range(len(pts)):
        x_step = (np.array(pts[i + 1 if i + 1 < 4 else 0]) - np.array(pts[i]))[0] / step
        y_step = (np.array(pts[i + 1 if i + 1 < 4 else 0]) - np.array(pts[i]))[1] / step
        for j in range(step):
            trace_pts.append((int(pts[i][0] + x_step * j), int(pts[i][1] + y_step * j)))
    trace_pts.append(tuple(pts[0]))
    return trace_pts


def coordinate_tansform(coor_x, coor_y, border, roi_x_l, roi_y_l):
    real_x = int(coor_x - border + roi_x_l)
    real_y = int(coor_y - border + roi_y_l)
    return real_x, real_y
