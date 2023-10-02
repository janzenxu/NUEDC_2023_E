import threading as th
import time
import serial as ser
import vision as vis
import cv2 as cv
import numpy as np

var_x = 320
var_y = 240
obj_x = 320
obj_y = 240

frame_stack = []

uart = ser.Serial("/dev/ttyAMA0", 115200)


def find_roi():
    video = cv.VideoCapture(0, cv.CAP_V4L2)
    video.set(cv.CAP_PROP_BUFFERSIZE, 1)

    rect_max = [0, 0, 0, 0]
    rect_count = 0
    rect_count_max = 10

    while video.isOpened():
        ret, frame = video.read()
        if ret:
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            block_size = 21
            C = 4
            frame_binary = cv.adaptiveThreshold(frame_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, block_size, C)

            erode_ksize = 1
            dilate_ksize = 4
            open_ksize = 1
            close_ksize = 5

            frame_morph = vis.morph(frame_binary, erode_ksize, dilate_ksize, open_ksize, close_ksize)

            min_area = 51060
            max_area = 62510

            cnts, hier = cv.findContours(frame_morph, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

            if rect_count < rect_count_max:
                if cnts:
                    for cnt in cnts:
                        rect_x, rect_y, rect_w, rect_h = cv.boundingRect(cnt)
                        rect_area = rect_w * rect_h
                        if min_area < rect_area < max_area:
                            rect_count += 1
                            if rect_area > rect_max[2] * rect_max[3]:
                                rect_max = [rect_x, rect_y, rect_w, rect_h]
            else:
                break
        else:
            break
    video.release()
    return rect_max


def find_screen(rect_x, rect_y, rect_w, rect_h, border):
    video = cv.VideoCapture(0, cv.CAP_V4L2)
    video.set(cv.CAP_PROP_BUFFERSIZE, 1)

    ctr_x, ctr_y = rect_w / 2 + border, rect_h / 2 + border
    pts_pre = [(ctr_x, ctr_y)] * 4
    frame_count = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            frame_roi = frame[rect_y - border: rect_y + rect_h + border, rect_x - border: rect_x + rect_w + border, :].copy()
            ret_screen, screen_pts = vis.find_screen_pts(frame_roi)
            if ret_screen:
                screen_pts_sorted = [(ctr_x, ctr_y)] * 4
                for pt in screen_pts:
                    if (pt[0] - ctr_x) * (pt[1] - ctr_y) > 0:
                        if pt[0] - ctr_x > 0:
                            idx = 0
                        else:
                            idx = 2
                    else:
                        if pt[0] - ctr_x > 0:
                            idx = 3
                        else:
                            idx = 1
                    screen_pts_sorted[idx] = pt

                if np.allclose(np.array(screen_pts_sorted), np.array(pts_pre), atol=2):
                    frame_count += 1
                else:
                    frame_count = 0

                pts_pre = (0.7 * np.array(pts_pre) + 0.3 * np.array(screen_pts_sorted)).tolist()

                if frame_count >= 10:
                    screen_pts_best = pts_pre

                    for pt in screen_pts_best:
                        pt[0] = int(pt[0] - border + rect_x)
                        pt[1] = int(pt[1] - border + rect_y)

                    ctr_x_temp = (screen_pts_best[0][0] + screen_pts_best[2][0]) / 2
                    ctr_y_temp = (screen_pts_best[0][1] + screen_pts_best[2][1]) / 2
                    ctr_x = (screen_pts_best[1][0] + screen_pts_best[3][0]) / 2
                    ctr_y = (screen_pts_best[1][1] + screen_pts_best[3][1]) / 2
                    ctr_x = int((ctr_x + ctr_x_temp) / 2)
                    ctr_y = int((ctr_y + ctr_y_temp) / 2)

                    video.release()
                    return screen_pts_best, (ctr_x, ctr_y)


def find_box(rect_x, rect_y, rect_w, rect_h, border):
    global frame_stack

    ctr_x, ctr_y = rect_w / 2 + border, rect_h / 2 + border
    pts_pre = [(ctr_x, ctr_y)] * 4
    frame_count = 0
    while True:
        frame = frame_stack[-1]
        ret_box, box_pts = vis.find_box_pts(frame)
        if ret_box:
            if np.allclose(np.array(box_pts), np.array(pts_pre), atol=2):
                frame_count += 1
            else:
                frame_count = 0
            pts_pre = (0.5 * np.array(pts_pre) + 0.5 * np.array(box_pts)).tolist()

            if frame_count >= 10:
                box_pts_best = pts_pre

                for pt in box_pts_best:
                    pt[0], pt[1] = vis.coordinate_tansform(pt[0], pt[1], border, rect_x, rect_y)
                return box_pts_best


def video_capture(roi_x, roi_y, roi_w, roi_h, border_width):
    global frame_stack
    video = cv.VideoCapture(0, cv.CAP_V4L2)
    video.set(cv.CAP_PROP_BUFFERSIZE, 5)
    lock = th.RLock()
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            lock.acquire()
            if len(frame_stack) >= 25:
                frame_stack.clear()
            frame_stack.append(frame[roi_y - border_width: roi_y + roi_h + border_width,
                                     roi_x - border_width: roi_x + roi_w + border_width, :])
            lock.release()
        else:
            break
        time.sleep(0.01)


def message_send():
    global var_x, var_y, obj_x, obj_y

    uart.write(b"Z")
    uart.write(bytes.fromhex("%04x" % var_x))
    uart.write(bytes.fromhex("%04x" % var_y))
    uart.write(bytes.fromhex("%04x" % obj_x))
    uart.write(bytes.fromhex("%04x" % obj_y))


def laser_control(center, border, roi_x_l, roi_y_l, rect_w, rect_h, trace_pts):
    global uart, frame_stack

    while True:
        uart.flushInput()
        message_count = uart.inWaiting()
        if message_count != 0:
            recv = uart.readline()
            print(recv)
            if recv == b"2\n":
                reset_center(center, border, roi_x_l, roi_y_l)
            elif recv == b"3\n":
                trace(trace_pts, border, roi_x_l, roi_y_l)
                reset_center(center, border, roi_x_l, roi_y_l)
            elif recv == b"4\n":
                box_points = find_box(roi_x_l, roi_y_l, rect_w, rect_h, border)
                box_trace = vis.get_trace(box_points, 21)
                trace(box_trace, border, roi_x_l, roi_y_l)
                reset_center(center, border, roi_x_l, roi_y_l)
        time.sleep(0.1)


def reset_center(center, border, roi_x_l, roi_y_l):
    global obj_x, obj_y, var_x, var_y, frame_stack

    obj_x = center[0]
    obj_y = center[1]

    while True:
        frame = frame_stack[-1]

        ret_red, var_x_roi, var_y_roi = vis.find_red_pt(frame)
        if ret_red:
            var_x, var_y = vis.coordinate_tansform(var_x_roi, var_y_roi, border, roi_x_l, roi_y_l)
            message_send()
            if vis.get_distance((obj_x, obj_y), (var_x, var_y)) < 2:
                break


def trace(trace_pts, border, roi_x_l, roi_y_l):
    global obj_x, obj_y, var_x, var_y, frame_stack

    obj_x = trace_pts[0][0]
    obj_y = trace_pts[0][1]
    trace_idx = 0

    while True:
        frame = frame_stack[-1]
        t0 = time.time()

        ret_red, var_x_roi, var_y_roi = vis.find_red_pt(frame)
        if ret_red:
            var_x, var_y = vis.coordinate_tansform(var_x_roi, var_y_roi, border, roi_x_l, roi_y_l)
            if vis.get_distance((obj_x, obj_y), (var_x, var_y)) < 15:
                trace_idx += 1

            message_send()

            if trace_idx == len(trace_pts):
                # trace_idx = 0
                break

            obj_x, obj_y = trace_pts[trace_idx]
        t1 = time.time()
        print(t1 - t0)


if __name__ == "__main__":
    roi_x, roi_y, roi_w, roi_h = find_roi()

    border_width = 10

    screen_points, screen_center = find_screen(roi_x, roi_y, roi_w, roi_h, border_width)

    screen_step = 10
    screen_trace = vis.get_trace(screen_points, screen_step)

    uart.write(b"A")

    thread_video_capture = th.Thread(target=video_capture, args=(roi_x, roi_y, roi_w, roi_h, border_width,))
    thread_laser_control = th.Thread(target=laser_control, args=(screen_center, border_width, roi_x, roi_y, roi_w, roi_h, screen_trace,))

    thread_video_capture.start()
    thread_laser_control.start()

    thread_video_capture.join()
    thread_laser_control.join()
