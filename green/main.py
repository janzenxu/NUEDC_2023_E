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


def laser_control(border, roi_x_l, roi_y_l):
    global obj_x, obj_y, var_x, var_y, frame_stack

    while True:
        frame = frame_stack[-1]
        ret_red, obj_x_roi, obj_y_roi = vis.find_red_pt(frame)
        ret_green, var_x_roi, var_y_roi = vis.find_green_pt(frame)

        if ret_green and ret_red:
            var_x, var_y = vis.coordinate_tansform(var_x_roi, var_y_roi, border, roi_x_l, roi_y_l)
            obj_x, obj_y = vis.coordinate_tansform(obj_x_roi, obj_y_roi, border, roi_x_l, roi_y_l)

            message_send()


if __name__ == "__main__":
    roi_x, roi_y, roi_w, roi_h = find_roi()

    border_width = 10

    uart.write(b"A")

    thread_video_capture = th.Thread(target=video_capture, args=(roi_x, roi_y, roi_w, roi_h, border_width,))
    thread_laser_control = th.Thread(target=laser_control, args=(border_width, roi_x, roi_y, ))

    thread_video_capture.start()
    thread_laser_control.start()

    thread_video_capture.join()
    thread_laser_control.join()
