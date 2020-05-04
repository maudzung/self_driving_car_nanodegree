import time
import os
import numpy as np

import cv2
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

from undistort_img import undistort, calibrate
from gradient import get_binary_img
from perspective_transform import get_transform_matrix, warped_birdview
from detect_lanelines import find_lane_sliding_window, find_lane_based_on_previous_frame
from utils import transform_to_the_road, Line


def process_image(image):
    global lane_left, lane_right, frame_idx
    frame_idx += 1
    h, w = image.shape[:2]
    src = np.float32([
        [(w / 2) - 55, h / 2 + 100],
        [((w / 6) - 10), h],
        [(w * 5 / 6) + 60, h],
        [(w / 2 + 55), h / 2 + 100]
    ])
    dst = np.float32([
        [(w / 4), 0],
        [(w / 4), h],
        [(w * 3 / 4), h],
        [(w * 3 / 4), 0]
    ])

    undistorted_img = undistort(image, mtx, dist)
    binary_output = get_binary_img(undistorted_img, thresh_gradx, thresh_grady, thresh_mag, thresh_dir,
                                   thresh_s_channel)

    M, Minv = get_transform_matrix(src, dst)
    binary_warped = warped_birdview(binary_output, M)

    if (frame_idx > 0) and lane_left.detected and lane_right.detected:
        out_img, lane_left, lane_right, left_fit_x, right_fit_x, ploty = find_lane_based_on_previous_frame(
            binary_warped, margin, lane_left, lane_right)
    else:
        out_img, lane_left, lane_right, left_fit_x, right_fit_x, ploty = find_lane_sliding_window(binary_warped,
                                                                                                  nwindows, margin,
                                                                                                  minpix, lane_left,
                                                                                                  lane_right)

    out_img = transform_to_the_road(undistorted_img, Minv, left_fit_x, right_fit_x, ploty)

    # lane_left.cal_curvature(left_fit, h)
    # lane_right.cal_curvature(right_fit, h)

    return out_img


def main():
    video_output_dir = '../test_videos_output'
    if not os.path.isdir(video_output_dir):
        os.makedirs(video_output_dir)

    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    # video_fn = 'project_video.mp4'
    video_fn = 'challenge_video.mp4'
    video_fn = 'harder_challenge_video.mp4'
    video_output_path = os.path.join(video_output_dir, video_fn)
    clip1 = VideoFileClip(os.path.join('../', video_fn))
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(video_output_path, audio=False)


if __name__ == '__main__':
    lane_left, lane_right = Line(buffer_len=20), Line(buffer_len=20)
    frame_idx = -1
    nwindows = 9
    margin = 100
    minpix = 50
    thresh_gradx = (20, 100)
    thresh_grady = (20, 100)
    thresh_mag = (30, 100)
    thresh_dir = (0.7, 1.3)
    thresh_s_channel = (170, 255)
    ret, mtx, dist, rvecs, tvecs = calibrate(is_save=False)
    main()
