import os
from glob import glob
import sys
import collections

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, buffer_len=20):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.last_fit = None
        self.recent_fits = collections.deque(maxlen=buffer_len)
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def update_lane(self, new_curve_fit, detected, new_lane_x, new_lane_y):
        self.detected = detected
        self.last_fit = new_curve_fit
        self.recent_fits.append(new_curve_fit)
        self.allx = new_lane_x
        self.ally = new_lane_y

    def cal_curvature(self, curve_fit, h):
        y_eval = int(h / 2)
        self.radius_of_curvature = np.sqrt((1 + (2 * curve_fit[0] * y_eval + curve_fit[1]) ** 2) ** 3) / np.absolute(
            2 * curve_fit[0])


# def transform_to_the_road(undistorted_img, Minv, left_lane, right_lane):
def transform_to_the_road(undistorted_img, Minv, left_fit_x, right_fit_x, ploty):
    h, w = undistorted_img.shape[:2]

    road_warped = np.zeros_like(undistorted_img, dtype=np.uint8)
    pts_left = np.array([np.transpose(np.vstack([left_fit_x, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(road_warped, np.int_([pts]), (0, 255, 0))
    road_unwarped = cv2.warpPerspective(road_warped, Minv, (w, h))  # Warp back to original image space

    blend_img = cv2.addWeighted(undistorted_img, 1., road_unwarped, 0.5, 0)

    return blend_img


if __name__ == '__main__':
    from undistort_img import undistort, calibrate
    from gradient import get_binary_img
    from perspective_transform import get_transform_matrix, warped_birdview
    from detect_lanelines import fit_polynomial

    nwindows = 9
    margin = 100
    minpix = 50
    thresh_gradx = (20, 100)
    thresh_grady = (20, 100)
    thresh_mag = (30, 100)
    thresh_dir = (0.7, 1.3)
    thresh_s_channel = (170, 255)

    output_images_dir = '../output_images'
    output_detectedline_img = os.path.join(output_images_dir, 'onroad_test_images')
    if not os.path.isdir(output_detectedline_img):
        os.makedirs(output_detectedline_img)

    img_paths = glob('../test_images/*.jpg')

    ret, mtx, dist, rvecs, tvecs = calibrate(is_save=True)

    for idx, img_path_ in enumerate(img_paths):
        img_fn = os.path.basename(img_path_)[:-4]
        img = cv2.cvtColor(cv2.imread(img_path_), cv2.COLOR_BGR2RGB)  # BGR --> RGB
        undistorted_img = undistort(img, mtx, dist)
        binary_output = get_binary_img(undistorted_img, thresh_gradx, thresh_grady, thresh_mag, thresh_dir,
                                       thresh_s_channel)

        h, w = binary_output.shape[:2]
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
        M, Minv = get_transform_matrix(src, dst)

        warped = warped_birdview(img, M)
        binary_warped = warped_birdview(binary_output, M)

        #     left_lane_x, left_lane_y, right_lane_x, right_lane_y, out_img = find_lane_boundary(binary_warped)
        out_img, left_fit, right_fit, left_fit_x, right_fit_x, ploty = fit_polynomial(binary_warped, nwindows, margin,
                                                                                      minpix)

        blend_img = transform_to_the_road(undistorted_img, Minv, left_fit, right_fit)

        plt.cla()
        # plt.plot(left_fit_x, ploty, color='yellow')
        # plt.plot(right_fit_x, ploty, color='yellow')
        plt.imshow(blend_img)
        plt.savefig(os.path.join(output_detectedline_img, '{}.jpg'.format(img_fn)))
    #   plt.imshow(binary_warped, cmap='gray')
