# Functions which are used for fish tracking
import os
import datetime

import yaml
import numpy as np
import cv2

from ola_behave.io.yaml import load_yaml


def roi_input():
    """input function for 1. asking how many ROIs"""
    while True:
        roi_nums = input("How many ROIs would you like to select?: ")
        try:
            rois = int(roi_nums)
            print("Will do", roi_nums, "region/s of interest")
            return rois
        except ValueError:
            print("Input must be an integer")


def define_roi_still(image_input, folder_path):
    roi_num = roi_input()
    image = np.array(image_input, copy=True)
    # rr = np.arange(4 * roi_num).reshape(roi_num, 4)
    scalingF = 1
    dict_file = {"cam_ID": "na"}
    height, width = image.shape
    if roi_num == 0:
        dict_file["roi_0"] = tuple([0, 0, width, height])
    else:
        for roi in range(roi_num):
            frameRS = cv2.resize(image, (int(width / scalingF), int(height / scalingF)))
            rr = cv2.selectROI(("Select ROI" + str(roi)), frameRS)
            # output: (x,y,w,h)
            dict_file["roi_" + str(roi)] = tuple(i * scalingF for i in rr)
            # add in ROIs
            start_point = (rr[0], rr[1])
            end_point = (rr[0] + rr[2], rr[1] + rr[3])
            cv2.rectangle(image, start_point, end_point, 220, 2)
            cv2.destroyAllWindows()
        print(dict_file)

    with open(os.path.join(folder_path, "roi_file.yaml"), "w") as file:
        documents = yaml.dump(dict_file, file)

    print("File has now been saved in specified folder as roi_file.yaml")


def background_vid(videofilepath, nth_frame, percentile):
    """ (str, int, int, int)
     This function will create a median image of the defined area"""
    try:
        cap = cv2.VideoCapture(videofilepath)
    except:
        print("problem reading video file, check path")
        return

    counter = 0
    gatheredFramess = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break
        if counter % nth_frame == 0:
            print("Frame {}".format(counter))
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # settings with Blur settings for the video loop
            gatheredFramess.append(image)
        counter += 1

    background = np.percentile(gatheredFramess, int(percentile), axis=0).astype(dtype=np.uint8)
    cv2.imshow('Calculated Background from {} percentile'.format(percentile), background)

    # background = np.percentile(frameMedian, 90, axis=0).astype(dtype=np.uint8)
    # cv2.imshow('Calculated background', background)

    date = datetime.datetime.now().strftime("%Y%m%d")
    vid_name = videofilepath[0:-4]
    # vid_name = os.path.split(videofilepath)[1][0:-4]
    # vid_folder_path = os.path.split(videofilepath)[0]
    cv2.imwrite('{}_per{}_background.png'.format(vid_name, percentile), background)

    cap.release()
    cv2.destroyAllWindows()
    return background


def print_roi(roi_path, video_path):
    rois = load_yaml(roi_path, "roi_file")

    try:
        cap = cv2.VideoCapture(video_path)
    except:
        print("problem reading video file, check path")
        return False

    while (cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break

        for roi in range(0, len(rois) - 1):
            # for the frame define an ROI and crop image
            curr_roi = rois["roi_" + str(roi)]
            # add in ROIs
            start_point = (curr_roi[0], curr_roi[1])
            end_point = (curr_roi[0] + curr_roi[2], curr_roi[1] + curr_roi[3])
            cv2.rectangle(frame, start_point, end_point, 220, 2)
            cv2.imshow('Roi printed on video frame', frame)
        return frame


def threshold_select(video_path, median_full, rois):
    """ Function that takes a video path, a median file, and rois. It then uses background subtraction and centroid
    tracking to find the XZ coordinates of the largest contour. This script has a threshold bar which allows you to try
    different levels. Once desired threshold level is found. Press 'q' to quit and the selected value will be used """
    t_pos = 35
    frame_id = 0

    # create trackbar for setting threshold
    def nothing(x):
        pass

    # create window
    cv2.namedWindow("Live thresholded")

    # create threshold trackbar
    cv2.createTrackbar('threshold', "Live thresholded", t_pos, 255, nothing)

    # load video
    video = cv2.VideoCapture(video_path)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print("reached end of video")
            video.release()
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frameDelta_full = cv2.absdiff(median_full, gray)

        # tracking
        cx = list()
        cy = list()
        contourOI = list()
        contourOI_ = list()
        t_pos = cv2.getTrackbarPos('threshold', "Live thresholded")
        for roi in range(0, len(rois) - 1):
            # for the frame define an ROI and crop image
            curr_roi = rois["roi_" + str(roi)]
            frameDelta = frameDelta_full[curr_roi[1]:curr_roi[1] + curr_roi[3], curr_roi[0]:curr_roi[0] + curr_roi[2]]
            image_thresholded = cv2.threshold(frameDelta, t_pos, 255, cv2.THRESH_TOZERO)[1]
            (contours, _) = cv2.findContours(image_thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                contourOI_.append(max(contours, key=cv2.contourArea))
                area = cv2.contourArea(contourOI_[roi])
                if area > 0:
                    contourOI.append(cv2.convexHull(contourOI_[roi]))
                    M = cv2.moments(contourOI[roi])
                    cx.append(int(M["m10"] / M["m00"]))
                    cy.append(int(M["m01"] / M["m00"]))
                else:
                    print("no large enough contour found for roi {}!".format(roi))
                    contourOI_[-1] = False
                    contourOI.append(False)
                    cx.append(np.nan)
                    cy.append(np.nan)
            else:
                print("no contour found for roi {}!".format(roi))
                contourOI_.append(False)
                contourOI.append(False)
                cx.append(np.nan)
                cy.append(np.nan)

        full_image_thresholded = (cv2.threshold(frameDelta_full, t_pos, 255, cv2.THRESH_TOZERO)[1])
        # Live display of full resolution and ROIs
        cv2.putText(full_image_thresholded, "Framenum: {}".format(frame_id), (30, full_image_thresholded.shape[0] -
                                                                              30),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=255)
        cv2.putText(full_image_thresholded, "Choose the threshold, press 'q' once happy, threshold: {}".format(t_pos),
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 200), 1)

        for roi in range(0, len(rois) - 1):
            if np.all(contourOI_[roi] != False):
                curr_roi = rois["roi_" + str(roi)]
                # add in contours
                corrected_contour = np.empty(contourOI_[roi].shape)
                corrected_contour[:, 0, 0] = contourOI_[roi][:, 0, 0] + curr_roi[0]
                corrected_contour[:, 0, 1] = contourOI_[roi][:, 0, 1] + curr_roi[1]
                cv2.drawContours(full_image_thresholded, corrected_contour.astype(int), -1, 255, 1)

                # add in centroid
                cv2.circle(full_image_thresholded, (cx[roi] + curr_roi[0], cy[roi] + curr_roi[1]), 8, 255, 1)

            cv2.imshow("Live thresholded", full_image_thresholded)
            cv2.imshow("Live", gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    print("Tracking finished on video cleaning up")
    cv2.destroyAllWindows()
    return t_pos
