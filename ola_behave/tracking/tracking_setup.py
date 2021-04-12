# This script will allow you to set up a fish tracking on a recording
# First you will be prompted to select the folder for the recording
# Next you will be prompted to run the median script
# then the define_ROI script
# Then you can press run

import datetime
import os
import glob
import time

import cv2
from tkinter.filedialog import askdirectory
from tkinter import *

from ola_behave.io.yaml import load_yaml
from ola_behave.tracking.tracking_functions import define_roi_still, background_vid, threshold_select
from ola_behave.tracking.offline_tracker import tracker


date = datetime.datetime.now().strftime("%Y%m%d")

# Allows a user to select top directory
root = Tk()
root.withdraw()
root.update()
videodir = askdirectory()
root.destroy()

rootdir = os.path.dirname(videodir)
recording_roi = videodir[-1]
print("Will track all movies in {0}, which corresponds to the videos of roi {1}".format(videodir, recording_roi))

videofiles = []
for file in os.listdir(videodir):
    if not file.startswith('.'):
        if file.endswith(".mp4"):
            videofiles.append(os.path.join(videodir, file))
        elif file.endswith(".avi"):
            print('Didn\'t find mp4 file so looking for avi files')
            if file.endswith(".avi"):
                videofiles.append(os.path.join(videodir, file))

# load yaml config file
params = load_yaml(rootdir, "config")

# find and load background file
if len(glob.glob(os.path.join(videodir, "*background*"))) == 1:
    # case 1: video background has been defined previously, load and then define tracking video rois
    print("using background in video folder")
    background_video = cv2.imread(glob.glob(os.path.join(videodir, "*background*"))[0], 0)
    if len(glob.glob(os.path.join(videodir, "*roi*"))) != 1:
        # Define video rois
        define_roi_still(background_video, videodir)
else:
    if len(glob.glob(os.path.join(rootdir, "*background*"))) != 1:
        # case 2: make video background from video, load and then define tracking video rois
        print('No background file in folder:' + rootdir)
        # make_M = input("Make background from video? y/n: ")
        make_M = "y"
        print("Making background from videos")
        if make_M == "y":
            nth_frame = 1000
            print("making background from every {} nth frame".format(nth_frame))
            background_vid(videofiles[0], nth_frame, 50)
            background_video = cv2.imread(glob.glob(os.path.join(videodir, "*background*"))[0], 0)
            # Define video rois
            define_roi_still(background_video, videodir)
        else:
            sys.exit()
    else:
        # case 3: make video background from full background taken at recording, load and then define tracking video rois
        # In this case, need to crop the recording background to the size of the video before defining the video rois

        # find and load roi file
        if len(glob.glob(os.path.join(rootdir, "roi_file*"))) == 1:
            print("Using previously define ROIs for these videos")
            rois = load_yaml(rootdir, "roi_file")
        else:
            print("No roi file for all recordings, so can't trim background correctly")
            sys.exit()

        background_full = cv2.imread(glob.glob(os.path.join(rootdir, "*background*"))[0], 0)
        # Create roi specific cropped backgrounds in list
        curr_roi = rois["roi_{}".format(recording_roi)]
        width_trim = 1280
        height_trim = 960
        background_video = background_full.reshape((height_trim, width_trim))[curr_roi[1]:curr_roi[1] + curr_roi[3],
                          curr_roi[0]:curr_roi[0] + curr_roi[2]]
        # Define video rois
        define_roi_still(background_video, videodir)

# load video rois
rois = load_yaml(videodir, "roi_file")

# find and check threshold
threshold = threshold_select(videofiles[0], background_video, rois)

# Run tracking script
for video_path in videofiles:
    precall = time.perf_counter()
    tracker(video_path, background_video, rois, threshold, display=False, area_size=0)
    print("Movie {} took {:.2f} seconds to track".format(video_path, time.perf_counter() - precall))
