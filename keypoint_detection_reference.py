"""
Keypoint detection reference:
when running this as is without changing parameters, here's what you would expect:
This will initialize the camera and openpose wrapper.
Then will loop and every few frames will try to find the pose and hands of a single person.
When it finds keypoints, it will mark them (blue/red for right/left hands and green for entire body) and will
highlight the face (light blue) and arms (purple).
Will exit when ESC is pressed.

Note - change the openpose import section so that it will work for you, rest should work as is.
Extra requirements:
    cv2, numpy
"""

import os,sys
import time
import cv2
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))
OPENPOSE_DIR = '/home/david/openpose/'
MODEL_DIR = os.path.join(OPENPOSE_DIR,'models/')
try:
    sys.path.append(os.path.join(OPENPOSE_DIR,'build/python/'))
    from openpose import pyopenpose as op
except ImportError as e:
    print(
        'Error: OpenPose library could not be found in {}. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?'.format(os.path.join(OPENPOSE_DIR,'build/python/')))
    raise e

ESC_BUTTON_KEY_CODE = 27
CV2_WINDOW_TITLE = "Openpose keypoint detector"
BGR_GREEN = (0, 255, 0)
BGR_RED = (0, 0, 255)
BGR_BLUE = (255, 0, 0)
BGR_LIGHT_BLUE = (255, 255, 0)
BGR_PURPLE = (128, 0, 128)
HAND_DETECTOR_MODEL = 2
FACE_KEYPOINT_PART_NAMES = ["Nose", "LEye", "REye", "LEar", "REar"]
ARMS_KEYPOINT_PART_NAMES = ["RShoulder", "LShoulder","RElbow","LElbow","LWrist","RWrist"]
POSE_INDICES = {"Nose": 0,
                "Neck": 1,
                "RShoulder": 2,
                "RElbow": 3,
                "RWrist": 4,
                "LShoulder": 5,
                "LElbow": 6,
                "LWrist": 7,
                "MidHip": 8,
                "RHip": 9,
                "RKnee": 10,
                "RAnkle": 11,
                "LHip": 12,
                "LKnee": 13,
                "LAnkle": 14,
                "REye": 15,
                "LEye": 16,
                "REar": 17,
                "LEar": 18,
                "LBigToe": 19,
                "LSmallToe": 20,
                "LHeel": 21,
                "RBigToe": 22,
                "RSmallToe": 23,
                "RHeel": 24}


def get_configured_opWrapper(hand=True,face=False,body=1, number_people_max=1,frame_step=3,render_threshold=0.5,model_folder=MODEL_DIR):
    """
    model_folder should basically point to openpose/models/
    Face takes an enormous amount of VRAM, this can make openpose crash due to insufficient VRAM, so face defaults
    to False.
    """
    params = dict()
    params["model_folder"] = model_folder
    params["hand"] = hand
    params["hand_detector"] = 3
    if hand and body != 1:
        # since body detection is off, we cannot use it's model (the default) so we use hand detection model
        params["hand_detector"] = HAND_DETECTOR_MODEL
    params["body"] = body
    params["face"] = face
    params["number_people_max"] = number_people_max
    params["frame_step"] = frame_step
    params["render_threshold"] = render_threshold
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    return opWrapper

def draw_square_around_pixel(image, pixel, dist, border_width=1, color=BGR_GREEN):
    x,y = pixel
    top_left = (x-dist,y-dist)
    bot_right = (x+dist,y+dist)
    cv2.rectangle(image,top_left,bot_right,color,border_width)

def mark_keypoint_array(frame, keypoint_coords, dist=1, width=1, color=BGR_GREEN):
    """
    Draws squares around all given keypoint coordinates.
    """
    for x,y in keypoint_coords:
        draw_square_around_pixel(frame, (int(x),int(y)), dist=dist, border_width=width, color = color)
    return keypoint_coords

def get_pose_keypoint_coord_dict(pose_keypoints,requested_parts=None):
    """
    Will return whatever is in the correct index (expects keypoints array or coordinates array).
    """
    if requested_parts is None: # return all of the coords
        requested_parts = list(POSE_INDICES.keys())
    out_dict = dict()
    for part in requested_parts:
        # update the out_dict with the wanted pose keypoint
        out_dict[part] = pose_keypoints[POSE_INDICES[part]]
    return out_dict


def draw_specific_pose_keypoints(frame, pose_dict, body_part_list, color=BGR_RED):
    specific_coords = []
    for part in body_part_list:
        draw_square_around_pixel(frame, pose_dict[part], dist=1, border_width=5, color=color)
        specific_coords.append(pose_dict[part])

def detect_keypoints_from_camera(camera, visualize_detection = True, draw_openpose_render = False, detect_hands=True, detect_face=False, detect_pose=True, seconds_between_frames=0.01, frames_between_detection=4, print_keypoint_traces=True, verbose=False):
    """
    :param camera: the cv2 VideoCapture object that represents our camera
    :param visualize_detection: if True - will enable rendering and showing of frames and detections
    :param draw_openpose_render: if this is True, the render from openpose will be drawn (can be turned off with False)
    :param detect_hands: True to detect, False to ignore hands
    :param detect_face: same as detect_hands
    :param detect_pose: same as detect_hands
    :param seconds_between_frames: self explanatory - lower numbers = higher load but higher numbers = higher latency (lower FPS as well)
    :param frames_between_detection: attempt detection on every n'th frame - lower numbers = higher load but higher numbers = higher latency
    :param print_keypoint_traces: print last known coordinates of all keypoints between frames
    :param verbose: print data to the console
    """
    if camera is None:
        raise Exception("Camera is None.. are you kidding me??")
    print("Initializing...")
    cv2.namedWindow(CV2_WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

    opWrapper = get_configured_opWrapper(hand=detect_hands, face=detect_face, body=int(detect_pose), number_people_max=1, frame_step=15)
    opWrapper.start()
    datum = op.Datum()
    finished = False
    current_frame_num = 0
    print("Done\nPress ESC to quit.")
    while not finished:  # run until user presses ESC
        # frame timing parameters
        current_frame_num = current_frame_num%frames_between_detection
        time.sleep(seconds_between_frames)
        # read a frame from the camera
        _, frame = camera.read()
        # refresh the datum object and detect keypoints
        if current_frame_num == 0:
            datum.cvInputData = frame
            opWrapper.emplaceAndPop([datum])
            if visualize_detection and draw_openpose_render:
                frame = datum.cvOutputData
        """
        Extract the coordinates of all keypoints - you can use:
            left_coords = left detected hand coordinates
            right_coords = right detected hand coordinates
            pose_dict = detected pose coordinates (entire body dict, key:value -> part:coords)
        """
        if detect_hands:
            left,right = np.array(datum.handKeypoints)
            left_coords,right_coords = left[0,:,:2],right[0,:,:2]
            if visualize_detection and print_keypoint_traces:
                # draw both hands
                mark_keypoint_array(frame, left_coords, color=BGR_RED)
                mark_keypoint_array(frame, right_coords, color=BGR_BLUE)
        if detect_pose:
            pose_coords = np.array(datum.poseKeypoints[0,:,:2]).astype("int")
            # create usable dictionary of keypoint coordinates by body part
            pose_dict = get_pose_keypoint_coord_dict(pose_coords, None)
            if visualize_detection and print_keypoint_traces:
                # draw entire keypoint array
                mark_keypoint_array(frame, pose_coords, width=2, color=BGR_GREEN)
                # highlight specific keypoints
                draw_specific_pose_keypoints(frame, pose_dict, FACE_KEYPOINT_PART_NAMES, color=BGR_LIGHT_BLUE)
                draw_specific_pose_keypoints(frame, pose_dict, ARMS_KEYPOINT_PART_NAMES, color=BGR_PURPLE)
        cv2.imshow(CV2_WINDOW_TITLE, frame)
        current_frame_num += 1
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ESC_BUTTON_KEY_CODE:  # escape button means exit and finish the run
            print("User pressed the exit button (ESC), exiting...")
            exit()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    camera = cv2.VideoCapture(0)
    detect_keypoints_from_camera(camera)
    camera.release()
