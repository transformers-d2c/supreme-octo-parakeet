import camera as cam
#import numpy as np

url_input = str(input("Enter URL of the Camera (0 for default): ")) # takes the url as input, can be empty if wants the default '0' as camera
global camera_url;
global main_camera;  # this will be calibrated

if url_input=="0" or url_input=="":
    camera_url = 0
else:
    camera_url = url_input

main_camera = cam.Camera(camera_url);

chess_sq_length = float(input("Enter the Chess Square Length (must be a float value): "))  # chess square length
marker_sq_length = float(input("Enter the Marker Square Length (must be a float value): ")) # marker length
saving_url = str(input("Location of place to save the calibrated file: ")) # place to save the caloibration file

main_camera.calibrate_with_charuco(50, chess_sq_length, marker_sq_length, True, saving_url);

'''
camera has been calibrated now and the file has been saved in the required url
'''

cam_matrix = main_camera.camera_matrix()
dis_coeff = main_camera.distortion_coeff()

# added the above two as they can be called if required or can be removed as per requirement.... 

