import pickle
import cv2
from cv2 import aruco
import numpy as np
from pathlib import Path
from typing import List, Tuple, Any
import os
import glob


class Camera:
    """ The class implements the camera. """

    def __init__(self, cam_url=0):
        self.mtx = None
        self.dist = None
        self._calibrated = False
        self.cam_url = cam_url

    def calibrated(self) -> bool:
        """ The property is equal true if the camera calibrated """
        return self._calibrated

    def camera_url(self):
        return self.cam_url

    def undistort(self, img: Any):
        """ Returns undistorted image. Takes gray scaled image as an input. """
        if self.calibrated():
            return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        else:
            return None


    # def detect_charuco_corners(self, video_in: str, video_out: str) -> None:
    #     """ Detect charuco corners on video and overlay an additional information """
    #     cap = cv2.VideoCapture(video_in)
    #     board = Camera._create_charuco_board()
    #     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     output = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         corners, ids, rejected = aruco.detectMarkers(gray, Camera._charuco_dict())
    #         cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejected)
    #         out = frame.copy()
    #         if corners:
    #             cv2.aruco.drawDetectedMarkers(out, corners, ids, borderColor=(0, 255, 0))
    #             res, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board,
    #                                                                                    self.mtx, self.dist)
    #             if ret:
    #                 cv2.aruco.drawDetectedCornersCharuco(out, charuco_corners, charuco_ids, cornerColor=(0, 0, 255))
    #             rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, .02, self.mtx, self.dist)
    #             for i in range(len(tvecs)):
    #                 cv2.aruco.drawAxis(out, self.mtx, self.dist, rvecs[i], tvecs[i], 0.03)
    #         output.write(out)

    #     cap.release()
    #     output.release()

    def show_video(self,detectMarkers = False):
        cap = cv2.VideoCapture(self.camera_url())
        aruco_dictionary = Camera._charuco_dict()
        print("Press q to Quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print('Cannot read from: '+ self.camera_url())
                break
            if detectMarkers:
                corners, ids, _ = aruco.detectMarkers(frame,aruco_dictionary)
                frame = aruco.drawDetectedMarkers(frame,corners,ids,(0,255,0))
            cv2.imshow(self.camera_url(),frame)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def calibrate_with_charuco(self, frame_rate=10, chess_square_length = 0.04, marker_square_length = 0.03, show_markers = True, save = None) -> None:
        if self.calibrated():
            return (self.camera_matrix(),self.distortion_coeff())
        
        cap = cv2.VideoCapture(self.camera_url())
        _dict = Camera._charuco_dict()
        board = self._create_charuco_board(chess_square_length,marker_square_length)
        all_corners = []
        all_ids = []

        for frame_id in range(500):
            ret, frame = cap.read()
            if not ret:
                print('Cannot read from: '+self.camera_url())
                break

            frame_id += 1
            corners, ids, _ = cv2.aruco.detectMarkers(frame, _dict)
            if show_markers:
                aruco.drawDetectedMarkers(frame,corners,ids)
            
            if frame_id % frame_rate != 0:
                continue

            
            if corners:
                res, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, frame, board)
                if res > 4:
                    all_corners.append(charuco_corners)
                    all_ids.append(charuco_ids)
                aruco.drawDetectedCornersCharuco(frame,charuco_corners,charuco_ids,(0,0,255))
                cv2.imshow(str(self.camera_url()), frame)
            
            cv2.waitKey(1)
            

        _, self.mtx, self.dist, _, _ = cv2.aruco.calibrateCameraCharuco(all_corners, all_ids, board, frame.shape[0:2], None, None)
        cap.release()
        self._calibrated = True
        if save!=None:
            self._save(str(save))
        cv2.destroyAllWindows()
        return (self.mtx,self.dist)

    def _save(self, filename):
        data = {
            "mtx": self.mtx,
            "dist": self.dist,
        }
        pickle.dump(data, open(filename, "wb"))

    def _load(self,filename):
        self._calibrated=True
        data = pickle.load(open(filename, "rb"))
        self.mtx = data["mtx"]
        self.dist = data["dist"]

    def _create_charuco_board(self,chess_square_length = 0.04, marker_square_length = 0.03):
        return cv2.aruco.CharucoBoard_create(5, 5, chess_square_length, marker_square_length, Camera._charuco_dict())
    
    def save_charuco_board_to_file(self,filename):
        board = self._create_charuco_board()
        img = board.draw((1920,1080))
        cv2.imwrite(filename,img)

    def _charuco_dict():
        return cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    def camera_matrix(self):
        if self._calibrated==False:
            return None
        return self.mtx
    
    def distortion_coeff(self):
        if self._calibrated==False:
            return None
        return self.dist

    def markerpose_cameraframe(self,frame,markerlength):
        if not self.calibrated():
            return None
        else:
            print("yes")
            aruco_dictionary = Camera._charuco_dict()
            corners, ids, _ = aruco.detectMarkers(frame,aruco_dictionary)
            rvec,tvec,_= cv2.aruco.estimatePoseSingleMarkers(corners,markerlength,self.camera_matrix(),self.distortion_coeff())
            if rvec is None:
                print("yes")
                return [],[]
            return rvec,tvec

    # def test(self):
    #     cap=cv2.VideoCapture(self.camera_url())
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             print('Cannot read from: '+ self.camera_url())
    #             break
    #         rvecs,tvecs=self.markerpose_cameraframe(frame,0.05)
    #         for i in range(len(rvecs)):
    #             frame=aruco.drawAxis(frame,self.camera_matrix(),self.distortion_coeff(),rvecs[i],tvecs[i],0.05)  
    #         cv2.imshow(self.camera_url(),frame)
    #         if cv2.waitKey(1) == ord('q'):
    #             break
    #     cap.release()
    #     cv2.destroyAllWindows()

        
        


        
        





