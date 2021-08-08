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

    def show_video(self, detectMarkers=False, showPoseAxis = False, showCharucoPose = False):
        cap = cv2.VideoCapture(self.camera_url())
        aruco_dictionary = Camera._charuco_dict()
        print("Press q to Quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print('Cannot read from: ' + self.camera_url())
                break
            if detectMarkers:
                corners, ids, _ = aruco.detectMarkers(frame, aruco_dictionary)
                frame = aruco.drawDetectedMarkers(
                    frame, corners, ids, (0, 255, 0))
            if showPoseAxis and self.calibrated():
                rvecs,tvecs,_ = self.marker_pose_in_camera_frame(frame,1)
                for rvec,tvec in zip(rvecs,tvecs):
                    frame = aruco.drawAxis(frame,self.camera_matrix(),self.distortion_coeff(),rvec,tvec,1)
            if showCharucoPose:
                rvecs,tvecs = self.charuco_pose_in_camera_frame(frame)
                if(len(rvecs)!=0):
                    frame = aruco.drawAxis(frame,self.camera_matrix(),self.distortion_coeff(),rvecs,tvecs,3)
            cv2.imshow(self.camera_url(), frame)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def calibrate_with_charuco(self, frame_rate=50, chess_square_length=1, marker_square_length=0.7, show_markers=True, save=None) -> None:
        if self.calibrated():
            return (self.camera_matrix(), self.distortion_coeff())

        cap = cv2.VideoCapture(self.camera_url())
        _dict = Camera._charuco_dict()
        board = self._create_charuco_board(
            chess_square_length, marker_square_length)
        all_corners = []
        all_ids = []

        for frame_id in range(1500):
            ret, frame = cap.read()
            if not ret:
                print('Cannot read from: '+self.camera_url())
                break

            frame_id += 1
            corners, ids, _ = cv2.aruco.detectMarkers(frame, _dict)
            if show_markers:
                aruco.drawDetectedMarkers(frame, corners, ids)

            if frame_id % frame_rate != 0:
                continue

            if corners:
                res, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, frame, board)
                if res > 4:
                    all_corners.append(charuco_corners)
                    all_ids.append(charuco_ids)
                aruco.drawDetectedCornersCharuco(
                    frame, charuco_corners, charuco_ids, (0, 0, 255))
                
            cv2.imshow(str(self.camera_url()), frame)
            cv2.waitKey(1)

        try:
            projectionError, self.mtx, self.dist, _, _ = cv2.aruco.calibrateCameraCharuco(
            all_corners, all_ids, board, frame.shape[0:2], None, None, criteria = (cv2.TERM_CRITERIA_MAX_ITER & cv2.TERM_CRITERIA_EPS,50000,1e-9))
            print(projectionError)
        except Exception as e:
            print(e)
            return None,None
        finally:
            cap.release()
            cv2.destroyAllWindows()
        self._calibrated = True
        if save != None and self.calibrated():
            self._save(str(save))
        return (self.mtx, self.dist)

    def _save(self, filename):
        data = {
            "mtx": self.mtx,
            "dist": self.dist,
        }
        pickle.dump(data, open(filename, "wb"))

    def _load(self, filename):
        self._calibrated = True
        data = pickle.load(open(filename, "rb"))
        self.mtx = data["mtx"]
        self.dist = data["dist"]

    def _create_charuco_board(self, chess_square_length=1, marker_square_length=0.7):
        return cv2.aruco.CharucoBoard_create(5, 5, chess_square_length, marker_square_length, Camera._charuco_dict())

    def save_charuco_board_to_file(self, filename):
        board = self._create_charuco_board()
        img = board.draw((1920, 1080))
        cv2.imwrite(filename, img)

    def _charuco_dict():
        return cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)

    def camera_matrix(self):
        if self._calibrated == False:
            return None
        return self.mtx

    def distortion_coeff(self):
        if self._calibrated == False:
            return None
        return self.dist

    def marker_pose_in_camera_frame(self, frame, markerlength = 0.7):
        if not self.calibrated():
            return None
        else:
            aruco_dictionary = Camera._charuco_dict()
            para = aruco.DetectorParameters_create()
            para.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
            corners, ids, _ = aruco.detectMarkers(frame, aruco_dictionary,parameters = para)
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, markerlength, self.camera_matrix(), self.distortion_coeff())
            if rvec is None:
                return [], [], []
            return rvec, tvec, ids

    def charuco_pose_in_camera_frame(self,frame):
        if not self.calibrated():
            return None
        else:
            aruco_dictionary = Camera._charuco_dict()
            board = self._create_charuco_board(1,0.7)
            corners, ids, _ = aruco.detectMarkers(frame,aruco_dictionary)
            if(len(corners)==0):
                return [],[]
            _, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners,ids,frame,board)
            _,rvec,tvec = aruco.estimatePoseCharucoBoard(charucoCorners,charucoIds,board,self.camera_matrix(),self.distortion_coeff(),None,None)
            if rvec is None:
                return [],[]
            return rvec,tvec

    def cameratoglobal(self, frame, markerlength):
        if not self.calibrated():
            return None
        rvecs, tvecs, ids = self.marker_pose_in_camera_frame(frame, markerlength)
        rmat, _ = cv2.Rodrigues(rvecs)
        half = markerlength/2
        lengthmat = np.array([half, half, 0])
        camc = np.add(np.matmul(rmat, lengthmat), tvecs)
        n = len(ids)
        j = 0
        k = 0
        for i in range(n):
            if ids[i] < 50:
                j = i
                k = ids[i]
                break
        globalmatrix = matdict[k]
        ref = camc[j]
        for i in range(n):
            camc[i] = np.matmul(np.subtract(camc[i], ref), globalmatrix)
        return camc, ids
