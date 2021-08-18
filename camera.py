import pickle
import cv2
from cv2 import aruco
import pickle
import time
from threading import Thread
import numpy as np

class Camera:
    """ The class implements the camera. """

    def __init__(self, cam_url=0):
        self.mtx = None
        self.dist = None
        self._calibrated = False
        self.cam_url = cam_url
        self.map = None
        self.video = self.VideoGet(cam_url)

    def get_frame(self):
        """ If new thread is started then wait until frame is available and then return it. """
        if self.video.stopped:
            print('Cannot read frame')
        else:
            while self.video.frame is None:
                continue
            return self.video.frame

    def calibrated(self) -> bool:
        """ The property is equal true if the camera calibrated """
        return self._calibrated

    def camera_url(self):
        return self.cam_url

    def undistort(self, img):
        """ Returns undistorted image. Takes gray scaled image as an input. """
        if self.calibrated():
            return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        else:
            return None


    def show_video(self, detectMarkers=False, showPoseAxis = False, showCharucoPose = False, showMapPose = False, showRobotPose=False):
        """
        Show video.

        Parameters:
            detectMarkers(bool): If true it will detect and draw border around markers
            showPoseAxis(bool): If true it will show pose axis for each detected marker
            showCharucoPose(bool): If treu it will show pose for detected chatuco board
            showMapPose(bool): If true it will show pose for the map
        """
        self.video.start()
        aruco_dictionary = Camera._charuco_dict()
        print("Press q to Quit")
        frame_id = 0
        tick = time.time()
        while True:
            frame = self.get_frame()
            if self.video.stopped:
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
            if showRobotPose:
                corners,ids1,_=aruco.detectMarkers(frame,aruco_dictionary)
                camc,ids=self.cord_rel_to_marker(frame)
                if len(camc)>0:
                    print(camc[0])
                    #frame=cv2.putText(frame,str(camc[0]),corners[0][0],cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
            if showMapPose:
                rvec,tvec = self.map_pose_in_camera_frame(frame,None,None,True)
                if len(rvec)>0:
                    frame = aruco.drawAxis(frame,self.camera_matrix(),self.distortion_coeff(),rvec,tvec,30)
            cv2.imshow(self.camera_url(), frame)
            frame_id = frame_id + 1
            if frame_id%100==0:
                print('fps: '+ str(100/(time.time()-tick)))
                tick = time.time()
            if cv2.waitKey(1) == ord('q'):
                break
        self.video.stop()
        cv2.destroyAllWindows()

    def calibrate_with_charuco(self, frame_rate=50, chess_square_length=1, marker_square_length=0.7, show_markers=True, save=None) -> None:
        if self.calibrated():
            return (self.camera_matrix(), self.distortion_coeff())

        cap = cv2.VideoCapture(self.camera_url())
        _dict = Camera._charuco_dict()
        board = self._create_charuco_board(chess_square_length, marker_square_length)
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

    def load_map(self,filename):
        with open(filename,'rb') as inp:
            data = pickle.load(inp)
            self.map = aruco.GridBoard_create(data[0],data[1],data[2],data[3],Camera._charuco_dict())

    def _create_charuco_board(self, chess_square_length=1, marker_square_length=0.7):
        return cv2.aruco.CharucoBoard_create(7, 5, chess_square_length, marker_square_length, Camera._charuco_dict())

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
        """
        Returns rvec and tvec for detected markers.
        """
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
        """
        Returns rvec and tvec for charuco board.
        """
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


    def map_pose_in_camera_frame(self,frame,corners,ids,useFrame = False):
        """
        Returns rvec and tvec for map.
        """
        if not self.map:
            return [],[]
        if useFrame:
            corners,ids,_ = aruco.detectMarkers(frame,Camera._charuco_dict())
        
        if ids is not None and len(ids)>0:
            ret,rvec,tvec = aruco.estimatePoseBoard(corners,ids,self.map,self.camera_matrix(),self.distortion_coeff(),None,None)
            if ret>0:
                return rvec,tvec
        return [],[]
    
    def cord_rel_to_marker(self, frame, markerlength=0.7):
        corners, ids, _ = aruco.detectMarkers(frame, Camera._charuco_dict())
        r, t = self.map_pose_in_camera_frame(frame, corners, ids)
        if (r, t) == ([], []):
            return [], []
        rvec, tvec, ids = self.marker_pose_in_camera_frame(frame, markerlength)
        rvec2 = []
        tvec2 = []
        ids2 = []
        n = len(ids)
        for i in range(n):
            if ids[i] > 200:
                rvec2.append(rvec[i])
                tvec2.append(tvec[i])
                ids2.append((ids[i]))
        rmat1, _ = cv2.Rodrigues(r)
        camc1 = np.negative(np.matmul(np.linalg.inv(rmat1), t))
        camc2 = []
        n = len(rvec2)
        #print(tvec2)
        #print(np.shape(tvec2))
        for i in range(n):
            rmat2, _ = cv2.Rodrigues(rvec2[i])
            camc2.append(np.negative(np.matmul(np.linalg.inv(rmat2), np.transpose(tvec2[i]))))
        n = len(camc2)
        for i in range(n):
            camc2[i] = np.subtract(camc2[i], camc1)
        return camc2, ids2



    class VideoGet:
        """
        Creates a separate thread for reading video frames.
        Read self.frame to get the video frame
        """

        def __init__(self,src = 0):
            self.grabbed = False
            self.frame = None
            self.stopped = True
            self.thread = Thread(target=self.read_frame,args=())
            self.video_stream = None
            self.src = src
        
        def start(self):
            """
            Starts the new thread.
            """
            if not self.stopped:
                return
            self.video_stream = cv2.VideoCapture(self.src)
            if self.thread is None:
                self.thread = Thread(target=self.read_frame,args=())
            self.stopped = False
            self.thread.start()
        
        def read_frame(self):
            while not self.stopped:
                self.grabbed,self.frame = self.video_stream.read()
                if not self.grabbed:
                    self.stop()
                
        def stop(self):
            """
            Stops the thread.
            """
            self.stopped = True
            self.thread.join()
            self.thread = None
            self.video_stream.release()
            self.video_stream = None
