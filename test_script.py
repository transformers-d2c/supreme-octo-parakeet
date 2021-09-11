from numpy.linalg.linalg import det
import camera
if __name__=='__main__':
    c=camera.Camera('http://192.168.0.103:8080/video')
    c._load('calibre')
    c.load_map('board/round1/round1board.pkl')
    c.show_video(detectMarkers=True)
    # c.show_video(detectMarkers=True,showPoseAxis=True)