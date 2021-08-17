import camera
if __name__=='__main__':
    c=camera.Camera('http://192.168.190.133:8080/video')
    c._load('calib2')
    c.load_map('./board/round1/round1board.pkl')
    c.show_video(helper=True)