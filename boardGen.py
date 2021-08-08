import cv2
import json


def main():
    round = int(input('Round?'))
    if round == 1:
        boardDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
        board = cv2.aruco.GridBoard_create(20, 10, 1, 1/15.2400, boardDict)
        img = board.draw((4096, 4096), marginSize=20)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite('board/round1/round_1_board.png', img)
        corners, ids, _ = cv2.aruco.detectMarkers(img, boardDict)
        print(corners[0][0][0][0])
        print(ids[0])
        img = cv2.aruco.drawDetectedMarkers(img, corners, ids, (255, 255, 0))
        cv2.imwrite('board/round1/round_1_board_with_id.png', img)
        for id in ids:
            img = cv2.aruco.drawMarker(boardDict,id[0],720)
            cv2.imwrite('board/round1/individualmarkers/id'+str(id[0])+'.png',img)
        with open('board/round1/round1board.json','w') as outp:
            d = {'markersX':20, 'markersY':10, 'markerLenght':15.24000, 'spacing':1}
            json.dump(d,outp)



if __name__ == '__main__':
    main()
