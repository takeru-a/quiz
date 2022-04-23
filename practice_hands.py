import cv2
import mediapipe as mp
import time
import numpy as np


mp_hands = mp.solutions.hands

device = 0 # カメラのデバイス番号

#カメラ画像とイラストの合成
def combi(img):
    #ワニの写真を変数に格納
    wimg  = cv2.imread('./imgs/four_choice/hackathon_quiz_a1.jpg')
    #サイズの変更
    wimg = cv2.resize(wimg, dsize=None, fx=0.8, fy=0.8)

    white = np.ones((img.shape), dtype=np.uint8) * 255 #カメラ画像と同じサイズの白画像

    wimg2 = wimg
    wimg3 = wimg
    wimg4 = wimg

    #x始点
    x = 0
    #y終点
    y = img.shape[0]-wimg.shape[0]
    #x終点
    xd = wimg.shape[1]

    #2匹のワニの位置の更新
    # for i in range(3):
    #     cnt[i] = cnt[i]%(img.shape[1]-wimg.shape[1])

    #white[y_start:y_end,x_start:x_end] = wimg
    #y_end - y_start == wimg.shape[0]
    #x_end - x_start == wimg.shape[1]
    white[y:img.shape[0],0:xd] = wimg
    white[0:wimg2.shape[0],0:xd] = wimg2
    white[0:wimg3.shape[0],400:xd+400] = wimg3
    white[y:img.shape[0],400:xd+400] = wimg4


    #カメラ画像にワニの画像を貼り付ける
    dwhite = white

    #ワニがある部分(位置)をカメラ画像から切り抜き、ワニの画像を貼り付ける
    img[dwhite!=[255, 255, 255]] = dwhite[dwhite!=[255, 255, 255]]
    return img

def getFrameNumber(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * 1000 / fps)

    return frame_now

#指のランドマークを表示
def drawFingertip(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        # 画面上の位置に変換
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y, landmark_z])

    cv2.circle(image, (landmark_point[8][0], landmark_point[8][1]), 7, (0, 0, 255), -1)
    # 指定したインデックスに点を打つ
    # for i in range(len(landmark_point)):
        #サークルを描画
        # cv2.circle(image, (landmark_point[i][0], landmark_point[i][1]), 7, (0, 0, 255), -1)


    # if landmark_point[8][0] > 248 and  landmark_point[8][0] < image_width - 1 :
    #             cv2.putText(img, "Great!", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)

def main():
    # For webcam input:
    global device

    cap = cv2.VideoCapture(device)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print("Size:", ht, "x", wt, "/Fps: ", fps)

    start = time.perf_counter()
    frame_prv = -1

    cv2.namedWindow('MediaPipe Hands', cv2.WINDOW_NORMAL)
    with mp_hands.Hands(
        #検出する手の数(1~2)
        max_num_hands = 1,
        #信用度の設定(0~1)
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            frame_now=getFrameNumber(start, fps)
            if frame_now == frame_prv:
                continue
            frame_prv = frame_now

            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue


            #反転処理
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = hands.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            combi(frame)
            cv2.putText(frame, "True!", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
            cv2.putText(frame, "False!", (200,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,20,234), 2)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    drawFingertip(frame, hand_landmarks)
            cv2.imshow('MediaPipe Hands', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

if __name__ == '__main__':
    main()
