import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands

device = 0 # カメラのデバイス番号

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

    # 指定したインデックスに点を打つ
    for i in range(len(landmark_point)):
        #サークルを描画
        cv2.circle(image, (landmark_point[i][0], landmark_point[i][1]), 7, (0, 0, 255), -1)
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

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    drawFingertip(frame, hand_landmarks)
            cv2.imshow('MediaPipe Hands', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

if __name__ == '__main__':
    main()
