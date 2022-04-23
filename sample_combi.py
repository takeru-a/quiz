# Sample of video-image processing adapted the frame rate
# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np

device = 0 

#カメラ画像とイラストの合成
def combi(img, cnt):
    #ワニの写真を変数に格納
    wimg  = cv2.imread('./imgs/wani.jpg')
    #サイズの変更
    wimg = cv2.resize(wimg, dsize=None, fx=0.2, fy=0.2)
    #2匹目のワニ
    wimg2 = wimg

    white = np.ones((img.shape), dtype=np.uint8) * 255 #カメラ画像と同じサイズの白画像

    x = 0
    y = img.shape[0]-wimg.shape[0]
    xd = wimg.shape[1]

    #2匹のワニの位置の更新
    for i in range(2):
        cnt[i] = cnt[i]%(img.shape[1]-wimg.shape[1])
    
    #白画像とワニの画像を合成
    #white[y_start:y_end,x_start:x_end] = wimg
    #y_end - y_start == wimg.shape[0]
    #x_end - x_start == wimg.shape[1]
    white[y:img.shape[0],x+cnt[0]:xd+cnt[0]] = wimg
    white[0:wimg2.shape[0],0:xd] = wimg2
    #(0,0)->(xd,wh)

    #カメラ画像にワニの画像を貼り付ける
    dwhite = white
    #ワニがある部分(位置)をカメラ画像から切り抜き、ワニの画像を貼り付ける
    img[dwhite!=[255, 255, 255]] = dwhite[dwhite!=[255, 255, 255]]
    return img


# added function -----------------------------------------
def getFrameNumber(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * 1000 / fps)

    return frame_now

# main----------------------------------------------------
def main():
    global device

    cap = cv2.VideoCapture(device)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print("Size:", ht, "x", wt, "/Fps: ", fps)

    start = time.perf_counter()
    frame_prv = -1
    cnt = [0,0] #ワニの初期位置
    while cap.isOpened() :
        frame_now=getFrameNumber(start, fps)
        if frame_now == frame_prv:
            continue
        frame_prv = frame_now

        ret, frame = cap.read()
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #合成する
        combi(frame, cnt)
        #スピードの設定
        speed = [8, 5]
        cnt[0] = cnt[0] + speed[0]
        cnt[1] = cnt[1] + speed[1]
        cv2.namedWindow("video", cv2.WINDOW_NORMAL)
        cv2.imshow("video", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

# run-----------------------------------------------------
if __name__ == '__main__':
    main()
