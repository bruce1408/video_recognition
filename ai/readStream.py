# import cv2
#
# cap = cv2.VideoCapture("http://ivi.bupt.edu.cn/hls/cctv1hd.m3u8")
# # cap = cv2.VideoCapture("/home/bruce/bigVolumn/VideoData/labelVideo/work/1.mp4")
#
# print(cap.isOpened())
# while cap.isOpened():
#     success, frame = cap.read()
#     cv2.imshow("frame", frame)
#     cv2.waitKey(1)
# ret, frame = cap.read()
# while ret:
#     ret, frame = cap.read()
#     cv2.imshow("frame", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()

# import vlc
# import time
#
# player = vlc.MediaPlayer('rtsp://bruce:bruce123@192.168.1.6:30002/test')
# player.play()
# #
# index = 0
# while True:
#     # time.sleep(1)
#     player.video_take_snapshot(0, './img/'+str(index) + '.png', 0, 0)
#     index += 1


import vlc
import ctypes
import time
import sys
import cv2
import numpy
from PIL import Image
index = 0
vlcInstance = vlc.Instance()
# 机场内
# m = vlcInstance.media_new("")
# 机场外
# 记得换url,最好也和上面一样进行测试一下
url = "rtsp://bruce:bruce123@192.168.1.6:30001/test"
m = vlcInstance.media_new(url)
mp = vlc.libvlc_media_player_new_from_media(m)

# ***如果显示不完整，调整以下宽度和高度的值来适应不同分辨率的图像***
video_width = 1080
video_height = 640

size = video_width * video_height * 4
buf = (ctypes.c_ubyte * size)()
buf_p = ctypes.cast(buf, ctypes.c_void_p)

VideoLockCb = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))


@VideoLockCb
def _lockcb(opaque, planes):
    # print("lock", file=sys.stderr)
    planes[0] = buf_p


@vlc.CallbackDecorators.VideoDisplayCb
def _display(opaque, picture):
    global index, flag
    try:
        img = Image.frombuffer("RGBA", (video_width, video_height), buf, "raw", "BGRA", 0, 1)
        opencv_image = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
        cv2.imshow('image', opencv_image)
        cv2.imwrite("./img/"+str(index)+".jpg", opencv_image)
        cv2.waitKey(1)
        index += 1
    except :
        print('error')
        flag = False
        sys.exit(0)


vlc.libvlc_video_set_callbacks(mp, _lockcb, None, _display, None)
mp.video_set_format("BGRA", video_width, video_height, video_width * 4)
flag = True
while flag:
    mp.play()
# time.sleep(1)





