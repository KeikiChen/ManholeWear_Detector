import cv2
import os
import time

save_folder_l = 'c:\\users\\c8703\\desktop\\test\\L'  #floder name
save_folder_r = 'c:\\users\\c8703\\desktop\\test\\R'
image_prefix = "manhole"  # file name    
image_count = 10
interval = 100   #mini seconds

capture_l = cv2.VideoCapture(0)  # camera
capture_r = cv2.VideoCapture(1)

time.sleep(2)

# 设置摄像头分辨率为1920x1080
capture_l.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture_l.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
capture_r.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture_r.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not capture_l.isOpened() & capture_r.isOpened():
    print("Cant open camera")
    exit()

if not os.path.exists(save_folder_l):
    os.makedirs(save_folder_l)
if not os.path.exists(save_folder_r):
    os.makedirs(save_folder_r)

namber = 268    # capture namber

def save_image_l(frame,count):
    save_path = os.path.join(save_folder_l, f"{image_prefix}_{namber +1}_{count+1}_{time.strftime('%Y%m%d_%H%M')}.jpg")
    cv2.imwrite(save_path, frame)

def save_image_r(frame,count):
    save_path = os.path.join(save_folder_r, f"{image_prefix}_{namber +1}_{count+1}_{time.strftime('%Y%m%d_%H%M')}.jpg")
    cv2.imwrite(save_path, frame)

# 创建窗口并设置窗口的大小
cv2.namedWindow("Camera_l", cv2.WINDOW_NORMAL)
cv2.namedWindow("Camera_r", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera_l", 960, 540)
cv2.resizeWindow("Camera_r", 960, 540)

while True:
    ret, frame_l = capture_l.read()  
    ret, frame_r = capture_r.read()  
    cv2.imshow("Camera_l", frame_l)  
    cv2.imshow("Camera_r", frame_r)
    key = cv2.waitKey(1)
    # enter Q to save image
    if key == ord('q'):
        break
    if key == 13:
        print(time.strftime('%Y%m%d_%H%M'))
        for i in range(image_count):
            ret, frame_l = capture_l.read()  
            ret, frame_r = capture_r.read()  
            key = cv2.waitKey(interval)
            save_image_l(frame_l,i)
            save_image_r(frame_r,i)
        print("over")
        namber +=1

capture_l.release()  
capture_r.release()
cv2.destroyAllWindows()  