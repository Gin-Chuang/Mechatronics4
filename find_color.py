
import cv2
import numpy as np
import time


#img=cv2.imread('C:\\Users\\User\\Dropbox\\PC\\Desktop\\water_color.JPG')
#imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        
    
def ROI(image):
    height=image.shape[0]
    width=image.shape[1]
    mask = np.zeros_like(image)
    match_mask_color = (255,255,255)
    point=np.array([[width/4,height*0.6],[width/4,height],[width*3/4,height],[width*3/4,height*0.6]])#y座標由上而下
    cv2.fillPoly(mask, np.int32([point]), match_mask_color)
    masked_image = cv2.bitwise_and(image, mask) 
    return masked_image
def empty(object):
    pass
# cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars", 640, 240)
# cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
# cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
# cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
# cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
# cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
# cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)
cap = cv2.VideoCapture(0)
target='nothing'
complete_target=0
while(cap.isOpened()):
    if(complete_target==0):
        ret, frame = cap.read()
        #dst = cv2.pyrMeanShiftFiltering(frame, 10, 50)#濾波
        ROI_img=ROI(frame)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        thershold=64
        hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imshow('frame',ROI_img)
        
        #調參數用
        # h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
        # h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
        # s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
        # s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
        # v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
        # v_max = cv2.getTrackbarPos("Val Max", "TrackBars")

        # lower = np.array([h_min, s_min, v_min])
        # upper = np.array([h_max, s_max, v_max])
        # mask = cv2.inRange(hsv, lower, upper)
        #imgResult = cv2.bitwise_and(frame, frame, mask=mask)

        # if cv2.waitKey(10) & 0xFF == ord('q'):
        #     break
        # cv2.imshow('Result',imgResult)

        
        lower_blue=np.array([71,0,0])
        upper_blue=np.array([179,200,182])
        mask_blue=cv2.inRange(hsv,lower_blue,upper_blue)
        lower_red = np.array([0,180,170])
        upper_red = np.array([179,255,255])
        mask_red=cv2.inRange(hsv,lower_red,upper_red)
        lower_yellow=np.array([20,175,0])
        upper_yellow=np.array([29,255,255])
        mask_yellow=cv2.inRange(hsv,lower_yellow,upper_yellow)
        lower_black=np.array([0,0,0])
        upper_black=np.array([179,255,81])
        mask_black=cv2.inRange(hsv,lower_black,upper_black)
        Result_blue = cv2.bitwise_and(ROI_img, frame, mask=mask_blue)
        Result_red = cv2.bitwise_and(ROI_img, frame, mask=mask_red)
        Result_yellow = cv2.bitwise_and(ROI_img, frame, mask=mask_yellow)
        Result_black = cv2.bitwise_and(ROI_img,frame,mask=mask_black)
        cv2.imshow('blue',Result_blue)
        cv2.imshow('red',Result_red)
        cv2.imshow('yellow',Result_yellow)
        cv2.imshow('black',Result_black)
        
        area_blue=[0]
        area_red=[0]
        area_yellow=[0]
        area_black=[0]
        color=[]
        cntblue, hierarchy =cv2.findContours(mask_blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for i in cntblue:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.01*peri, True)
            area_blue.append(cv2.contourArea(i))
        max_blue=max(area_blue)
        color.append(max_blue)
        cntbred, hierarchy =cv2.findContours(mask_red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for i in cntbred:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.01*peri, True)
            area_red.append(cv2.contourArea(i))
        max_red=max(area_red)
        color.append(max_red)

        cntyellow, hierarchy =cv2.findContours(mask_yellow,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for i in cntyellow:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.01*peri, True)
            area_yellow.append(cv2.contourArea(i))
        max_yellow=max(area_yellow)
        color.append(max_yellow)
        
        cntblack, hierarchy =cv2.findContours(mask_black,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        for i in cntblack:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.01*peri, True)
            area_black.append(cv2.contourArea(i))
            #cv2.imshow('target', img)
        max_black=max(area_black)-306081
        color.append(max_black)
        
        actual_color=max(color)
        if actual_color<20000:
            cv2.putText(frame,'no fucking color',(10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("result",frame)
        else:
            if actual_color==max_black:
                target='black'
                cv2.putText(frame,'black',(10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame,str(actual_color),(60, 90), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame,'yellow',(10, 100), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame,str(max_yellow),(60, 190), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow("result",frame)
            elif actual_color==max_red:
                target='red'
                cv2.putText(frame,'red',(10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame,str(actual_color),(60, 90), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("result",frame)
            elif actual_color==max_yellow:
                target='yellow'
                cv2.putText(frame,'yellow',(10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame,str(actual_color),(60, 90), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("result",frame)
                complete_target==1
            elif actual_color==max_blue:
                target='blue'
                cv2.putText(frame,'blue',(10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame,str(actual_color),(60, 90), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("result",frame)

        key=cv2.waitKey(1)
        if key==27:
            break
        if target=='nothing':
             continue
        #time.sleep(3)
    print(target)
    # cap = cv2.VideoCapture(0)
    # while(cap.isOpened()):
    #     ret, frame = cap.read()
    #output=[]
   
    # if target=='black':
    #     output = cv2.inRange(frame, lower_black, upper_black)
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    #     output = cv2.dilate(output, kernel)
    #     output = cv2.erode(output, kernel)
    #     contours, hierarchy = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # elif target=='blue':
    #     output = cv2.inRange(frame, lower_blue, upper_blue)
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    #     output = cv2.dilate(output, kernel)
    #     output = cv2.erode(output, kernel)
    #     print(output)
    # elif target=='yellow':
    #     print(1)
    #     output = cv2.inRange(hsv, lower_yellow, upper_yellow)
    #     print(output)
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    #     output = cv2.dilate(output, kernel)
    #     output = cv2.erode(output, kernel)
        
    # elif target=='red':
    #     output = cv2.inRange(frame, lower_yellow, upper_yellow)
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    #     output = cv2.dilate(output, kernel)
    #     output = cv2.erode(output, kernel)
    # contours, hierarchy = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)        
    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     color = (0,0,255)
    #     x, y, w, h = cv2.boundingRect(contour)                      # 取得座標與長寬尺寸
    #     img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
    #     center_x=x+w/2
    #     center_y=y+h/2
    #     cv2.putText(img2,str(center_x),(100, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
    #     cv2.putText(img2,str(center_y),(100, 100), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
    #     cv2.imshow('img2',img2)
    
    
    
    


    


