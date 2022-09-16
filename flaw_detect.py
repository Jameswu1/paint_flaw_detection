import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import glob


array_of_img = [] 

def on_press(event):
    print("my position:" ,event.button,event.xdata, event.ydata)

def read_directory(directory_name):
    flag = 1
    tmp1 = 0
    for filename in os.listdir(directory_name):
        tmp1 += 1
        if filename[-3:] != "jpg":
            continue
        ## dataset
        path = []
        path1 = directory_name + "/test_Bilateral"
        path2 = directory_name + "/test_Highlight"
        path3 = directory_name + "/test_Edge"
        path4 = directory_name + "/test_Post"
        path5 = directory_name + "/test_All"
        path6 = directory_name + "/test_pic"
        path7 = directory_name + "/test"
        path.append(path1)
        path.append(path2)
        path.append(path3)
        path.append(path4)
        path.append(path5)
        path.append(path6)
        path.append(path7)

        for i in range(7):
            if not os.path.isdir(path[i]):
                os.mkdir(path[i])

        #if not os.path.isdir(path7):
        #    os.mkdir(path7)


        img = cv2.imread(directory_name + "/" + filename)
        tmp = img

        if flag == 1:
            G = img
            flag -= 1
        ##Bilateral
        img1 = cv2.bilateralFilter(img, 13, 50, 50)
        #img1 = cv2.medianBlur(img, 11)
        #img1 = cv2.blur(img1, (11, 11))
        cv2.imwrite(directory_name + "/test_Bilateral/" + filename ,img1)

        ##High
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        ret , dst = cv2.threshold(gray,180,255,cv2.THRESH_BINARY)
        cv2.imwrite(directory_name + "/test_gray/" + filename ,gray)
        cv2.imwrite(directory_name + "/test_Highlight/" + filename ,dst)
     
        #img = cv2.resize(img,(256,256),interpolation=cv2.INTER_AREA )
        #dst = cv2.resize(dst,(256,256),interpolation=cv2.INTER_AREA )
        #cv2.imshow("2.jpg",img)
        #cv2.imshow("1.jpg",dst)
        #cv2.waitKey(10000)

        ##Edge
        kernel = np.ones((5,5), np.uint8)
        dst = cv2.erode(dst, kernel, iterations = 1)
        
        x = cv2.Sobel(dst, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(dst, cv2.CV_16S, 0, 1)

        absX = cv2.convertScaleAbs(x)# 轉回uint8
        absY = cv2.convertScaleAbs(y)

        dst1 = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        cv2.imwrite(directory_name + "/test_Edge/" + filename ,dst1)

        ##Post
        kernel1 = np.ones((5,5), np.uint8)
        dst1 = cv2.dilate(dst1, kernel1, iterations = 1)
        dst1[dst1>50] = 255


        cv2.imwrite(directory_name + "/test_Post/" + filename ,dst1)


        #####
        counter , hierarcy = cv2.findContours(dst1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        tmp2 = 1
        for i in counter:
            (x, y, w, h) = cv2.boundingRect(i)
            if cv2.contourArea(i) < 500:

                #第一版 只抓取 瑕疵位置 並且進行resize
                #test1 = img[y-10:y+h+20,x-10:x+w+20]
                #test1 = cv2.resize(test1,(256,256),interpolation=cv2.INTER_AREA )

                #第二版 抓取中心點 並且根據 最後模型需求的size進行可動式調整
                M = cv2.moments(i)
                x = int( M['m10'] / M['m00'] )
                y = int( M['m01'] / M['m00'] )
                if y + 64 > 2048 or x + 64 > 2448 or x - 64 < 0 or y - 64 < 0 :
                    continue
                test = img[ y-64 : y+64 , x-64 : x+64 ]

                P = directory_name + "/test_pic/" + str(tmp1) 
                if not os.path.isdir(P):
                    os.mkdir(P)
                T = img.copy()
                cv2.rectangle(T, (x-32, y-32), (x+32,y+32), (0, 255, 0), 5)
                cv2.imwrite(directory_name + "/test_pic/" + str(tmp1) + "/" + "X_" + str(x) + "_Y_" + str(y)  + ".jpg" , T)
                cv2.imwrite(directory_name + "/test_pic/" + str(tmp1) + "/" + "X_" + str(x) + "_Y_" + str(y) + "pic" + ".jpg" , test)
                cv2.imwrite(directory_name + "/test/" + str(tmp1) + "_" + "X_" + str(x) + "_Y_" + str(y) +  ".jpg" , test)
                #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #img1 = cv2.resize(tmp,(1024,1024),interpolation=cv2.INTER_AREA )
                #T = cv2.resize(T,(1024,1024),interpolation=cv2.INTER_AREA )
                tmp2 += 1
                #cv2.imshow("23jpg",T)
                #cv2.imshow("2.jpg",test)
                #cv2.imshow("1.jpg",test1)
                #n = cv2.waitKey(1000000)
                #if n == ord('a'):
                #    #cv2.imwrite(directory_name + "/test_flaw/" + str(tmp1) + "/" + str(tmp2) + "_" + str(x) + "__" +  str(y) + filename , test1)
                #    cv2.imwrite(directory_name + "/test_dot_flaw/" +  str(tmp2) + "_" + str(x) + "__" +  str(y) + filename , test)
                #    print("AAA")
                #elif n == ord('s'):
                #    #cv2.imwrite(directory_name + "/test_Noflaw/" + str(tmp1) + "/" + str(tmp2) + "_" + str(x) + "__" +  str(y) + filename , test1)
                #    cv2.imwrite(directory_name + "/test_fair_flaw/"  + str(tmp2) + "_" + str(x) + "__" +  str(y) + filename , test)
                #    print("AA")
                #elif n == ord('d'):
                #    #cv2.imwrite(directory_name + "/test_Noflaw/" + str(tmp1) + "/" + str(tmp2) + "_" + str(x) + "__" +  str(y) + filename , test1)
                #    cv2.imwrite(directory_name + "/test_hole_flaw/"  + str(tmp2) + "_" + str(x) + "__" +  str(y) + filename , test)
                #    print("AA")
                #elif n == ord('f'):
                #    #cv2.imwrite(directory_name + "/test_Noflaw/" + str(tmp1) + "/" + str(tmp2) + "_" + str(x) + "__" +  str(y) + filename , test1)
                #    cv2.imwrite(directory_name + "/test_Noflaw/"  + str(tmp2) + "_" + str(x) + "__" +  str(y) + filename , test)
                #    print("AA")
                #print("A")

        cv2.imwrite(directory_name + "/test_all/" + filename ,img)
        #cv2.imwrite(directory_name + "/test_final/" + filename ,img)
    #fig = plt.figure()
    #plt.imshow(G, animated= True)
    #fig.canvas.mpl_connect('button_press_event', on_press)
    #plt.show()    

read_directory("C:\\Users\\Buslab_GG\\Desktop\\2022-07-19\\AA_DL\\T_2")



