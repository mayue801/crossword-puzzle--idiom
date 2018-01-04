# -*- coding: UTF-8 -*-
import glob as gb
import cv2
import numpy as np

#针对的是印刷版的汉字，所以采用了投影法分割
#此函数是行分割，结果是一行文字
def YShadow(path):
    img  = cv2.imread(path)       
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    height,width = img.shape[:2]
    
    #blur = cv2.GaussianBlur(gray,(5,5),0)
    
    blur = cv2.blur(gray,(8,8))
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2) 
    temp = thresh
    
    if(width > 500 and height > 400):
        kernel = np.ones((5,5),np.uint8) #卷积核
        dilation = cv2.dilate(thresh,kernel,iterations = 1) #膨胀操作使得单个文字图像被黑像素填充
        temp = dilation
    
    '''
    cv2.imshow('image',temp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
    perPixelValue = 1 #每个像素的值
    projectValArry = np.zeros(width, np.int8) #创建一个用于储存每列黑色像素个数的数组


    for i in range(0,height):
        for j in range(0,width):
            perPixelValue = temp[i,j]
            if (perPixelValue == 255): #如果是黑字
                projectValArry[i] += 1
       # print(projectValArry[i])
            
    canvas = np.zeros((height,width), dtype="uint8")
    
    for i in range(0,height):
        for j in range(0,width):
            perPixelValue = 255 #白色背景
            canvas[i, j] = perPixelValue
   

    for i in range(0,height):
        for j in range(0,projectValArry[i]):
            perPixelValue = 0 #黑色直方图投影
            canvas[i, width-j-1] = perPixelValue
    '''
    cv2.imshow('canvas',canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
    list = []
    startIndex = 0 #记录进入字符区的索引  
    endIndex = 0 #记录进入空白区域的索引  
    inBlock = 0 #是否遍历到了字符区内  


    for i in range(height):
        if (inBlock == 0 and projectValArry[i] != 0):
            inBlock = 1  
            startIndex = i
        elif (inBlock == 1 and projectValArry[i] == 0):
            endIndex = i
            inBlock = 0
            subImg = gray[startIndex:endIndex+1,0:width] #endIndex+1
            #print(startIndex,endIndex+1)
            list.append(subImg)
    #print(len(list))
    return list


#对行字进行单个字的分割
def XShadow(path):
    img  = cv2.imread(path)       
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    height,width = img.shape[:2]
   # print(height,width)
    #blur = cv2.GaussianBlur(gray,(5,5),0)
    
    blur = cv2.blur(gray,(8,8))
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2) 
    
    if(width > 500):
        kernel = np.ones((4, 4),np.uint8) #卷积核
    else:
        kernel = np.ones((2, 2),np.uint8) #卷积核
    dilation = cv2.dilate(thresh,kernel,iterations = 1) #膨胀操作使得单个文字图像被黑像素填充
    
    '''
    cv2.imshow('image',thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
    perPixelValue = 1 #每个像素的值
    projectValArry = np.zeros(width, np.int8) #创建一个用于储存每列黑色像素个数的数组


    for i in range(0,width):
        for j in range(0,height):
            perPixelValue = dilation[j,i]
            if (perPixelValue == 255): #如果是黑字
                projectValArry[i] += 1
       # print(projectValArry[i])
            
    canvas = np.zeros((height,width), dtype="uint8")
    
    for i in range(0,width):
        for j in range(0,height):
            perPixelValue = 255 #白色背景
            canvas[j, i] = perPixelValue
   

    for i in range(0,width):
        for j in range(0,projectValArry[i]):
            perPixelValue = 0 #黑色直方图投影
            canvas[height-j-1, i] = perPixelValue
    '''
    cv2.imshow('canvas',canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
    list = []
    startIndex = 0 #记录进入字符区的索引  
    endIndex = 0 #记录进入空白区域的索引  
    inBlock = 0 #是否遍历到了字符区内  


    for i in range(width):
        if (inBlock == 0 and projectValArry[i] != 0):
            inBlock = 1  
            startIndex = i
        elif (inBlock == 1 and projectValArry[i] == 0):
            endIndex = i
            inBlock = 0
            #subImg = gray[0:height, startIndex:endIndex+1] #endIndex+1
            #print(startIndex,endIndex+1)
            list.append([startIndex, 0, endIndex-startIndex-1, height])
    #print(len(list))
    return list
    
chars = ['qing', 'feng', 'ming', 'yue', 'lao', 'ku', 'gong', 'gao', 'bian', 'shi', 'fei', 'qin', 'gu', 'shan', 'liu', 'shui', 'xin',
          'ru', 'zhi', 'bu', 'zi', 'feng1', 'bi', 'wu', 'ke', 'yi', 'shuang', 'fei1', 'lian', 'hua', 'nu', 'fang', 'hu', 'gui',
          'hou', 'e','qi','pu', 'zhi', 'huo', 'xiu']   
rowChars=['one','two','three']
labels=[]
samples=[]
#实际的分割函数，并生成行字与单个字的图片
def createImgLabel(realpath, k):
    n = 0
    listY = YShadow(realpath)
    for i in range(len(listY)):
        path = 'rowChars\\'+ (rowChars[i]) + '\\'+ (str(k)) + '.jpg'
        #j += 1
        cv2.imwrite(path,listY[i])
        listX = XShadow(path)
        list_sorted = sorted(listX,key = lambda t : t[0])
        img  = cv2.imread(path) 
        #print(list_sorted)
        '''
        cv2.imshow('canvas1',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        print(len(listX))
        for m in range(len(listX)):
            [x1,y1,w1,h1] = list_sorted[m]
            #print(x1,y1,w1,h1)
            ## 切割出每一个数字
            number_roi = gray[y1:y1+h1, x1:x1+w1] #Cut the frame to size
            ## 对图片进行大小统一和预处理
            #blur = cv2.GaussianBlur(number_roi,(5,5),0)
           # u = cv2.adaptiveThreshold(blur,255,1,1,11,2)
            
            
            resized_roi=cv2.resize(number_roi,(30,30))
            thresh = cv2.adaptiveThreshold(resized_roi,255,1,1,11,2)
            #归一化处理
            normalized_roi = thresh/255
            '''
            cv2.imshow('thresh',number_roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            '''
           # print(n)
            
            sub_path = 'chars\\'+ (chars[n]) + '\\'+(str(k))+'.jpg'
            cv2.imwrite(sub_path,thresh)
            #print(len(normalized_roi)*len(normalized_roi[0]))
            ## 把图片展开成一行，然后保存到samples
            ## 保存一个图片信息，保存一个对应的标签
            sample1 = normalized_roi.reshape((1,900))
            samples.append(sample1[0])
            labels.append(float(n+1))
            n += 1
            k += 1
            #print(n)
        '''
        cv2.imshow('canvas1',gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
    
        
        
        
## 获取samples文件夹下所有文件路径
img_path = gb.glob("samples\\*")

## 对每一张图片进行处理
k = 0
for path in img_path:
    print(path)
    createImgLabel(path, k)
    k += 41
    
    

## 这里还是把它们保存成了np.array...
samples = np.array(samples,np.float32)
labels = np.array(labels,np.float32)
labels = labels.reshape((labels.size,1))

np.save('samples.npy',samples)
np.save('label.npy',labels)
'''
## 保存完加载一下试试
test = np.load('samples.npy')
label = np.load('label.npy')
print(test[0])
print(test[0].shape)
print('label: ', label[0])
'''
    
    
    
    
    
    
    
    
    
    
    
    
    

 