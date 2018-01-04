# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 21:43:33 2017

@author: Administrator
"""
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

chengyus=[['劳','苦','功','高'],
        ['非','亲','非','故'],
        ['山','清','水','秀'],
        ['明','辨','是','非'],
        ['清','风','明','月'],
        ['比','屋','可','封'],
        ['心','如','止','水'],
        ['高','山','流','水'],
        ['步','步','莲','花'],
        ['故','步','自','封'],
        ['后','起','之','秀'],
        ['比','翼','双','飞'],
        ['心','花','怒','放'],
        ['放','虎','归','山'],
        ['飞','蛾','扑','火']]
#寻找横向的四字成语方格
def findXFour(mi, i, j):
    if (j+1 == 10 or mi[i][j+1] == '0') and j-1>=0 and mi[i][j-1] != '0':
        return [[mi[i][j-3],mi[i][j-2],mi[i][j-1],mi[i][j]],
                [i,j-3,i,j-2,i,j-1,i,j]]
    elif (j+2 ==10 or (0 if j+2>10 else mi[i][j+2] == '0')) and j-1>=0 and mi[i][j-1] != '0':
        return [[mi[i][j-2],mi[i][j-1],mi[i][j],mi[i][j+1]],
                [i,j-2,i,j-1,i,j,i,j+1]]
    elif (j+3 ==10 or (0 if j+3>10 else mi[i][j+3] == '0')) and j-1>=0 and mi[i][j-1] != '0':
        return [[mi[i][j-1],mi[i][j],mi[i][j+1],mi[i][j+2]],
                [i,j-1,i,j,i,j+1,i,j+2]]
    elif (j+4 ==10 or (0 if j+4>10 else mi[i][j+4] == '0')) and (mi[i][j+1] != '0'):
        return [[mi[i][j],mi[i][j+1],mi[i][j+2],mi[i][j+3]],
                [i,j,i,j+1,i,j+2,i,j+3]]
    else:
        return []
#寻找纵向的四字成语方格
def findYFour(mi, i, j):
    if (i+1==10 or mi[i+1][j] == '0') and i-1>=0 and mi[i-1][j] != '0':
        return [[mi[i-3][j],mi[i-2][j],mi[i-1][j],mi[i][j]],
                [i-3,j,i-2,j,i-1,j,i,j]]
    elif (i+2==10 or (0 if i+2>10 else mi[i+2][j] == '0')) and i-1>=0 and mi[i-1][j] != '0':
        return [[mi[i-2][j],mi[i-1][j],mi[i][j],mi[i+1][j]],
                [i-2,j,i-1,j,i,j,i+1,j]]
    elif (i+3==10 or (0 if i+3>10 else mi[i+3][j] == '0')) and i-1>=0 and mi[i-1][j] != '0':
        return [[mi[i-1][j],mi[i][j],mi[i+1][j],mi[i+2][j]],
                [i-1,j,i,j,i+1,j,i+2,j]]
    elif (i+4==10 or (0 if i+4>10 else mi[i+4][j] == '0')) and (mi[i+1][j] == '0'):
        return [[mi[i][j],mi[i+1][j],mi[i+2][j],mi[i+3][j]],
                [i,j,i+1,j,i+2,j,i+3,j]]
    else:
        return []

            
#改变对应方格的原生成成语填字矩阵
def changeMi(res, micopy):
    counts= []
    for j in range(len(chengyus)):
        count = 0
        if res[0][0] == chengyus[j][0]:
            count += 1
        if res[0][1] == chengyus[j][1]:
            count += 1
        if res[0][2] == chengyus[j][2]:
            count += 1
        if res[0][3] == chengyus[j][3]:
            count += 1
        counts.append(count)
    m = counts.index(max(counts))    
    #print(max(counts))
    micopy[res[1][0]][res[1][1]] = chengyus[m][0]
    micopy[res[1][2]][res[1][3]] = chengyus[m][1]
    micopy[res[1][4]][res[1][5]] = chengyus[m][2]
    micopy[res[1][6]][res[1][7]] = chengyus[m][3]
    chengyus.remove([chengyus[m][0],chengyus[m][1],chengyus[m][2],chengyus[m][3]])

#判断res此四字成语是否在nodes里面存在    
def isExist(nodes,res):
    for i in range(len(nodes)):
        if(nodes[i][0][0] == res[0][0] and nodes[i][0][1] == res[0][1] 
           and nodes[i][0][2] == res[0][2] and nodes[i][0][3] == res[0][3]):
            return 1
    return 0

#实际的查找成语方格并修改原成语矩阵的函数
def getResult(mi, i,j, nodes, micopy):
    temp = findXFour(mi, i, j)
    res = (len(temp) != 0 and temp or findYFour(mi, i, j))
    
    if(len(res) > 0 and not isExist(nodes, res)):
        nodes.append(res)
        changeMi(res, micopy)
             
import copy
#总体的揭解谜函数    
def solve(mi):
    nodes = []
    micopy = copy.deepcopy(mi)  #对象拷贝，深拷贝
    for i in range(len(mi)):
        for j in range(len(mi[i])):
            if(mi[i][j] != '0'):
                getResult(mi, i,j, nodes, micopy)
   # print(nodes)
    return micopy


## 训练knn模型
samples = np.load('samples.npy')
labels = np.load('label.npy')


#print(len(samples))
k = 1102
train_label = labels[:k]
train_input = samples[:k]
test_input = samples[k:]
test_label = labels[k:]


'''
from sklearn.neighbors import KNeighborsClassifier
# fit a k-nearest neighbor model to the data
model = KNeighborsClassifier()

from sklearn.svm import SVC
# fit a SVM model to the data
model = SVC()
model.fit(samples, labels)
'''
'''
# make predictions
predicted = model.predict(test_input)
print(predicted)
print(test_label.reshape(1,len(test_label))[0])
'''
#创建knn对象并训练样本
model = cv2.ml.KNearest_create()
model.train(samples,cv2.ml.ROW_SAMPLE,labels)

'''
retval, results, neigh_resp, dists = model.findNearest(test_input, 1)
string = results.ravel()
print(string)
print(test_label.reshape(1,len(test_label))[0])
'''
#读入原图像
img = cv2.imread('phrases\\phrase.jpg')
#灰度图像生成
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
## 阈值分割
ret,thresh = cv2.threshold(gray,200,255,1)
##膨胀处理
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5, 5))     
dilated = cv2.dilate(thresh,kernel)
 
## 轮廓提取
image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

'''
cv2.imshow("img", dilated)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


##　提取100个小方格
boxes = []
indexs = []
#print(len(hierarchy[0]))
for i in range(len(hierarchy[0])):
    if hierarchy[0][i][3] == 0:
        boxes.append(hierarchy[0][i])
        indexs.append(i)
#print(boxes)        
#print(indexs)
#获取原图像长宽
height,width = img.shape[:2]
box_h = height/10
box_w = width/10
number_boxes = []
numbers = []

miyu =  [[0 for i in range(10)] for i in range(10)]
## 填字游戏初始化为零阵
for i in range(10):
    for j in range(10):
        miyu[i][j] = "0" 
        

hanzis = ['清', '风', '明', '月', '劳', '苦', '功', '高', '辨', '是', '非', '亲', '故', '山', '流', '水', '心',
          '如', '止', '步', '自', '封', '比', '屋', '可', '翼', '双', '飞', '莲', '花', '怒', '放', '虎', '归',
          '后', '蛾','起','扑', '之', '火', '秀']


#print(boxes)
#为了在图片上显示中文的一系列操作
cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2和PIL中颜色的hex码的储存顺序不同
pil_im = Image.fromarray(cv2_im) 
draw = ImageDraw.Draw(pil_im) # 括号中为需要打印的canvas，这里就是在图片上直接打印
font = ImageFont.truetype("font\\kaijian.ttf", 25, encoding="utf-8") # 第一个参数为字体文件路径，第二个为字体大小
n=0

#提取方格中的汉字
for j in range(len(boxes)):
    if boxes[j][2] == -1:
        x,y,w,h = cv2.boundingRect(contours[indexs[j]])
        number_boxes.append([x,y,w,h])
        #cv2.rectangle(img,(x-1,y-1),(x+w-10,y+h-10),(0,0,255),1)
        centerColor = img[round((2*y+h)/2),round((2*x+w)/2)]
        #print(centerColor)
        if(centerColor[0] > 200): #根据原图片当前位置的像素值区分出黄色格与白色格
            #print(y/box_h,round(y/box_h),x/box_w,round(x/box_w))
            miyu[round(y/box_h)][round(x/box_w)] = "1" #白色空格填‘1’
    elif boxes[j][2] != -1:
       # print("有%d"%(j))
        x,y,w,h = cv2.boundingRect(contours[boxes[j][2]])
        #print(x,y,w,h)
       # x,y,w,h = cv2.boundingRect(contours[boxes[j][2]])
        #print(x,y,w,h)
        number_boxes.append([x,y,w,h])
        #cv2.rectangle(img,(x-1,y-1),(x+w+1,y+h+1),(0,255,0),1)
        #img = cv2.drawContours(img, contours, boxes[j][2], (0,255,0), 1)
        ## 对提取的数字进行处理
        number_roi = gray[y:y+h, x:x+w]
        ## 统一大小
        resized_roi=cv2.resize(number_roi,(30,30))
        thresh1 = cv2.adaptiveThreshold(resized_roi,255,1,1,11,2) 
        ## 归一化像素值
        normalized_roi = thresh1/255.  
        '''
        cv2.imshow("thresh1", thresh1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        ## 展开成一行让knn识别
        sample1 = normalized_roi.reshape((1,len(normalized_roi)*len(normalized_roi[0])))
        sample1 = np.array(sample1,np.float32)
        
        ## knn识别
        retval, results, neigh_resp, dists = model.findNearest(sample1, 1)        
        number = int(results.ravel()[0])
        #print(number)
        #numbers.append(number)
        
        '''
        ###
        results = model.predict(sample1)
        number = int(results.ravel()[0])
        '''
        # 第一个参数为打印的坐标，第二个为打印的文本，第三个为字体颜色，第四个为字体
        draw.text((x+(w/2)+10,y-10), str(hanzis[number-1]), (0, 0, 255), font=font) 
        
        ## 求在矩阵中的位置
        miyu[round(y/box_h)][round(x/box_w)] = str(hanzis[number-1])
               
cv2_text_im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_BGR2RGB)
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img", cv2_text_im)
cv2.waitKey(0)
cv2.destroyAllWindows()



print("\n生成的填字游戏\n")
print(miyu)
print("\n求解后的填字游戏\n")
result = solve(miyu)  
print(result)

#将解谜结果填到图片上
for i in range(len(number_boxes)):
    if result[int(i/10)][i%10] != '0':
        x,y,w,h = number_boxes[99-i]
        # 第一个参数为打印的坐标，第二个为打印的文本，第三个为字体颜色，第四个为字体
        draw.text((x+10,y+10), str(result[int(i/10)][i%10]), (0, 0, 255), font=font)


cv2_text_im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_BGR2RGB)
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img", cv2_text_im)
cv2.waitKey(0)
cv2.destroyAllWindows()
