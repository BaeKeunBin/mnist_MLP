# -*- coding: utf-8 -*-
"""
Created on Sun May 20 01:21:08 2018
"""

import numpy as np

import struct as st

import matplotlib.pyplot as plt


    
#파일 읽기 
fp_image = open("mnistData\\train-images.idx3-ubyte",'rb')
fp_label = open("mnistData\\train-labels.idx1-ubyte",'rb')

image_list = []
image_target = []
l = 0

s = fp_image.read(16)    #데이터의 앞 16바이트 는 제외
l = fp_label.read(8)     #데이터의 앞 8비트 는 제외


#read mnist
while True:    
    s = fp_image.read(784) #784바이트씩 읽음
    l = fp_label.read(1) #1바이트씩 읽음

    if not s:
        break; 
    if not l:
        break;
        
#unpack
    img = np.reshape( st.unpack(len(s)*'B',s), (28,28)) 
    image_list.append(img.flatten()) #각 숫자이미지를 추가
    image_target.append(int(l[0]))
    
# Layer 별  노드수
inputdim=784    #inputLayer node count
hiddendim=128    #hiddenLayer node count
outputdim=10    #outputLayrt node count

#weight를 초기화
wji=[] #input -> hidden weight     wji[hiddenLayer][inputLayer] 으로 사용
wkj=[] # hie=dden -> output weight wkj[outputLayer][hiddenLayer]

## noname initialization
wji = 0.5*(np.random.rand(hiddendim,inputdim)-np.ones((hiddendim,inputdim))*0.5)
wkj = 0.5*(np.random.rand(outputdim,hiddendim)-np.ones((outputdim,hiddendim))*0.5)

# weight 를 노드별로 계산하기 쉽게 행렬로 정제
wji = np.reshape(wji,(hiddendim,inputdim))
wkj = np.reshape(wkj,(outputdim,hiddendim))




#target 행렬
target = np.eye(10) # 1000000000 = 0, 0100000000 = 1...

#학습률 
eta = 0.01
#모멘텀
a = 1

#학습 반복 횟수
trainEpoch = 10
#1 : 81.3%
#1 : 88.16%

#error 을 그리기 위한 배열 및 변수

#list 에서 max 값의 index 를 리턴하는 함수
def findMaxIndex(list_data):
    maxData = max(list_data)
    i=0
    while True:
        if list_data[i] == maxData:
            return i
        i= i+1
        
def sigmoid(net):
    op = 1 / (np.ones(np.size(net)) + np.exp(-net))
#    print(op)
    return op
#net function
def net(opi,weight):    
    net = np.dot(opi,np.transpose(weight))
    return net
    
# i image_list 의 이미지별 개수 
size_num = len(image_list) #60000개
#size_num = 500
erroraxis =[]
errorgraph=[]
itera =[]
k=0
#######training
print("training start!!")
for n in range(0,trainEpoch):
    print("times : ",n," training...")
    
    erroraxis =[]
    itera.append(k)
    k=k+1

    for i in range(0,size_num):
        opi = image_list[i]
        
        netpj = net(opi,wji)
        opj = sigmoid(netpj)                #input -> hidden
        
        netpk = net(opj,wkj)    
        opk = sigmoid(netpk)                #hidden -> output

        #에러 계산 (LMS) 학습당
        myerror = ((target[image_target[i]] - opk)**2)
        myerrorSum = sum(myerror)/2
        erroraxis.append(myerrorSum)
        
        #델타 계산
        deltapk = (target[image_target[i]]-opk)*(opk*(np.ones(np.size(opk))-opk)) #(tpk-opk)*opk*(1-opk))
        deltapj = opj*(np.ones(np.size(opj))-opj)*np.dot(deltapk,wkj) #opj(1-opj) * deltapk* wkj
        
        #print(np.transpose(deltapk))
        #temp = np.reshape(deltapk,(1,10))
        
        #경사강하
        wkjNew = eta * (np.dot(np.reshape(deltapk,(outputdim,1)) , np.reshape(opj,(1,hiddendim))))
        wjiNew = eta * (np.dot(np.reshape(deltapj,(hiddendim,1)), np.reshape(opi,(1,inputdim))))
        
        wkj = (a*wkj) + wkjNew
        wji = (a*wji) + wjiNew
        
    #에러의 평균으로 그래프를 그림
    errorgraph.append(np.mean(erroraxis))
    
# error 그래프
plt.plot(itera,errorgraph,label="error")
plt.xlabel('iterator')
plt.ylabel('error')
plt.title('graph')
plt.legend()
plt.show()
    
print("training end.")


fp_image.close()
fp_label.close()

######## test
fp_image = open("mnistData\\t10k-images.idx3-ubyte",'rb')
fp_label = open("mnistData\\t10k-labels.idx1-ubyte",'rb')
image_list = []
image_target = []
l = 0

s = fp_image.read(16)    #데이터의 앞 16바이트 는 제외
l = fp_label.read(8)     #데이터의 앞 8비트 는 제외


#read mnist
while True:    
    s = fp_image.read(784) #784바이트씩 읽음
    l = fp_label.read(1) #1바이트씩 읽음

    if not s:
        break; 
    if not l:
        break;
        
#unpack
    img = np.reshape( st.unpack(len(s)*'B',s), (28,28)) 
    image_list.append(img.flatten()) #각 숫자이미지를 추가
    image_target.append(int(l[0]))


        
cnt=0
size_num = len(image_list) # i image_list 의 이미지별 개수
total = size_num
for i in range(0,size_num):
    #image 의 총 개수
    
    
#    print("image : ",i," training...")
#    print(size_num)
    opi = image_list[i]
    
    netpj = net(opi,wji)
    opj = sigmoid(netpj)                #input -> hidden

    netpk = net(opj,wkj)    
    opk = sigmoid(netpk)                #hidden -> output

    
    targetData = findMaxIndex(opk)      #출력에서 최대값의 인덱스가 찾아낸 숫자
    
    if targetData == image_target[i]:
        cnt = cnt+1
#       print(i,"숫자 정답")
            
accuracy = (cnt/total) *100
print(accuracy,"%")