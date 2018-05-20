
"""
Created on Sun May 20 01:21:08 2018
"""

import numpy as np

import struct as st

import matplotlib.pyplot as plt



    
#파일 읽기 
fp_image = open("mnistData\\train-images.idx3-ubyte",'rb')
fp_label = open("mnistData\\train-labels.idx1-ubyte",'rb')

image_list = [ [],[],[],[],[],[],[],[],[],[] ] #숫자별로 저장 (0 ~ 9)
d = 0
l = 0
index=0 

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

    index = int(l[0])

#unpack
    img = np.reshape( st.unpack(len(s)*'B',s), (28,28)) 
    image_list[index].append(img.flatten()) #각 숫자영역별로 해당이미지를 추가
    
#print(image_list[0][0])


#weight를 초기화

wji=[] #input -> hidden weight     wji[hiddenLayer][inputLayer] 으로 사용
wkj=[] # hie=dden -> output weight wkj[outputLayer][hiddenLayer]

inputdim=784    #inputLayer node count
hiddendim=256    #hiddenLayer node count
outputdim=10    #outputLayrt node count

    
#초기 weight 를 설정
wji = 0.5*(np.random.random((hiddendim,inputdim))-np.ones((hiddendim,inputdim))*0.5)
wkj = 0.5*(np.random.random((outputdim,hiddendim))-np.ones((outputdim,hiddendim))*0.5)

# weight 를 노드별로 계산하기 쉽게 행렬로 정제
wji = np.reshape(wji,(hiddendim,784))
wkj = np.reshape(wkj,(outputdim,hiddendim))

#target 행렬
target = np.eye(10)

#학습률 
eta = 0.01

#trainepoch
trainEpoch =100



def sigmoid(net):
    op = 1.0 / (np.ones(np.size(net)) + np.exp(-1.0*net))
#    print(op)
    return op
#net function
def net(opi,weight):
    net = np.dot(opi,np.transpose(weight))
    return net

for n in range(0,trainEpoch):
    #error 을 그리기 위한 배열 및 변수
    erroraxis =[]
    itera =[]
    k=0
    print(n," 번째 학습중")
#input 인가
    for i in range(0,10):
        size_num = len(image_list[i]) # i image_list 의 이미지별 개수 
    #    print("image : ",i," training...")
    #    print(size_num)
        for j in range(0,size_num):
            opi = image_list[i][j]
            
            netpj = net(image_list[i][j],wji)
            opj = sigmoid(netpj)                #input -> hidden
            
            netpk = net(opj,wkj)    
            opk = sigmoid(netpk)                #hidden -> output
            
            #에러 계산 (LMS)
            myerror = ((target[i] - opk)**2)
            myerror = sum(myerror)/2
            erroraxis.append(myerror)
            itera.append(k)
            k=k+1
            #델타 계산
            deltapk = (target[i]-opk)*(opk*(np.ones(np.size(opk))-opk)) #(tpk-opk)*opk*(1-opk))
            deltapj = opj*(np.ones(np.size(opj))-opj)*np.dot(deltapk,wkj) #opj(1-opj) * deltapk* wkj
            
            #print(np.transpose(deltapk))
            #temp = np.reshape(deltapk,(1,10))
            
            #경사강하
            wkjNew = eta * (np.dot(np.reshape(deltapk,(outputdim,1)) , np.reshape(opj,(1,hiddendim))))
            wjiNew = eta * (np.dot(np.reshape(deltapj,(hiddendim,1)), np.reshape(opi,(1,inputdim))))
            
            wkj = wkj + wkjNew
            wji = wji + wjiNew
    
    
    # error 그래프
    plt.plot(itera,erroraxis,label="error")
    plt.xlabel('iterator')
    plt.ylabel('error')
    plt.title('graph')
    plt.legend()
    plt.show()

f= open("weight.txt","w")

f.write("wji : "+wji+"\n")
f.write("wkj : "+wkj+"\n")
f.close()

fp_image.close()
fp_label.close()

####### test
fp_image = open("mnistData\\t10k-images.idx3-ubyte",'rb')
fp_label = open("mnistData\\t10k-labels.idx1-ubyte",'rb')

image_list = [ [],[],[],[],[],[],[],[],[],[] ] #숫자별로 저장 (0 ~ 9)
d = 0
l = 0
index=0 

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

    index = int(l[0])

#unpack
    img = np.reshape( st.unpack(len(s)*'B',s), (28,28)) 
    image_list[index].append(img.flatten()) #각 숫자영역별로 해당이미지를 추가


#list 에서 max 값의 index 를 리턴하는 함수
def findMaxIndex(list_data):
    maxData = max(list_data)
    i=0
    while True:
        if list_data[i] == maxData:
            return i
        i= i+1
        
cnt=0
total =0
for i in range(0,10):
    size_num = len(image_list[i]) # i image_list 의 이미지별 개수 
    
    #image 의 총 개수
    total = size_num + total
    
    print("image : ",i," training...")
#    print(size_num)
    for j in range(0,size_num):
        opi = image_list[i][j]
        
        netpj = net(image_list[i][j],wji)
        opj = sigmoid(netpj)                #input -> hidden
        
        netpk = net(opj,wkj)    
        opk = sigmoid(netpk)                #hidden -> output

        targetData = findMaxIndex(opk)
        if targetData == i:
            cnt = cnt+1
#            print(i,"숫자 정답")
            
accuracy = (cnt/total) *100
print(accuracy,"%")