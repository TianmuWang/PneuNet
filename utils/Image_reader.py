import numpy as np
import os
import pandas as pd
import glob
import cv2
import PIL
from PIL import Image


class Imagereader(object):

    def __init__(self):
        self.df_train = None
        self.df_test = None
        self.df_validatiom = None
        self.train_normal = None
        self.train_pneumonia = None
        self.train_Covid19 = None
        self.test_normal = None
        self.test_pneumonia = None
        self.test_Covid19 = None
        pass

    def readImage(self, path, n_C):#path is the folder path and n_C is the number of channel
        datas=[]
        # nW = []
        # nH = []
        x_dirs=os.listdir(path)

        for x_file in x_dirs:
            
            fpath=os.path.join(path,x_file)
            if n_C == 1 :
                _x=Image.open(fpath).convert('L')
                #plt.imshow(_x,"gray")  
            elif n_C ==3:
                _x=Image.open(fpath)
                #plt.imshow(_x) 
            else:       
                print("错误：图像维数错误")
            n_W=_x.size[0]
            n_H=_x.size[1]
            #若要对图像进行放大缩小，激活（去掉注释）以下函数
            '''
            rat=0.8          #放大/缩小倍数
            n_W=int(rat*n_W)
            n_H=int(rat*n_H)
            _x=_x.resize((n_W,n_H))  #直接给n_W,n_H赋值可将图像变为任意大小
            '''
            datas.append(np.array(_x))
            # nH.append(n_H)
            # nW.append(n_W)
            # print(np.shape(_x))
            _x.close() 
    
        datas=np.array(datas)
    
        m=datas.shape[0]
        datas=datas.reshape((m,n_H,n_W,n_C))
        # print(np.min(nH),np.min(nW))
        #print(datas.shape)
        return datas
        pass

    def datasetGen(self, main_path, testGen = False):
        train_path = os.path.join(main_path,"train")
        test_path = os.path.join(main_path,"test")

        train_normal = glob.glob(train_path+"/NORMAL/*.jpeg")
        train_pneumonia = glob.glob(train_path+"/PNEUMONIA/*.jpeg")
        train_Covid19 = glob.glob(train_path+"/Covid19/*.jpeg")

        test_normal = glob.glob(test_path+"/NORMAL/*.jpeg")
        test_pneumonia = glob.glob(test_path+"/PNEUMONIA/*.jpeg")
        test_Covid19 = glob.glob(test_path+"/Covid19/*.jpeg")

        self.train_normal = train_normal
        self.train_pneumonia = train_pneumonia
        self.train_Covid19 = train_Covid19

        self.test_normal = test_normal
        self.test_pneumonia = test_pneumonia
        self.test_Covid19 = test_Covid19

        train_list = [x for x in train_normal]
        train_list.extend([x for x in train_pneumonia])
        train_list.extend([x for x in train_Covid19])


        df_train = pd.DataFrame(np.concatenate([['Normal']*len(train_normal) , ['Pneumonia']*len(train_pneumonia) , ['Covid19']*len(train_Covid19)]), columns = ['class'])
        df_train['image'] = [x for x in train_list]

        test_list = [x for x in test_normal]
        test_list.extend([x for x in test_pneumonia])
        test_list.extend([x for x in test_Covid19])

        df_test = pd.DataFrame(np.concatenate([['Normal']*len(test_normal) , ['Pneumonia']*len(test_pneumonia), ['Covid19']*len(test_Covid19)]), columns = ['class'])
        df_test['image'] = [x for x in test_list]


        if testGen:
            self.df_train = df_train
            self.df_test = df_test
            return df_train, df_test
        else:
            self.df_train = df_train
            return df_train
        
        pass

