import os
import shutil

def copyFiles2(srcPath,dstPath):
    if not os.path.exists(srcPath):
        print("src path not exist!")
    if not os.path.exists(dstPath):
        os.makedirs(dstPath)
    #递归遍历文件夹下的文件，用os.walk函数返回一个三元组
    for root,dirs,files in os.walk(srcPath):
        for eachfile in files:
            if eachfile.find("DS_Store") > 0:
                continue
            shutil.copy(os.path.join(root,eachfile),dstPath)
            print(eachfile+" copy succeeded")

copyFiles2("D:\\cardiac_data\\Sunnybrook\\Sunnybrook Cardiac MR Database ContoursPart3\\TrainingDataContours",
           "D:\\cardiac_data\\Sunnybrook\\Sunnybrook Cardiac MR Database ContoursPart3\\TrainingDataContours")