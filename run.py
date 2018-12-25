#coding=utf-8
import cv2
import random


def add_hat_do(preimg):

    outpic=preimg+".result.jpg";
    fromPic=preimg;
    resultPic=outpic;

    # OpenCV 人脸检测      haarcascade_frontalface_alt.xml  lbpcascade_frontalface_improved.xml  haarcascade_profileface.xml
    face_patterns = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml') #opencv的人脸检测库文件
    sample_image = cv2.imread(fromPic)   #你要加帽子的头像图像
    faces = face_patterns.detectMultiScale(sample_image,
                                           scaleFactor=1.1,
                                           minNeighbors=8,
                                           minSize=(50, 50))     #这三行参数可调，以识别出人脸。

    #返回人脸的坐标  x y w h :xy左上角的点的坐标  wh是人脸的长和宽.
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(sample_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #
    # cv2.imwrite('face_detected.png', sample_image);   #把人脸框出来.
    #
    # print faces


    # 圣诞帽
    hats = []
    for i in range(3):   #备选3顶帽子
        hats.append(cv2.imread('hat/hat%d.png' % i, -1))

    for face in faces:  #一张头像内多张脸的话，则需要face循环.
        # 随机一顶帽子
        hat = random.choice(hats)
        print hat.shape
        # 调整帽子尺寸
        print face[3]  #w值 即框出的人脸的宽.
        scale = float(face[3]) / hat.shape[0] * 1.5   #这里不要把float漏了,不然代码一直报错的,因为python默认当分子分母均是int的话,分子小于分母则商为0.
        print scale
        hat = cv2.resize(hat, (0, 0), fx=scale, fy=scale)
        # 根据人脸坐标调整帽子位置
        x_offset = int(face[0] + face[2] / 2 - hat.shape[1] / 2)
        y_offset = int(face[1] - hat.shape[0] / 2)
        # 计算贴图位置，注意防止超出边界的情况
        x1, x2 = max(x_offset, 0), min(x_offset + hat.shape[1], sample_image.shape[1])
        y1, y2 = max(y_offset, 0), min(y_offset + hat.shape[0], sample_image.shape[0])
        hat_x1 = max(0, -x_offset)
        hat_x2 = hat_x1 + x2 - x1
        hat_y1 = max(0, -y_offset)
        hat_y2 = hat_y1 + y2 - y1
        # 透明部分的处理
        alpha_h = hat[hat_y1:hat_y2, hat_x1:hat_x2, 3] / 255
        alpha = 1 - alpha_h
        # 按3个通道合并图片
        for c in range(0, 3):
            sample_image[y1:y2, x1:x2, c] = (alpha_h * hat[hat_y1:hat_y2, hat_x1:hat_x2, c] + alpha * sample_image[y1:y2, x1:x2, c])

    # 保存最终结果
    cv2.imwrite(resultPic, sample_image);
    return outpic;

if __name__ == '__main__':

	re=add_hat_do("pic/test1.jpg");

	print re;
