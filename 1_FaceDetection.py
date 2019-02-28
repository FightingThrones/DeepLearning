"""
Face detect demo
"""
# import face_recognition
# import cv2
#
# img=face_recognition.load_image_file("./images/face.jpg")
# face_locations=face_recognition.face_locations(img)
#
# #显示图片
#
# img=cv2.imread('./images/face.jpg')
# cv2.namedWindow("OriginalPicture")
# cv2.imshow("OriginalPicture",img)
#
# #遍历每个人脸，并标注
# faceNum=len(face_locations)
# for i in range(0,faceNum):
#     top=face_locations[i][0]
#     right=face_locations[i][1]
#     bottom=face_locations[i][2]
#     left=face_locations[i][3]
#
#     start=(left,top)
#     end=(right,bottom)
#     color=(55,34,35)
#     thickness=3
#     cv2.rectangle(img,start,end,color,thickness)
#
#     cv2.namedWindow("FaceDetection")
#     cv2.imshow("FaceDetection",img)
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

""""
案例二 识别图片中的人脸
"""

# import face_recognition
# Tom_Cruise_image=face_recognition.load_image_file("./images/Tom_Cruise.jpg")
# John_Salley_image=face_recognition.load_image_file("./images/John_Salley.jpg")
# test_image=face_recognition.load_image_file("./images/test.jpg")
#
# Tom_Cruise_encoding=face_recognition.face_encodings(Tom_Cruise_image)[0]
# John_Salley_encoding=face_recognition.face_encodings(John_Salley_image)[0]
# test_encoding=face_recognition.face_encodings(test_image)[0]
#
# results=face_recognition.compare_faces([Tom_Cruise_encoding,John_Salley_encoding],test_encoding)
# labels=['Tom_Cruise','John_Salley']
#
# print('results:'+str(results))
#
# for i in range(0,len(results)):
#     if results[i]==True:
#         print("The person is:"+labels[i])

# """
# 案例三  摄像头实时识别人脸
# """
# import  face_recognition
# import cv2
#
# video_capture=cv2.VideoCapture(0)
#
# John_Salley_img=face_recognition.load_image_file("./images/John_Salley.jpg")
# John_Salley_face_encoding=face_recognition.face_encodings(John_Salley_img)[0]
#
# face_locations=[]
# face_encodings=[]
# face_names=[]
# process_this_frame=True
#
# while True:
#     ret,frame=video_capture.read()
#     small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
#
#     if process_this_frame:
#         face_locations=face_recognition.face_locations(small_frame)
#         face_encodings=face_recognition.face_encodings(small_frame,face_locations)
#
#         face_names=[]
#         for face_encoding in face_encodings:
#             match=face_recognition.compare_faces([John_Salley_face_encoding],face_encoding)
#
#             if match[0]:
#                 name="John_Salley"
#             else:
#                 name="unknown"
#             face_names.append(name)
#
#     process_this_frame=not process_this_frame
#     for (top,right,bottom,left),name in zip(face_locations,face_names):
#         top*=4
#         right*=4
#         bottom*=4
#         left*=4
#
#         cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
#
#         cv2.rectangle(frame,(left,bottom-35),(right,bottom),(0,0,255),2)
#         font=cv2.FONT_HERSHEY_DUPLEX
#         cv2.putText(frame,name,(left+6,bottom-6),font,1.0,(255,255,255),1)
#
#     cv2.imshow('Video',frame)
#
#     if cv2.waitKey(1) & 0xFF==ord('q'):
#         break
#
# video_capture.release()
# cv2.destroyAllWindows()

""""
案例四  dlib opencv库应用
"""
# import sys
# import dlib
# import cv2
#
# detector=dlib.get_frontal_face_detector()
#
# #传入参数测试
# for f in sys.argv[1:]:
#     #利用opencv读取图片
#     img=cv2.imread(f,cv2.IMREAD_COLOR)
#
#     #分离三个颜色通道
#     b,g,r=cv2.split(img)
#     #融合三个颜色通道生成新图片
#     img2=cv2.merge([r,g,b])
#
#
#     #利用detector进行人脸检测
#     #dets接收返回的结果
#     dets=detector(img,1)
#     #命令行打印检测到的人脸数
#     print("Number of faces detected:{}".format(len(dets)))
#
#     for index,face in enumerate(dets):
#         print('face{};left{};top{};right{};bottom{}'.format(index,face.left(),face.top(),face.right(),face.bottom()))
#
#     #用矩形框标记人脸
#     left=face.left()
#     top=face.top()
#     right=face.right()
#     bottom=face.bottom()
#     cv2.rectangle(img,(left,top),(right,bottom),(0,255,0),3)
#     cv2.namedWindow(f,cv2.WINDOW_AUTOSIZE)
#     cv2.imshow(f,img)
#
# k=cv2.waitKey(0)
# cv2.destroyAllWindows()

import sys
import dlib
import cv2
import os

#获取当前路径
current_path=os.getcwd()
#模型库所在的位置
predictor_path=current_path+"\\model/shape_predictor_68_face_landmarks.dat"
#人脸存放的位置，运行时指定图片的名字，程序到此文件夹读取图片
face_directory_path=current_path+"\\images\\"

#获取人脸分类器
detector=dlib.get_frontal_face_detector()
#获取人脸检测器
predictor=dlib.shape_predictor(predictor_path)

#传入的命令行参数
for f in  sys.argv[1:]:
    #图片路径，目录+文件名
    face_path=face_directory_path+f

    #opencv读取图片，并显示
    img=cv2.imread(face_path,cv2.IMREAD_COLOR)

    #分离三个颜色通道
    b,g,r=cv2.split(img)
    #融合三个颜色通道生成新图片
    img2=cv2.merge([r,g,b])

    #dets接收detector检测到的人脸
    dets=detector(img,1)
    #打印检测到的人脸个数
    print("Number of faces detected:{}".format(len(dets)))

    for index,face in enumerate(dets):
        print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(),
                                                                     face.bottom()))

        #shape接收检测到的68个人脸标记点
        shape=predictor(img,face)
        #遍历所有标记点，并且在图片上显示标记点
        for index,pt in enumerate(shape.parts()):
            print('Part{}:{}'.format(index,pt))
            pt_pos=(pt.x,pt.y)
            cv2.circle(img,pt_pos,2,(255,0,0),1)

        cv2.namedWindow(f,cv2.WINDOW_AUTOSIZE)
        cv2.imshow(f,img)

k=cv2.waitKey(0)
cv2.destroyAllWindows()



