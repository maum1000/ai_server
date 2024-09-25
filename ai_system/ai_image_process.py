
import os
from idlelib.iomenu import encoding
from pathlib import Path
import cv2
import numpy as np
import logging
import os
import cv2
import dlib
import torch
import torch.nn as nn
import numpy as np
import pickle
import face_recognition
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import truetype
from sympy import false
from torchvision import models, transforms
from ultralytics import YOLO
from mtcnn import MTCNN
import time
import piexif
from abc import ABC, abstractmethod
import io
import shutil
from pathlib import Path
import warnings
from .image import Image_Process
from io import BytesIO



from .CModel import YOLOFaceDetector,DlibFaceDetector,FairFacePredictor,MTCNNFaceDetector,FaceDetectors


base_dir         = os.path.join(Path(__file__).resolve().parent, 'ai_files')
django_media_dir = os.path.join(Path(__file__).resolve().parent.parent.parent, 'media/pybo/answer_image')

config = {
    "dlib_model_path": os.path.join(base_dir, 'ai_models', 'DilbCNN', 'mmod_human_face_detector.dat'),
    "yolo_model_path": os.path.join(base_dir, 'ai_models', 'YOLOv8', 'yolov8n-face.pt'),
    "fair_face_model_path": os.path.join(base_dir, 'ai_models', 'FairFace', 'resnet34_fair_face_4.pt'),
    "image_folder": os.path.join(base_dir, 'image_test', 'test_park_mind_problem'),
    "pickle_path": os.path.join(base_dir, 'embedings', 'FaceRecognition(ResNet34).pkl'),
    "font_path": os.path.join(base_dir, 'fonts', 'NanumGothic.ttf'),
    "results_folder": django_media_dir,
}



class AIImage_Process:

    yoloDetector        = YOLOFaceDetector(config['yolo_model_path'])
    fairfacePredictor   = FairFacePredictor(config['fair_face_model_path'])
    mtCNNFaceDetactor   = MTCNNFaceDetector()
    dlibFaceDetactor    = DlibFaceDetector(config['dlib_model_path'])



    # 전송받은 이미지에서 얼굴탐지
    def _detect_faces_single(self, image_data):
        all_results =[]
        try:
         image = Image.open(BytesIO(image_data))
         image_np = np.array(image)
         if image_np.size ==0:
             print("image is empty")

         image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)  # RGB로 변환
         all_results.extend(self.yoloDetector.predict(image_np))
         print("yolo Detector==================")

        except Exception as e:
            print("error exception ",e)

        return image_rgb, self._apply_non_max_suppression(all_results)


    def _detect_faces(self, image_path):
         all_results =[]

         try:

              image = cv2.imread(image_path)  # 이미지 읽기

              if image is None:
                raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

              image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB로 변환

              all_results.extend(self.yoloDetector.predict(image))
             # all_results.extend(self.mtCNNFaceDetactor.predict(image_rgb))
              all_results.extend(self.dlibFaceDetactor.predict(image_rgb))
              print("detect_faces",all_results)
              return image_rgb, self._apply_non_max_suppression(all_results)

         except Exception as e:
              logging.error(f"얼굴 탐지 중 오류 발생: {e}")
         finally:
                logging.info(f"{len(all_results)}개의 얼굴 검출")




    @staticmethod
    def _apply_non_max_suppression(faces):
        if len(faces) == 0:
            return []
            #
        #
        # 얼굴 영역 좌표 비최대 억제 로직
        boxes = np.array(faces).astype("float")
        pick = []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        #
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            #
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            #
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:last]]
            #
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > 0.3)[0])))
            #
        #
        return boxes[pick].astype("int").tolist()

    def _complicate_predictions(self, image_rgb, faces):
        """얼굴을 예측 결과를 잘 추합해서 반환"""
        predictions = []
        face_cnt = 0
        race_cnt = {'백인': 0, '흑인': 0, '아시아': 0, '중동': 0}
        male_cnt = 0
        #
        for face in faces:
            prediction = self._predict_face(image_rgb, face)
            if prediction:
                predictions.append(prediction)
                face_cnt += 1
                race_text, gender_text = prediction[4], prediction[5]
                race_cnt[race_text] += 1
                if gender_text == '남성':
                    male_cnt += 1
                    #
                #
            #
        #
        return predictions, face_cnt, race_cnt, male_cnt

    # model_type은 유사도 측정인지 특정 인물 검색인지
    def _encoding_face(self, img_rgb, face_info):

        print("face_info",face_info)
        x, y, x2, y2 = face_info  # 얼굴 좌표

        face_width, face_height = x2 - x, y2 - y  # 얼굴 크기

        face_location = (y,x2,y2,x)
        encoding_value =  face_recognition.face_encodings(img_rgb,[face_location] )  # 얼굴 인코딩

        #[(y, x + face_width, y + face_height, x)]
        return encoding_value


    import cv2


    def process_detect_image(self, img_buffer):

        img_rgb, info = self._detect_faces_single(img_buffer)
        return self._draw_result_single(img_rgb,info)


    def _predict_face_similarity(self, img_rgb_ori,img_rgb_dest, ori_face, des_face):
        """단일 얼굴에 대해 예측 수행"""
        print("predict face=============similarity ")
        try:

            # ori_face, des_face첫 리스트 정보만 이용하기
            #list로 데이타가 넘어옴
            x, y, x2, y2            = ori_face[0] # 얼굴 좌표
            face_width, face_height = x2 - x, y2 - y  # 얼굴 크기

            x_d,y_d,x2_d,y2_d = des_face[0]
            face_width_d, face_height_d = x2_d -x_d, y2_d - y_d

            print(f"Image Shape: {img_rgb_ori.shape}")
            print(f"Face Slice Coordinates: y:{y}:{y2}, x:{x}:{x2}")

            face_image_ori = img_rgb_ori[y:y2, x:x2]  # 얼굴 이미지 원본
            face_image_des =  img_rgb_dest[y_d:y2_d, x_d:x2_d]  # 얼굴 이미지2 대상

            encoding_ori = self._encoding_face(img_rgb_ori,ori_face[0])
            encoding_dest = self._encoding_face(img_rgb_dest,des_face[0])

            #cv2.imwrite('face_image_ori.png', cv2.cvtColor(face_image_ori, cv2.COLOR_RGB2BGR))


            #얼굴 인코딩 실패 시 예외 발생
            if not encoding_ori or not encoding_dest:
                logging.warning(f"얼굴 인코딩 실패: {ori_face[0]}")
                return None
            print("=============encoding ====================")

            prediction_result = self.fairfacePredictor.predict(face_image_ori)  # 얼굴 예측


            race_text = prediction_result.get("race", "알 수 없음")  # 인종
            gender_text = prediction_result.get("gender", "알 수 없음")  # 성별
            box_color = prediction_result.get("box_color", (0, 0, 0))  # 박스 색상
            age_text = prediction_result.get("age", "알 수 없음")  # 나이
            #
            # 예측 결과 텍스트
            is_gaka = any(face_recognition.compare_faces([encoding_ori[0]], encoding_dest[0], tolerance=0.3))
            distance = face_recognition.face_distance([encoding_ori[0]],encoding_dest[0])
            similarity_ratio = (1- distance)*100
            print("similarity : " , similarity_ratio)



            prediction_text = f"유사도 : {similarity_ratio} %"
            #
            #return x, y, x2 - x, y2 - y, race_text, gender_text, box_color, prediction_text
            return x_d, y_d, x2_d - x_d, y2_d - y_d, race_text, gender_text, box_color, prediction_text
            #
        #
        except Exception as e:
            logging.error(f"  _predict_face_similarity  단일 얼굴 처리 중 오류 발생: {e}")
            return None

    def _predict_face(self, image_rgb, face):
        """단일 얼굴에 대해 예측 수행"""


        try:
            with open(config['pickle_path'], 'rb') as f:
                 target_encodings = np.array(pickle.load(f))


            x, y, x2, y2 = face  # 얼굴 좌표
            face_width, face_height = x2 - x, y2 - y  # 얼굴 크기
            face_image = image_rgb[y:y2, x:x2]  # 얼굴 이미지
            encodings = face_recognition.face_encodings(image_rgb, [(y, x + face_width, y + face_height, x)])  # 얼굴 인코딩
            #
            # 얼굴 인코딩 실패 시 예외 발생
            if not encodings:
                logging.warning(f"얼굴 인코딩 실패: {face}")
                return None
                #
            #
            prediction_result = self._detect_faces(face_image)  # 얼굴 예측

            race_text = prediction_result.get("race", "알 수 없음")  # 인종
            gender_text = prediction_result.get("gender", "알 수 없음")  # 성별
            box_color = prediction_result.get("box_color", (0, 0, 0))  # 박스 색상
            age_text = prediction_result.get("age", "알 수 없음")  # 나이
            #
            # 예측 결과 텍스트
            is_gaka = any(face_recognition.compare_faces(target_encodings, encodings[0], tolerance=0.3))
            prediction_text = '가카!' if is_gaka and gender_text == '남성' else age_text
            #
            return x, y, x2 - x, y2 - y, race_text, gender_text, box_color, prediction_text
            #
        #
        except Exception as e:
            logging.error(f"단일 얼굴 처리 중 오류 발생: {e}")
            return None

    def process_image_similarity(self, org_img_path, des_img_path):

        try:

            print('process image path ',org_img_path )
            print('process image path ', des_img_path)
            image_rgb, faces = self._detect_faces(org_img_path)  # 얼굴 탐지
            image_rgb_des, des_faces = self._detect_faces(des_img_path)  #얼굴 탐지

            print(f"image ori face ", faces)
            print(f"image des face" , des_faces)

            print(len(faces))

            #if(len(faces)>1 or len(faces)==0): # 얼굴은 한명만 있게 체크 하기 한명도 없으면 에러메시지 띄우기
            #    return False

            #predictions, face_cnt, race_cnt, male_cnt = self._predict_face_similarity(image_rgb, faces,image_rgb_des, des_faces)
            predictions = self._predict_face_similarity(image_rgb, image_rgb_des,faces,des_faces)

            print("*****************************",predictions)
            face_cnt = 1
            male_cnt = 1
            race_cnt = 1


            # 얼굴 예측
            result_image = self._draw_results(image_rgb_des, predictions, face_cnt, male_cnt, race_cnt)  # 결과 그리기
            output_path  = self._save_results(des_img_path, result_image, predictions)  # 결과 저장

            django_path = os.path.join(
                # 이미지 경로를 Django에서 사용할 수 있는 형태로 변환
                'pybo/answer_image',
                os.path.basename(output_path)
            )

            print("image path :",django_path)

            return django_path

        except Exception as e:
            logging.error(f" process_image_similarity 이미지 처리 중 오류 발생: {e}")
            return None




    def process_image(self,img_path, model_type):
        try:
            image_rgb, faces                          = self._detect_faces(img_path)  # 얼굴 탐지
            predictions, face_cnt, race_cnt, male_cnt = self._complicate_predictions(image_rgb, faces)  # 얼굴 예측
            result_image                              = self._draw_results(image_rgb, predictions, face_cnt, male_cnt, race_cnt)  # 결과 그리기
            self._save_results(img_path, result_image, predictions)  # 결과 저장
        except Exception as e:
            logging.error(f"이미지 처리 중 오류 발생: {e}")
            #
    def _draw_result_single(self, image_rgb, predictions):
        """결과를 이미지에 그린 후 리턴"""
        font_size = max(12, int(image_rgb.shape[1] / 200)) # 폰트 크기

        #이미지 정보를 얻어 오기
        result_boxs, result_padding, scale2 = self.getScaleValueFromImage_single(image_rgb.shape, 512, predictions)
        #print("result box=============",result_boxs)
        resized_img = cv2.resize(image_rgb, (int(image_rgb.shape[1] * scale2), int(image_rgb.shape[0] * scale2)))
        color =(0,0,0)
        new_img = cv2.copyMakeBorder(resized_img, result_padding[0], result_padding[1], result_padding[2],  result_padding[3], cv2.BORDER_CONSTANT, value =color)


        color = (191, 191, 191)

        for rect in result_boxs:
            new_img = cv2.rectangle(new_img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]),color, 2)
            # new_img = draw_korean_text(font_path, new_img, rect[4], (rect[0], rect[1]), 15, font_color=(0, 0, 0),
            #                            background_color=(255, 0, 0))

        # info_text = f"검출된 인원 수: {face_cnt}명\n남성: {male_cnt}명\n여성: {face_cnt - male_cnt}명\n"
        # race_info = "\n".join([f"{race}: {count}명" for race, count in race_cnt.items() if count > 0])
        # new_img = extend_image_with_text(self.config, new_img, info_text + race_info, font_size)
        #
        return new_img


    def getScaleValueFromImage_single(self,image_shape, target_size, face_infos):

        #원본이미지 w,h값
        h, w ,_= image_shape

        scale = target_size / max(h, w)

        resize_width  = (int(w * scale))
        resize_height =  (int(h * scale))

        delta_w = target_size - resize_width
        delta_h = target_size - resize_height

        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        result_padding = [top,bottom, left, right]
        result_pos = []

        #print("FACE INFOS ", face_infos,'scale', scale)

        for  x1, y1, x2, y2 in face_infos :
            face_width, face_height = x2 - x1, y2 - y1  # 얼굴 크기
            x3 = int(x1 * scale) + left
            y3 = int(y1 * scale) + top
            w1 = int(face_width* scale)
            h1 = int(face_height* scale)
            li = (x3,y3,w1,h1)
            result_pos.append(li)

        print("result pos :", result_pos)

        return result_pos, result_padding, scale

    def getScaleValueFromImage(self,image_shape, target_size, predictions):

        #원본이미지 w,h값
        h, w ,_= image_shape

        #print(predictions)
        #print('w', w, 'h', h)

        scale = target_size / max(h, w)

        resize_width  = (int(w * scale))
        resize_height =  (int(h * scale))

        delta_w = target_size - resize_width
        delta_h = target_size - resize_height

        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        result_padding = [top,bottom, left, right]
        result_pos = []
        for  x_v, y_v, w_v, h_v, _, _, box_color, prediction_text in predictions :

            x1 = int(x_v * scale) + left
            y1 = int(y_v * scale) + top
            w1 = int(w_v * scale)
            h1 = int(h_v * scale)
            li = (x1,y1,w1,h1,prediction_text)

            result_pos.append(li)


        return result_pos, result_padding, scale

    def _draw_results(self, image_rgb, predictions, face_cnt, male_cnt, race_cnt):
        """결과를 이미지에 그린 후 리턴"""
        font_size = max(12, int(image_rgb.shape[1] / 200))  # 폰트 크기
        image_rgb, scale, top, left = Image_Process.resize_image_with_padding(image_rgb, 512)  # 이미지 리사이즈
        #
        # 예측 결과 그리기
       # for x, y, w, h, _, _, box_color, prediction_text in predictions:
        x, y, w, h, _, _, box_color,prediction_text = predictions

        x = int(x * scale) + left
        y = int(y * scale) + top
        w = int(w * scale)
        h = int(h * scale)
        image_rgb = Image_Process.draw_korean_text(config, image_rgb, prediction_text, (x, y), 15, font_color=(0, 0, 0),
                                         background_color=box_color)
        image_rgb = cv2.rectangle(image_rgb, (x, y), (x + w, y + h), box_color, 2)
            #
        #
        info_text = f"검출된 인원 수: {face_cnt}명\n남성: {male_cnt}명\n여성: {face_cnt - male_cnt}명\n"
        #race_info = "\n".join([f"{race}: {count}명" for race, count in race_cnt.items() if count > 0])
        image_rgb = Image_Process.extend_image_with_text(config, image_rgb, info_text, font_size)
        #
        return image_rgb
        #

    #
    def _save_results(self, image_path, image_rgb, predictions):
        """결과 이미지를 저장하고 메타데이터 추가"""
        try:
            output_path = os.path.join(config['results_folder'], os.path.basename(image_path))  # 결과 이미지 경로
            cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))  # 이미지 저장
            logging.info(f"이미지 분석 결과 저장: {output_path}")
            #
            print(predictions)
            #gaka_detected = any("가카" in pred[7] for pred in predictions)  # 가카 여부
            detection_folder = "detection_target"  #if gaka_detected else "detection_non_target"  # 타겟 여부에 따라 폴더 설정
            output_folder = os.path.join(config['results_folder'], detection_folder)  # 결과 폴더
            #
            Image_Process.copy_image_and_add_metadata(image_path, output_folder)  # 이미지 복사 및 메타데이터 추가
            #
            logging.info(f"메타데이터 추가된 이미지 저장: {output_folder}")
            return output_path
        except Exception as e:
            logging.error(f"결과 저장 중 오류 발생: {e}")
            #
        #


    #




