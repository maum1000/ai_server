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
from torchvision import models, transforms
from ultralytics import YOLO
from mtcnn import MTCNN
import time
import piexif
import io
import shutil
from pathlib import Path
import warnings
from abc import ABC, abstractmethod

class AIModel(ABC):
    @abstractmethod
    def __init__(self, model_path):
        pass

    @abstractmethod
    def predict(self, image, image_path=None):
        pass



"""Dlib 얼굴 탐지 모델 로드"""
class DlibFaceDetector(AIModel):
    def __init__(self, model_path):
        try:
            logging.info(f"Dlib 모델 로드 중: {model_path}")
            self.detector = dlib.cnn_face_detection_model_v1(model_path)
        except FileNotFoundError:
            logging.error(f"Dlib 모델 파일을 찾을 수 없습니다: {model_path}")
            self.detector = None
        except Exception as e:
            logging.error(f"Dlib 모델 로드 중 오류 발생: {e}")
            self.detector = None

    def predict(self, image):
        if self.detector is None:
            logging.error("Dlib 모델이 로드되지 않았습니다.")
            return []
        return [(d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()) for d in self.detector(image, 1)]


# =========================
# YOLO 모델 Face Detector 구현
# =========================
class YOLOFaceDetector(AIModel):
    def __init__(self, model_path):
        try:
            logging.info(f"YOLO 모델 로드 중: {model_path}")
            self.detector = YOLO(model_path)
        except FileNotFoundError:
            logging.error(f"YOLO 모델 파일을 찾을 수 없습니다: {model_path}")
            self.detector = None
        except Exception as e:
            logging.error(f"YOLO 모델 로드 중 오류 발생: {e}")
            self.detector = None
    def predict(self, image_path):
        if self.detector is None:
            logging.error("YOLO 모델이 로드되지 않았습니다.")
            return []
            #
        #
        results = self.detector.predict(image_path, conf=0.35, imgsz=1280, max_det=1000)
        return [(int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])) for result in results for box in result.boxes]


# =========================
# MTCNN 모델 Face Detector 구현
# =========================
class MTCNNFaceDetector(AIModel):
    def __init__(self):
        try:
            logging.info(f"MTCNN 모델 로드 중...")
            self.detector = MTCNN()
        except FileNotFoundError:
            logging.error(f"MTCNN 모델 파일을 찾을 수 없습니다!")
            self.detector = None
        except Exception as e:
            logging.error(f"MTCNN 모델 로드 중 오류 발생: {e}")
            self.detector = None

    def predict(self, image):
        if self.detector is None:
            logging.error("MTCNN 모델이 로드되지 않았습니다.")
            return []
        return [(f['box'][0], f['box'][1], f['box'][0] + f['box'][2], f['box'][1] + f['box'][3]) for f in self.detector.detect_faces(image)]


# =========================
# FairFace 모델 Face Predictor 구현
# =========================
class FairFacePredictor(AIModel):
    def __init__(self, model_path):
        try:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logging.info(f"FairFace 모델 load 중:\n{model_path}")
            model = models.resnet34(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, 18)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = model.to(self.device).eval()
            logging.info("FairFace 모델 load 완료")
        except Exception as e:
            logging.error(f"FairFace 모델 로드 중 오류 발생: {e}")
            self.model = None
            #
        #
    #
    def predict(self, face_image):
        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        try:
            face_image = trans(face_image).unsqueeze(0).to(self.device)
        except ValueError:
            logging.error("이미지가 너무 작거나 손상됨, 예측 건너뜀.")
            return None
            #
        #
        with torch.no_grad():
            outputs = self.model(face_image).cpu().numpy().squeeze()
        #
        race_pred = np.argmax(outputs[:4])
        gender_pred = np.argmax(outputs[7:9])
        age_pred = np.argmax(outputs[9:18])
        #
        race_text = ['백인', '흑인', '아시아', '중동'][race_pred]
        gender_text, box_color = [('남성', (50, 100, 255)), ('여성', (255, 100, 50))][gender_pred]
        age_text = ['영아', '유아', '10대', '20대', '30대', '40대', '50대', '60대', '70+'][age_pred]
        #
        return {"race": race_text, "gender": gender_text, "box_color": box_color, "age": age_text}
        #
    #
#
class FaceDetectors:
    def __init__(self):
        self.detectors = detectors

    def manage_prediction(self, image, image_path=None):
        """모든 탐지기를 사용해 얼굴을 탐지하고, 비최대 억제 적용"""
        logging.info("얼굴 탐지 시작...")
        all_faces = []
        for detector in self.detectors:
            try:
                if isinstance(detector, YOLOFaceDetector) and image_path: # YOLOFaceDetector의 경우 이미지 경로를 사용하여 탐지
                    faces =  detector.predict(image_path)
                else:
                    faces =  detector.predict(image)
            except Exception as e:
                logging.error(f"얼굴 탐지 중 오류 발생: {e}")
                raise
            finally:
                logging.info(f"{detector} : {len(faces)}개의 얼굴 검출")
                #
            #
            all_faces.extend(faces)
            #
        #
        logging.info(f"총 {len(all_faces)}개의 얼굴 검출.")
        #
        # 비최대 억제 적용
        return self._apply_non_max_suppression(all_faces)
        #
    #
    @staticmethod
    def _apply_non_max_suppression(faces):
        """
        비최대 억제를 적용하여 중복 얼굴 영역을 제거
        """
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
        #
    #
#