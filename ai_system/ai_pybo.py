import logging
import cv2
import os

from .ai_image_process import AIImage_Process
#
from .ai_system import setup_django_system

@setup_django_system
def start_ai(request, image_path, face_recognition_system, target_encodings, selected_detectors, selected_predictors):
    # 얼굴 인식 시스템을 이용해 이미지를 처리
    output_path = face_recognition_system.process_image(image_path, target_encodings)
    #
    return output_path
    #
#

def start_ai_check_image(image_path_ori,image_path_dest=None):
   
    # 이미지 한건으로 했을 경우와 두장일때 테스트로 처리
    
    ai_process = AIImage_Process(image_path_ori)

    print(image_path_ori)
    print(image_path_dest)

    output_path = ai_process.process_image_similarity(image_path_ori, image_path_dest)
    return output_path


def start_ai_check_image_buffer(img_data):
    print("ai processing========================")
    ai_process = AIImage_Process()
    img_result = ai_process.process_detect_image(img_data)
    return img_result

