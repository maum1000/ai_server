# =========================
# 이미지 처리 함수들
# =========================
#
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import shutil

import io
import piexif
import logging


class Image_Process :

    @staticmethod
    def resize_image_with_padding(image, target_size):
        """이미지를 리사이즈하고 패딩을 추가하는 함수"""
        #
        h, w, _ = image.shape
        scale = target_size / max(h, w)
        resized_img = cv2.resize(image, (int(w * scale), int(h * scale)))
        #
        delta_w = target_size - resized_img.shape[1]
        delta_h = target_size - resized_img.shape[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)


        #
        color = [0, 0, 0]
        new_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        #
        return new_img, scale, top, left
        #
    #
    @staticmethod
    def draw_korean_text(config, image, text, position, font_size, font_color=(255, 255, 255), background_color=(0, 0, 0)):
        """이미지에 한글 텍스트를 그리는 함수"""
        #
        font_path = config['font_path']
        #
        # 텍스트가 없으면 이미지 그대로 반환
        if text == '':
            return image
            #
        #
        font = ImageFont.truetype(font_path, int(font_size))
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        #
        # 텍스트 크기 측정
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        #
        # 텍스트에 맞춰 박스 크기 조정
        box_x0 = position[0] - 5
        box_y0 = position[1] - 5
        box_x1 = position[0] + text_width + 5
        box_y1 = position[1] + text_height + 5
        #
        # 이미지가 텍스트를 수용할 수 있도록 크기를 확장
        if box_x1 > image.shape[1] or box_y1 > image.shape[0]:
            new_width = max(box_x1, image.shape[1])
            new_height = max(box_y1, image.shape[0])
            extended_img = np.ones((new_height, new_width, 3), dtype=np.uint8) * 0  # 흰색 배경
            extended_img[:image.shape[0], :image.shape[1]] = image
            img_pil = Image.fromarray(extended_img)
            draw = ImageDraw.Draw(img_pil)
            #
        #
        # 배경 박스 그리기
        draw.rectangle([box_x0, box_y0, box_x1, box_y1], fill=background_color)
        #
        # 텍스트 그리기
        draw.text(position, text, font=font, fill=font_color)
        #
        return np.array(img_pil)
        #
    #
    @staticmethod
    def extend_image_with_text(config, image, text, font_size, font_color=(255, 255, 255), background_color=(0, 0, 0)):
        """이미지 확장 및 텍스트 추가 함수 (위쪽 확장)"""
        #
        font_path = config['font_path']
        # 이미지 크기 가져오기
        height, width, _ = image.shape
        #
        # 텍스트 크기 계산
        font = ImageFont.truetype(font_path, font_size)
        line_spacing = int(font_size * 1.5)
        text_lines = text.count('\n') + 1
        total_text_height = line_spacing * text_lines
        #
        # 새 이미지 생성 (텍스트를 위한 공간 + 원본 이미지)
        extended_image = np.zeros((height + total_text_height + 20, width, 3), dtype=np.uint8)  # 검은색 배경
        extended_image[total_text_height + 20:, 0:width] = image
        #
        # 텍스트 추가
        extended_image_pil = Image.fromarray(extended_image)
        draw = ImageDraw.Draw(extended_image_pil)
        draw.rectangle([(0, 0), (width, total_text_height + 20)], fill=background_color)
        draw.text((10, 10), text, font=font, fill=font_color)
        #
        return np.array(extended_image_pil)
        #
    #
    @staticmethod
    def copy_image_and_add_metadata(image_path, output_folder):
        """ 이미지 복사 및 메타데이터 추가 함수 """
        # 출력 폴더 생성
        os.makedirs(output_folder, exist_ok=True)
        # output_folder 경로에 이미지 복사
        shutil.copy(image_path, output_folder)
        # 복사된 이미지 경로
        copied_image_path = os.path.join(output_folder, os.path.basename(image_path))
        #
        # 복사된 이미지 연 후 메타데이터 추가
        with Image.open(copied_image_path) as meta_im:
            if meta_im.mode == 'RGBA':
                meta_im = meta_im.convert('RGB')
                #
            #
            thumb_im = meta_im.copy()
            o = io.BytesIO()
            thumb_im.thumbnail((50, 50), Image.Resampling.LANCZOS)
            thumb_im.save(o, "jpeg")
            thumbnail = o.getvalue()
            #
            zeroth_ifd = {
                piexif.ImageIFD.Make: u"oldcamera",
                piexif.ImageIFD.XResolution: (96, 1),
                piexif.ImageIFD.YResolution: (96, 1),
                piexif.ImageIFD.Software: u"piexif",
                piexif.ImageIFD.Artist: u"0!code",
            }
            #
            exif_ifd = {
                piexif.ExifIFD.DateTimeOriginal: u"2099:09:29 10:10:10",
                piexif.ExifIFD.LensMake: u"LensMake",
                piexif.ExifIFD.Sharpness: 65535,
                piexif.ExifIFD.LensSpecification: ((1, 1), (1, 1), (1, 1), (1, 1)),
            }
            #
            gps_ifd = {
                piexif.GPSIFD.GPSVersionID: (2, 0, 0, 0),
                piexif.GPSIFD.GPSAltitudeRef: 1,
                piexif.GPSIFD.GPSDateStamp: u"1999:99:99 99:99:99",
            }
            #
            first_ifd = {
                piexif.ImageIFD.Make: u"oldcamera",
                piexif.ImageIFD.XResolution: (40, 1),
                piexif.ImageIFD.YResolution: (40, 1),
                piexif.ImageIFD.Software: u"piexif"
            }
            #
            exif_dict = {"0th": zeroth_ifd, "Exif": exif_ifd, "GPS": gps_ifd, "1st": first_ifd, "thumbnail": thumbnail}
            exif_bytes = piexif.dump(exif_dict)
            #
            meta_im.save(copied_image_path, exif=exif_bytes)
            logging.info(f"이미지가 저장되었습니다. {copied_image_path}")
            #
        #
    #
    @staticmethod
    def print_image_exif_data(image_path):
        """ 이미지의 Exif 데이터 출력 함수 """
        with Image.open(image_path) as im:
            exif_data = piexif.load(im.info['exif'])
            print(exif_data)
            #
        #
    #
    @staticmethod
    def draw_face_boxes(image, faces, color=(0, 255, 0), thickness=2):
        """ 얼굴 주위에 바운딩박스 그리기 함수 """
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (w, h), color, thickness)
            #
        #