# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm' or ext == '.jfif':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

def saveResult(img_file, img, polys, bboxes, adjust_width=500, base_dir='static/', dirname='results/', verticals=None, texts=None, is_rotate=False):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)
        height, width, channel = img.shape
        adjust_height = int(height / width * adjust_width)
        rotate_bboxes = bboxes
        if is_rotate:
            adjust_height = int(width / height * adjust_width)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            rotate_bboxes = []
        poly_img = img.copy()

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        if not os.path.isdir(base_dir + dirname):
            os.mkdir(base_dir + dirname)

        img_file = base_dir + dirname + filename + '.jpg'

        # drawing poly box
        for idx, polybox in enumerate(polys):
            poly = np.array(polybox).astype(np.int32).reshape((-1))
            poly = poly.reshape(-1, 2)
            if is_rotate:
                # 90도 시계 방향 회전 + 수평 이동
                poly = np.dot(poly, np.array([[0, 1], [-1, 0]])) + np.array([height, 0])
                box = np.dot((bboxes[idx]), np.array([[0, 1], [-1, 0]])) + np.array([height, 0])
                rotate_bboxes.append(box)
            cv2.polylines(poly_img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
            ptColor = (0, 255, 255)
            if verticals is not None:
                if verticals[idx]:
                    ptColor = (255, 0, 0)

            if texts is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.putText(img, "{}".format(texts[idx]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                cv2.putText(img, "{}".format(texts[idx]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)

        # Save result image

        poly_img = cv2.resize(poly_img, dsize=(adjust_width, adjust_height), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(img_file, poly_img)
        return filename + '.jpg', img, rotate_bboxes, adjust_height, adjust_width

def paginate(page, per_page, count):
    offset = (page - 1) * per_page
    total_pages = int(count / per_page) + 1
    screen_pages = 10

    if page < 1:
        page = 1
    elif page > total_pages:
        page = total_pages

    start_page = (page - 1) // screen_pages * screen_pages + 1

    pages = []
    prev_num = start_page - screen_pages
    next_num = start_page + screen_pages

    if start_page - screen_pages > 0:
        has_prev = True
    else:
        has_prev = False
    if start_page + screen_pages > total_pages:
        has_next = False
    else:
        has_next = True
    if total_pages > screen_pages + start_page:
        for i in range(screen_pages):
            pages.append(i + start_page)
    elif total_pages < screen_pages:
        for i in range(total_pages):
            pages.append(i + start_page)
    else:
        for i in range(total_pages - start_page + 1):
            pages.append(i + start_page)
    paging = {'page':page,
              'has_prev':has_prev,
              'has_next':has_next,
              'prev_num':prev_num,
              'next_num':next_num,
              'count':count,
              'offset':offset,
              'pages':pages,
              'screen_pages':screen_pages,
              'total_pages':total_pages
              }
    return paging

