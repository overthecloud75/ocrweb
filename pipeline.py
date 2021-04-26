import os
import torch
from torch.autograd import Variable

import cv2
import numpy as np
import craft_utils
import imgproc

from collections import OrderedDict
from models import update_crop_images

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, args, refine_net=None):
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    # Post-processing
    boxes, polys, det_scores = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    return boxes, polys, det_scores

def crop(pts, image):
    """
    Takes inputs as 8 points
    and Returns cropped, masked image with a white background
    """
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    cropped = image[y:y + h, x:x + w].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(cropped, cropped, mask=mask)
    bg = np.ones_like(cropped, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst2 = bg + dst

    return dst2

def generate_words(image_name, score_bbox, image, base_dir='static/', dirname='results/'):
    num_bboxes = len(score_bbox)
    for num in range(num_bboxes):
        bbox_coords = score_bbox[num][1]
        bbox_coords = bbox_coords.astype(np.int64)
        if np.all(bbox_coords) > 0:
            word = crop(bbox_coords, image)

            # CHANGE DIR
            if os.path.isdir(os.path.join(base_dir, dirname, image_name.split('.')[0])) == False:
                os.makedirs(os.path.join(base_dir, dirname, image_name.split('.')[0]))

            try:
                height, width, channel = word.shape
                file_name = os.path.join(base_dir, dirname, image_name.split('.')[0], str(num))
                cv2.imwrite(file_name + '.jpg', word)
                request_data = {'path_folder':dirname + image_name.split('.')[0], 'order':num, 'name':str(num)+'.jpg', 'height':height, 'width':width}
                update_crop_images(request_data)
            except:
                continue
