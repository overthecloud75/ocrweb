import os
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import cv2
import numpy as np

from collections import OrderedDict
from models import update_crop_image
from main import args
import utils
from models import update_image

from recognition import imgproc, craft_utils
from recognition.dataset import RawDataset, AlignCollate

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def crop_image_sort(bbox_scores, overlap=0.2):
    num_bboxes = len(bbox_scores)
    sort_list = []
    for idx in range(num_bboxes):
        local_id = None
        local_min = None
        for idy in range(num_bboxes):
            bbox_coords = bbox_scores[idy][1]
            bbox_coords = bbox_coords.astype(np.int64)

            if idy not in sort_list:
                if local_min is None:
                    local_id = idy
                    local_min = bbox_coords
                    continue
                else:
                    local_height = local_min[2][1] - local_min[1][1]
                    diff = np.min(local_min, axis=0) - np.min(bbox_coords, axis=0)
                    if diff[0] <= 0 and diff[1] <= 0:
                        pass
                    elif diff[0] <= 0 and diff[1] > 0:
                        if diff[1] < local_height * overlap:
                            pass
                        else:
                            local_id = idy
                            local_min = bbox_coords
                    elif diff[0] > 0 and diff[1] <= 0:
                        if abs(diff[1]) < local_height * overlap:
                            local_id = idy
                            local_min = bbox_coords
                        else:
                            pass
                    else:
                        local_min = bbox_coords
        sort_list.append(local_id)

    bbox_coords_list = []
    for idx in sort_list:
        bbox_coords = bbox_scores[idx][1]
        bbox_coords = bbox_coords.astype(np.int64)
        bbox_coords_list.append(bbox_coords)
    return bbox_coords_list

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

def evaluate(model, converter, data):
    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=args.imgH, imgW=args.imgW, keep_ratio_with_pad=args.PAD)

    loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    prediction = []
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([args.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, args.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in args.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in args.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                try:
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                # tensor float로 변환
                # https://stackoverflow.com/questions/57727372/how-do-i-get-the-value-of-a-tensor-in-pytorch
                    prediction.append({'pred':pred, 'confidence':round(confidence_score.item(), 3)})
                except Exception as e:
                    prediction.append({'pred':'', 'confidence':0.0})
                    print('pred_max', e, pred_max_prob.cumprod(dim=0))
    return prediction

def execute_ocr(filename, file_path, net=None, refine_net=None, model=None, converter=None):
    image = imgproc.loadImage(file_path)
    bboxes, polys, det_scores = \
        test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly,
                          args, refine_net)
    bbox_scores = []
    for box_num in range(len(bboxes)):
        bbox_scores.append((str(det_scores[box_num]), bboxes[box_num]))
    box_filename, height, width = utils.saveResult(file_path, image[:, :, ::-1], polys,
                                                   adjust_width=args.result_width, base_dir=args.static_folder, dirname=args.result_folder)
    update_image({'path':args.result_folder + box_filename, 'height':height, 'width':width})
    image = cv2.imread(file_path)

    # crop image 정렬
    bbox_coords_list = crop_image_sort(bbox_scores, overlap=args.overlap)

    # crop images
    base_dir = args.static_folder
    dirname = args.result_folder

    crop_dir = os.path.join(base_dir, dirname, filename.split('.')[0])  # static/results/20210426101605

    if os.path.isdir(crop_dir) == False:
        os.makedirs(crop_dir)

    crop_image_list = []
    request_data_list = []

    for idx, bbox_coords in enumerate(bbox_coords_list):
        if np.all(bbox_coords) > 0:
            word = crop(bbox_coords, image)
            try:
                height, width, channel = word.shape
                img_name = str(idx) + '.jpg'
                static_dir = crop_dir + '/' + img_name # static/results/20210426101605/0.jpg
                cv2.imwrite(static_dir, word)
                crop_image_list.append(static_dir)
                request_data_list.append({'path_folder':dirname + filename.split('.')[0], 'order':idx, 'name':img_name, 'height':height, 'width':width})
            except:
                continue

    data = RawDataset(crop_image_list, opt=args)  # use RawDataset

    # prediction
    prediction = evaluate(model, converter, data)
    for idx, data in enumerate(request_data_list):
        request_data = data.copy()
        request_data['pred'] = prediction[idx]['pred']
        request_data['confidence'] = prediction[idx]['confidence']
        update_crop_image(request_data)
    return prediction



