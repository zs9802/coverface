import cv2
import numpy as np
import sys
from tqdm import tqdm

def convert_to_square(box):
    square_box = box.copy()
    h = box[:, 3]-box[:, 1]+1
    w = box[:, 2]-box[:, 0]+1
    max_side = np.maximum(w, h)
    square_box[:, 0] = box[:, 0]+w*0.5-max_side*0.5
    square_box[:, 1] = box[:, 1]+h*0.5-max_side*0.5
    square_box[:, 2] = square_box[:, 0]+max_side-1
    square_box[:, 3] = square_box[:, 1]+max_side-1
    return square_box

def py_nms(dets,thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter+1e-10)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


class MtcnnDetector:
    def __init__(self,detectors,
                min_face_size=20,
                stride=2,
                threshold=[0.6,0.7,0.7],
                scale_factor=0.79
                ):
        self.pnet_detector=detectors[0]
        self.rnet_detector=detectors[1]
        self.onet_detector=detectors[2]
        self.min_face_size=min_face_size
        self.stride=stride
        self.thresh=threshold
        self.scale_factor=scale_factor

    def detect_face(self,test_data):
        all_boxes=[]
        landmarks=[]
        batch_idx=0
        num_of_img=test_data.size
        empty_array=np.array([])
        for databatch in tqdm(test_data):
            batch_idx+=1
            im=databatch
            if self.pnet_detector:
                boxes,boxes_c,landmark=self.detect_pnet(im)
                if boxes_c is None:
                    all_boxes.append(empty_array)
                    landmarks.append(empty_array)
                    continue
            if self.rnet_detector:
                boxes, boxes_c, landmark = self.detect_rnet(im, boxes_c)
                if boxes_c is None:
                    all_boxes.append(empty_array)
                    landmarks.append(empty_array)
                    continue
            if self.onet_detector:
                boxes, boxes_c, landmark = self.detect_onet(im, boxes_c)
                if boxes_c is None:
                    all_boxes.append(empty_array)
                    landmarks.append(empty_array)
                    continue

            all_boxes.append(boxes_c)
            landmark = [1]
            landmarks.append(landmark)
        return all_boxes, landmarks
    def detect_pnet(self,im):
        h,w,c=im.shape
        net_size=12
        current_scale=float(net_size)/self.min_face_size
        im_resized=self.processed_image(im,current_scale)
        current_height,current_width,_=im_resized.shape
        all_boxes=list()
        while min(current_height,current_width)>net_size:
            cls_cls_map,reg=self.pnet_detector.predict(im_resized)
            boxes=self.generate_bbox(cls_cls_map[:,:,1],reg,current_scale,self.thresh[0])
            current_scale*=self.scale_factor
            im_resized=self.processed_image(im,current_scale)
            current_height,current_width,_=im_resized.shape
            if boxes.size==0:
                continue
            keep=py_nms(boxes[:,:5],0.5)
            boxes=boxes[keep]
            all_boxes.append(boxes)
        if len(all_boxes)==0:
            return None,None,None
        all_boxes=np.vstack(all_boxes)
        keep = py_nms(all_boxes[:, 0:5], 0.7)
        all_boxes = all_boxes[keep]
        boxes = all_boxes[:, :5]
        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        boxes_c = boxes_c.T

        return boxes, boxes_c, None
    def detect_rnet(self,im,dets):
        h,w,c=im.shape
        dets=convert_to_square(dets)
        dets[:,0:4]=np.round(dets[:,0:4])
        [dy,edy,dx,edx,y,ey,x,ex,tmpw,tmph]=self.pad(dets,w,h)
        delete_size=np.ones_like(tmpw)*20
        ones=np.ones_like(tmpw)
        zeros=np.zeros_like(tmpw)
        num_boxes=np.sum(np.where((np.minimum(tmpw,tmph)>=delete_size),ones,zeros))
        if num_boxes==0:
            return None,None,None
        cropped_ims=np.zeros((num_boxes,24,24,3),dtype=np.float32)
        for i in range(num_boxes):
            if tmph[i]<20 or tmpw[i]<20:
                continue
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (24, 24)) - 127.5) / 128

        cls_scores, reg, _ = self.rnet_detector.predict(cropped_ims)
        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > self.thresh[1])[0]
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
        else:
            return None, None, None

        keep = py_nms(boxes, 0.6)
        boxes = boxes[keep]
        boxes_c = self.calibrate_box(boxes, reg[keep])
        return boxes, boxes_c, None

    def detect_onet(self,im,dets):
        h,w,c=im.shape
        dets=convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 48, 48, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (48, 48)) - 127.5) / 128

        cls_scores, reg, landmark = self.onet_detector.predict(cropped_ims)

        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > self.thresh[2])[0]
        if len(keep_inds) > 0:

            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None, None

        w = boxes[:, 2] - boxes[:, 0] + 1
        h = boxes[:, 3] - boxes[:, 1] + 1
        landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
        landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
        boxes_c = self.calibrate_box(boxes, reg)

        boxes = boxes[py_nms(boxes, 0.6)]
        keep = py_nms(boxes_c, 0.6)
        boxes_c = boxes_c[keep]
        landmark = landmark[keep]
        return boxes, boxes_c, landmark

    def processed_image(self, img, scale):
        height, width, channels = img.shape
        new_height = int(height * scale)
        new_width = int(width * scale)
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)
        img_resized = (img_resized - 127.5) / 128
        return img_resized

    def generate_bbox(self, cls_map, reg, scale, threshold):
        stride = 2
        cellsize = 12
        t_index = np.where(cls_map > threshold)

        if t_index[0].size == 0:
            return np.array([])

        dx1, dy1, dx2, dy2 = [reg[t_index[0], t_index[1], i] for i in range(4)]

        reg = np.array([dx1, dy1, dx2, dy2])
        score = cls_map[t_index[0], t_index[1]]
        boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                                 np.round((stride * t_index[0]) / scale),
                                 np.round((stride * t_index[1] + cellsize) / scale),
                                 np.round((stride * t_index[0] + cellsize) / scale),
                                 score,
                                 reg])
        return boundingbox.T
    def pad(self, bboxes, w, h):
        tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
        num_box = bboxes.shape[0]

        dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
        edx, edy = tmpw.copy() - 1, tmph.copy() - 1
        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        tmp_index = np.where(ex > w - 1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h - 1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list
    def calibrate_box(self, bbox, reg):
        bbox_c = bbox.copy()
        w = bbox[:, 2] - bbox[:, 0] + 1
        w = np.expand_dims(w, 1)
        h = bbox[:, 3] - bbox[:, 1] + 1
        h = np.expand_dims(h, 1)
        reg_m = np.hstack([w, h, w, h])
        aug = reg_m * reg
        bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
        return bbox_c
    def detect(self, img):
        boxes = None
        if self.pnet_detector:
            boxes, boxes_c, _ = self.detect_pnet(img)
            if boxes_c is None:
                return np.array([]), np.array([])

        if self.rnet_detector:
            boxes, boxes_c, _ = self.detect_rnet(img, boxes_c)
            if boxes_c is None:
                return np.array([]), np.array([])

        if self.onet_detector:
            boxes, boxes_c, landmark = self.detect_onet(img, boxes_c)
            if boxes_c is None:
                return np.array([]), np.array([])

        return boxes_c, landmark
