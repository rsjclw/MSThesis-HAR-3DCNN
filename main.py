import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torch
import transforms as T
from movinets import MoViNet
from movinets.config import _C
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import object_detector
from copy import deepcopy, copy
from yolox.utils import vis
from threading import Thread, Lock
from time import sleep, time
from glob import glob

transform_test = transforms.Compose([                           
                                 T.ToFloatTensorInZeroOne(),
                                 T.Resize((172, 172)),
                                 #T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                 #T.CenterCrop((172, 172))
                                 ])

class ObjectTracker:
    def __init__(self, IoUThreshold, numLostFrameThreshold):
        self.IoUThreshold = IoUThreshold
        self.numLostFrameThreshold = numLostFrameThreshold
        self.currentObjects = np.array([])
        self.currentObjectsArea = np.array([])
        self.highestID = 0

    def track(self, boxes):
        boxes = np.array(boxes)
        if len(boxes) == 0: return []
        # box format: [[xmin, ymin, xmax, ymax]]
        boxesID = []
        notMatchedBoxes = []
        if len(self.currentObjects) == 0:
            self.currentObjects = np.array(boxes)
            self.currentObjects = np.append(self.currentObjects, np.zeros((len(self.currentObjects), 2)), axis=1)
            self.currentObjects[:,5] = range(self.highestID, len(self.currentObjects))
            self.currentObjects[:,4] += 1
            self.currentObjectsArea = np.multiply(np.absolute(self.currentObjects[:,0]-self.currentObjects[:,2]), np.absolute(self.currentObjects[:,1]-self.currentObjects[:,3]))
            self.highestID = self.highestID + len(boxes)
            return np.array(range(len(boxes))), []

        matchedBoxes = set()
        for box in boxes:
            objectID = -1
            xA = np.maximum(box[0], self.currentObjects[:,0])
            yA = np.maximum(box[1], self.currentObjects[:,1])
            xB = np.minimum(box[2], self.currentObjects[:,2])
            yB = np.minimum(box[3], self.currentObjects[:,3])
            interArea = np.multiply(np.maximum(0, xB - xA + 1), np.maximum(0, yB - yA + 1))
            iou = interArea/(((box[2]-box[0]+1)*(box[3]-box[1]+1))+self.currentObjectsArea-interArea)
            idx = 0
            for index in reversed(iou.argsort()):
                if iou[index] < self.IoUThreshold: break
                if index in matchedBoxes: continue
                matchedBoxes.add(index)
                objectID = self.currentObjects[index, 5]
                self.currentObjects[index, 4] = 0
                idx = index
                break

            if objectID == -1:
                notMatchedBoxes.append(list(box))
                objectID = self.highestID+len(notMatchedBoxes)-1
                notMatchedBoxes[-1].append(0)
                notMatchedBoxes[-1].append(objectID)
                self.highestID = objectID+1
            else: self.currentObjects[idx, :4] = np.copy(box)
            boxesID.append(int(objectID))

        if len(notMatchedBoxes) > 0:
            self.currentObjects = np.append(self.currentObjects, np.array(notMatchedBoxes), axis=0)
            
        lostObjects = np.where(self.currentObjects[:,4] >= self.numLostFrameThreshold)[0]
        lostIds = self.currentObjects[lostObjects][:,5]
        self.currentObjects = np.delete(self.currentObjects, lostObjects, axis=0)
        self.currentObjects[:,4] += 1
        self.currentObjectsArea = np.multiply(np.absolute(self.currentObjects[:,0]-self.currentObjects[:,2]), np.absolute(self.currentObjects[:,1]-self.currentObjects[:,3]))
        return boxesID, lostIds

class VideoBuffer:
    def __init__(self, frames_per_clip, id):
        self.buffer = []
        self.objectId = id
        self.halfIndex = frames_per_clip//2
        self.box = []
        self.updated = False
        self.actions = [7]
        self.lock = Lock()
        self.t = -1
    
    def length(self):
        return len(self.buffer)
    
    def append(self, frame, box):
        self.lock.acquire()
        self.buffer.append(frame)
        self.lock.release()
        self.box = box

    def getItem(self):
        self.lock.acquire()
        buffer = torch.from_numpy(np.array(self.buffer, dtype=np.float32))
        self.buffer = self.buffer[self.halfIndex:]
        self.t += 1
        out = cv2.VideoWriter('v/'+str(self.t)+'.avi',cv2.VideoWriter_fourcc(*'XVID'), 6, (172,172))
        print('v/'+str(self.t)+'.avi')
        for i in self.buffer:
            out.write(i)
        out.release()
        self.lock.release()
        return buffer, self.t
    
    def appendAction(self, cls):
        self.actions.append(cls)

class RealtimeVideoInference(Dataset):
    def __init__(self, model, cam_index=0, frames_per_clip=6, step_between_clips=4, frame_size=(172,172), transform=None,
        trackerIoUThreshold=0.1, numLostFrameThreshold=6, device='gpu', classMap='sop_dataset/label_map_sop.txt'):
        self.cap = cv2.VideoCapture(cam_index)
        self.step_between_clips = step_between_clips
        self.frames_per_clip = frames_per_clip
        self.transform = transform
        self.videoBuffers = {}
        self.halfIndex = self.frames_per_clip//2
        self.detector = object_detector.Predictor(device=device)
        # self.out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 6, (640,480))
        # self.out = cv2.VideoWriter('output_newmodel_Multiobject1.avi',cv2.VideoWriter_fourcc(*'XVID'), 6, (1920,1080))
        self.frame_size = frame_size
        self.numLostFrameThreshold = numLostFrameThreshold
        self.tracker = ObjectTracker(trackerIoUThreshold, numLostFrameThreshold)
        self.trackerTextColor = (0, 0, 255)
        self.txtSize = cv2.getTextSize('123', cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0][1]*3
        self.model = model
        self.device = device
        self.class_names = []
        self.softmax = torch.nn.Softmax(dim=1)
        self.totalTime = 0
        self.totalPredict = 0
        f = open(classMap, 'r')
        for line in f:
            self.class_names.append(line[:-1])
        f.close()


    def predict(self, buffer):
        with torch.no_grad():
            data, t = buffer.getItem()
            if self.transform != None:
                data = self.transform(data)
            data = torch.from_numpy(np.expand_dims(np.array(data, dtype=np.float32), axis=0))
            if self.device == "gpu":
                data = data.cuda()
            output = F.log_softmax(self.model(data), dim=1)
            output = self.softmax(output)
            score, pred = torch.max(output, dim=1)
            if score < 0.5:
                buffer.appendAction(buffer.actions[-1])
            else:
                buffer.appendAction(int(pred[0]))
            os.system('mv v/'+str(t)+'.avi'+' '+'v/'+str(t)+'_'+str(buffer.actions[-1])+'.avi')
    def run(self):
        flag = True
        ret, frame = self.cap.read()
        boundingBoxes = []
        s = self.step_between_clips
        c = 0
        t0 = time()
        while ret:
            if s == self.step_between_clips:
                c += 1
                frameCopy = deepcopy(frame)
                cropped_frame, bboxes = self.detector.detect(frame)
                if cropped_frame is None:
                    ret, frame = self.cap.read()
                    continue

                objectIds, lostIds = self.tracker.track(bboxes)

                for i in range(len(bboxes)):
                    if objectIds[i] not in self.videoBuffers:
                        self.videoBuffers[objectIds[i]] = VideoBuffer(self.frames_per_clip, objectIds[i])
                    buf = self.videoBuffers[objectIds[i]]
                    buf.append(cv2.resize(cropped_frame[i], self.frame_size, interpolation=cv2.INTER_LINEAR), bboxes[i])
                    buf.updated = True
                    cv2.putText(frame, str(objectIds[i]), (int(bboxes[i][0]), int(bboxes[i][1]) + self.txtSize), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.trackerTextColor, thickness=2)
                    cv2.putText(frame, self.class_names[buf.actions[-1]], (int(buf.box[0]), int(buf.box[1]) + self.txtSize*3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.trackerTextColor, thickness=2)

                for i in lostIds:
                    del self.videoBuffers[i]
                keys = [key for key in self.videoBuffers]
                for key in keys:
                    buf = self.videoBuffers[key]
                    if not buf.updated:
                        buf.append(cv2.resize(frameCopy[buf.box[1]:buf.box[3], buf.box[0]:buf.box[2], :], self.frame_size, interpolation=cv2.INTER_LINEAR), buf.box)
                        cv2.putText(frame, str(buf.objectId), (int(buf.box[0]), int(buf.box[1]) + self.txtSize), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.trackerTextColor, thickness=2)
                        cv2.putText(frame, self.class_names[buf.actions[-1]], (int(buf.box[0]), int(buf.box[1]) + self.txtSize*3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.trackerTextColor, thickness=2)
                        vis(img=frame, boxes=[buf.box], scores=[buf.objectId+1], cls_ids=[0])
                    buf.updated = False

                    if buf.length() == self.frames_per_clip:
                        # t = Thread(target=self.predict, args=(buf, ))
                        # t.start()
                        # t0 = time()
                        self.predict(buf)
                        # self.totalTime += time()-t0
                        # self.totalPredict += 1

                # print([key for key in self.videoBuffers])

                # cv2.putText(frame, str(time()-t0), (int(960), int(540) + self.txtSize), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.trackerTextColor, thickness=2)
                # if c > 60:
                #     print('execution time:', time()-t0)
                #     return
                s = 0
                # self.out.write(frame)
            else:
                s += 1
            ret, frame = self.cap.read()
        print('execution time:', time()-t0)
        # print('avg_prediction_time:', self.totalTime/self.totalPredict)
    
    def release(self):
        self.cap.release()
        # self.out.release()

if __name__ == "__main__":
    device = 'cpu'
    if device == 'cpu': os.environ["CUDA_VISIBLE_DEVICES"]=""
    # video_path = 'video_data/clap/103_years_old_japanese_woman__Nao_is_clapping_with_piano_music_by_beethoven_clap_u_nm_np1_fr_med_0.avi'
    video_path = 'sop_dataset/MultiObject_2.avi'
    model = torch.load('model_state_dict_sop_.pkl', map_location=device)
    # model = torch.load('model_state_dict_sop_movinet_standardpretrained_nohmdb.pkl', map_location=device)
    camera = RealtimeVideoInference(model, video_path, transform=transform_test, device=device)
    camera.run()
    camera.release()

