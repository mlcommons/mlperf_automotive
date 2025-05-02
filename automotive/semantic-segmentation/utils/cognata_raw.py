import os
from collections import namedtuple
import random
from PIL import Image
import numpy as np
import torch


def prepare_cognata(root, folders, cameras):
    files = []
    for folder in folders:
        for camera in cameras:
            label_folder = os.path.join(
                root, folder, camera + '_sl_png_label_id')
            img_folder = os.path.join(root, folder, camera + '_png')
            label_files = sorted([os.path.join(label_folder, f) for f in os.listdir(
                label_folder) if os.path.isfile(os.path.join(label_folder, f))])
            img_files = sorted([os.path.join(img_folder, f) for f in os.listdir(
                img_folder) if os.path.isfile(os.path.join(img_folder, f))])
            for i in range(len(label_files)):
                files.append({'img': img_files[i], 'label': label_files[i]})

    return files


def train_val_split(files, calibration_length=200):
    random.Random(5).shuffle(files)
    val_index = round(len(files) * 0.8)
    calibration_index = len(files) - calibration_length
    return {'train': files[:val_index], 'val': files[val_index:calibration_index],
            'calibration': files[calibration_index:]}


class Cognata():
    CognataClass = namedtuple(
        'CognataClass', [
            'name', 'id', 'train_id', 'color'])
    classes = [
        CognataClass('Unlabeled', 0, 255, (0, 0, 0)),
        CognataClass('Road', 1, 0, (164, 164, 164)),
        CognataClass('Parking', 2, 255, (128, 128, 164)),
        CognataClass('Railway', 3, 255, (192, 0, 64)),
        CognataClass('LaneMarking', 4, 255, (255, 255, 255)),
        CognataClass('RoadMarkingShape', 5, 255, (0, 0, 64)),
        CognataClass('Sidewalk', 6, 1, (60, 40, 128)),
        CognataClass('BikingLane', 7, 255, (255, 255, 186)),
        CognataClass('Car', 8, 13, (255, 0, 255)),
        CognataClass('Truck', 9, 14, (192, 64, 128)),
        CognataClass('Bus', 10, 15, (228, 156, 64)),
        CognataClass('Motorcycle', 11, 17, (194, 244, 174)),
        CognataClass('Drone', 12, 255, (90, 90, 90)),
        CognataClass('Rider', 13, 12, (0, 128, 192)),
        CognataClass('Pedestrian', 14, 11, (253, 221, 206)),
        CognataClass('Tree', 15, 8, (0, 255, 0)),
        CognataClass('Low_vegetation', 16, 8, (56, 160, 56)),
        CognataClass('Bird', 17, 255, (0, 0, 255)),
        CognataClass('Terrain', 18, 9, (64, 64, 32)),
        CognataClass('Sky', 19, 10, (200, 230, 255)),
        CognataClass('Curb', 20, 255, (255, 255, 0)),
        CognataClass('Building', 21, 2, (64, 64, 96)),
        CognataClass('Building_far', 22, 2, (64, 65, 96)),
        CognataClass('Tunnel', 23, 255, (96, 64, 150)),
        CognataClass('Bridge', 24, 255, (146, 114, 200)),
        CognataClass('Fence', 25, 4, (255, 0, 0)),
        CognataClass('Guardrail', 26, 255, (255, 190, 190)),
        CognataClass('AcousticWall', 27, 3, (251, 185, 255)),
        CognataClass('Traffic_light', 28, 6, (204, 187, 227)),
        CognataClass('Props', 29, 255, (172, 72, 32)),
        CognataClass('ElectricityCable', 30, 255, (255, 187, 187)),
        CognataClass('Pole', 31, 5, (128, 192, 192)),
        CognataClass('PoleElectricLight', 32, 255, (92, 188, 255)),
        CognataClass('TrafficSign', 33, 7, (192, 128, 128)),
        CognataClass('Ego', 34, 255, (128, 255, 255)),
        CognataClass('Bicycle', 35, 18, (254, 94, 29)),
        CognataClass('Van', 36, 16, (168, 89, 153)),
        CognataClass('Lane_Line_Type_Dashed', 37, 0, (183, 177, 153)),
        CognataClass('Lane_Line_Type_Solid', 38, 0, (147, 138, 102)),
        CognataClass('Lane_Line_Type_Double_Dashed', 39, 0, (173, 137, 3)),
        CognataClass('Lane_Line_Type_Double_Solid', 40, 0, (124, 98, 2)),
        CognataClass('Lane_Line_Type_Solid_and_Dashed', 41, 0, (76, 60, 1)),
        CognataClass('Lane_Line_Type_None', 42, 0, (40, 30, 0)),
        CognataClass('Annotation', 43, 255, (50, 100, 50)),
        CognataClass('Animal', 44, 255, (160, 25, 210)),
        CognataClass('Gantry', 45, 255, (213, 213, 213)),
        CognataClass('Trailer', 46, 255, (90, 30, 60)),
        CognataClass('LargeSign', 47, 255, (120, 80, 80)),
        CognataClass('PersonalMobility', 48, 255, (120, 160, 100)),
        CognataClass('ConstructionVehicle', 49, 255, (250, 160, 0)),
        CognataClass('Rock', 50, 255, (130, 100, 70)),
        CognataClass('Stains', 51, 255, (91, 68, 45))
    ]
    train_id_to_color = [
        c.color for c in classes if (
            c.train_id != -
            1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    def __init__(self, files, transform=None):
        self.transform = transform
        self.files = files

    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    def decode_target(cls, target):
        target[target == 255] = 19
        # target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def get_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.files[index]['img']).convert('RGB')
        target = Image.open(self.files[index]['label'])
        if self.transform:
            image, target = self.transform(image, target)
        target = self.encode_target(target)
        return image, torch.from_numpy(target)


    def __len__(self):
        return len(self.files)


