import os
import torch
import json
import numpy as np
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as T

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


class PestDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.samples = []

        for class_dir in os.listdir(self.root):
            class_path = os.path.join(self.root, class_dir)
            if os.path.isdir(class_path):
                img_folder = os.path.join(class_path, 'img')
                json_folder = os.path.join(class_path, 'json')
                if os.path.isdir(img_folder) and os.path.isdir(json_folder):
                    for file in os.listdir(img_folder):
                        if file.endswith('.jpg'):
                            image_path = os.path.join(img_folder, file)
                            label_file = file.replace('.jpg', '.json')
                            label_path = os.path.join(json_folder, label_file)
                            if os.path.isfile(label_path):
                                self.samples.append((image_path, label_path))

    def clip_coordinate(self, coord, min_val=0):
        return max(coord, min_val)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        img = read_image(img_path)

        with open(label_path, 'r') as f:
            label_data = json.load(f)

        xy_values = []
        labeling_data = label_data.get("labeling")
        for annotation in labeling_data:
            if "target" in annotation:
                target = annotation["target"]
                if "selector" in target and "value" in target["selector"]:
                    xy_value = target["selector"]["value"]
                    if xy_value.startswith("xywh=pixel:"):
                        xy_values.append(tuple(map(int, xy_value[len("xywh=pixel:"):].split(','))))

        img_shape = (label_data["info"]["Height"], label_data["info"]["Width"])

        masks = []
        boxes = []
        labels = []

        for xy_value in xy_values:
            x, y, w, h = map(self.clip_coordinate, xy_value)

            # 바운딩 박스의 크기가 0보다 큰 경우에만 추가
            if w > 0 and h > 0:
                mask = np.zeros(img_shape[:2], dtype=np.uint8)
                mask[y:y + h, x:x + w] = 1
                masks.append(mask)

                # boxes를 추가
                boxes.append([x, y, x + w, y + h])

                # labels를 추가
                labels.append(1)  # 모든 객체가 같은 라벨을 가지고 있다고 가정

        if not masks:
            masks.append(np.zeros(img_shape[:2], dtype=np.uint8))
            # 무효한 바운딩 박스를 추가
            boxes.append([0, 0, 1, 1])
            labels.append(0)

        masks = np.stack(masks, axis=0)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        img = torch.tensor(img, dtype=torch.float32) / 255.0
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        img = tv_tensors.Image(img)

        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img)),
            "masks": tv_tensors.Mask(masks),
            "labels": labels,
            "image_id": idx,
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.samples)
