import os
import torch
import matplotlib.pyplot as plt
from torchvision.ops import box_iou
from torchvision.utils import draw_bounding_boxes
from faster_rcnn_resnet50 import get_model_instance_segmentation
from dataset_fasterrcnn import get_transform
from dataset_fasterrcnn import PestDataset

print("========= START =========")

checkpoint_path = r'D:/fasterrcnn/checkpoint.pth'    # checkpoint 파일 경로 설정
model = get_model_instance_segmentation(num_classes=49)    # 클래스 수 설정
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

true_positives = 0
false_positives = 0
false_negatives = 0

# 결과 저장할 폴더 경로
save_result_folder = 'D:/fasterrcnn_pests/result'
os.makedirs(save_result_folder, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# dataset 경로 설정
dataset = PestDataset('D:/fasterrcnn_pests/data', get_transform(train=False))
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[-9600:])    # 데이터셋 비율 설정 (test 데이터)

iou_threshold = 0.2    # iou threshold
confidence_threshold = 0.5    # confidence_threshold

for i, (images, targets) in enumerate(dataset):
    with torch.no_grad():
        images = images.to(device)
        predictions = model([images])

        pred_boxes = predictions[0]['boxes']
        pred_labels = predictions[0]['labels']
        pred_scores = predictions[0]['scores']
        image_uint8 = (images * 255).to(torch.uint8)

        confidence_mask = predictions[0]['scores'] > confidence_threshold
        pred_boxes = pred_boxes[confidence_mask]
        pred_labels = pred_labels[confidence_mask]
        pred_scores = pred_scores[confidence_mask]

        pred_labels_str = [f"{label}: {score:.2f}" for label, score in zip(pred_labels, pred_scores)]

        true_boxes = targets['boxes'].to(device)
        true_labels = targets['labels'].to(device)

        true_labels_str = [f"Ground Truth" for label in targets['labels']]

        combined_image = draw_bounding_boxes(
            image_uint8,
            torch.cat((pred_boxes, true_boxes)),
            pred_labels_str + true_labels_str,
            colors=['blue'] * len(pred_labels) + ['red'] * len(true_labels),
            width=5
        )

        for box, label in zip(pred_boxes, pred_labels_str):
            x, y, w, h = box.tolist()
            plt.annotate(label, (x, y - 2), color='blue', fontsize=40, weight='bold')

        for box, label in zip(true_boxes, [str(label) for label in true_labels]):
            x, y, w, h = box.tolist()
            plt.annotate(label, (x, y - 2), color='red', fontsize=40, weight='bold')

        image_name = f"result_image_{i}.png"
        save_path = os.path.join(save_result_folder, image_name)
        plt.imsave(save_path, combined_image.permute(1, 2, 0).cpu().numpy())

        iou = box_iou(pred_boxes, true_boxes)

        match = torch.argmax(iou, dim=1)
        iou_mask = iou[torch.arange(len(match)), match] > iou_threshold

        true_positive_mask = iou_mask & (pred_labels == true_labels[match])
        false_positive_mask = ~iou_mask
        false_negative_mask = torch.ones(len(true_boxes), dtype=torch.bool)
        false_negative_mask[match[iou_mask]] = False

        true_positives += true_positive_mask.sum().item()
        false_positives += false_positive_mask.sum().item()
        false_negatives += false_negative_mask.sum().item()
        print('true_positives: ', true_positives, '\n')
        print('false_positives: ', false_positives, '\n')
        print('false_negatives: ', false_negatives, '\n')
        print("================ FINISH =================")

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)

sum = precision + recall
mul = precision * recall
f1score = 2 * (mul / sum)

print("============ FINAL Precision and Recall ===================")
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1score:.4f}')
print("===========================================================")