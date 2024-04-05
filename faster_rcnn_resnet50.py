import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model_instance_segmentation(num_classes):
    # 미리 학습된 Faster R-CNN 모델 불러오기
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # 기존의 클래스 수를 가져오기
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 새로운 클래스 수에 맞게 FastRCNNPredictor를 수정
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# 모델 인스턴스 생성 및 출력
num_classes = 49  # 변경할 클래스 수 (임의로 49로 정함 - 클래스 수에 맞게 변경 필요)
# 클래스는 기존 클래스 개수에 + 1(배경)
model = get_model_instance_segmentation(num_classes=num_classes)
print(model)