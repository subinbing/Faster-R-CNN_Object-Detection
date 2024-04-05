import torch
import utils
from engine import train_one_epoch, evaluate
from faster_rcnn_resnet50 import get_model_instance_segmentation
from dataset_fasterrcnn import PestDataset, get_transform
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def save_checkpoint(model, optimizer, lr_scheduler, epoch, checkpoint_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")


def load_checkpoint(model, optimizer, lr_scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from epoch {epoch}")
    return model, optimizer, lr_scheduler, epoch



if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 49   # 49개 클래스 + 배경 = 48 + 1 = 49 (클래스 수에 맞게 설정)

    dataset = PestDataset('D:/fasterrcnn_pests/data', get_transform(train=True))
    dataset_test = PestDataset('D:/fasterrcnn_pests/data', get_transform(train=False))

    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-9600])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-9600:])

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.collate_fn
    )

    model = get_model_instance_segmentation(num_classes)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.0001,
        momentum=0.9,
        weight_decay=0.0005
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[5, 10],
        gamma=0.1
    )

    num_epochs = 20

    checkpoint_dir = 'D:/fasterrcnn_pests/checkpoints'    # checkpoint 파일 저장할 경로 설정
    checkpoint_file = 'checkpoint.pth'                    # checkpoint 파일
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file).replace('\\', '/')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 0
    best_score = 0

    if os.path.exists(checkpoint_path):
        model, optimizer, lr_scheduler, start_epoch = load_checkpoint(model, optimizer, lr_scheduler, checkpoint_path)

    for epoch in range(start_epoch, num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()

        coco_evaluator = evaluate(model, data_loader_test, device=device)
        stats = coco_evaluator.coco_eval['bbox'].stats

        score = stats[0] if len(stats) > 0 else 0

        if score > best_score:
            best_score = score
            epoch_checkpoint_path = checkpoint_path.replace('.pth', f'_best.pth')
            save_checkpoint(model, optimizer, lr_scheduler, epoch, epoch_checkpoint_path)

        # checkpoint 저장
        epoch_checkpoint_path = checkpoint_path.replace('.pth', f'_epoch_{epoch}.pth')
        save_checkpoint(model, optimizer, lr_scheduler, epoch, epoch_checkpoint_path)

    print("Train finish!")