import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from ResNet import Resnet18
from Efficientnet_B0 import My_EfficientNet_B0
from Dataset_Classification import Falling_Dataset4Clss
import torch
import os
from sklearn.metrics import accuracy_score
from torch.utils.data import  DataLoader
from torchvision.transforms  import Compose , Resize,ToTensor,RandomAffine, ColorJitter,Normalize
import  torch.nn as nn
from tqdm import  tqdm  #Tạo thanh Progress Bar
from torch.utils.tensorboard import SummaryWriter
def compute_and_save_soft_labels(model_teacher, train_dataset, save_path, batch_size=64, T=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_teacher.to(device)
    model_teacher.eval()

    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
        pin_memory=True
    )

    all_soft_labels = []

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Computing soft labels in memory"):
            images = images.to(device, non_blocking=True)
            outputs = model_teacher(images)
            soft = torch.softmax(outputs / T, dim=1)
            all_soft_labels.append(soft.cpu())
            torch.cuda.empty_cache()


    soft_labels_tensor = torch.cat(all_soft_labels, dim=0)
    torch.save(soft_labels_tensor, save_path)
    print(f"Saved soft labels to: {save_path}")


# --------- STEP 2: Train student với soft labels đã lưu ----------
def train_with_KD_soft_labels(model_student, train_dataset, test_dataloader, soft_labels_path,
                              check_point, num_epochs, pathsave_model, model_name, tensorboard_name, T=2):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_student = model_student.to(device)
    writer = SummaryWriter(log_dir=pathsave_model, comment=model_name, filename_suffix=tensorboard_name)

    optimizer = torch.optim.Adam(model_student.parameters(), lr=1e-3, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    if check_point is not None:
        checkpoint = torch.load(check_point)
        start_epoch = checkpoint['epoch']
        model_student.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        start_epoch = 0

    soft_labels = torch.load(soft_labels_path)  # (N, num_classes)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        drop_last=False,
        pin_memory=True
    )

    best_accuracy = 0

    for epoch in range(start_epoch, num_epochs):
        model_student.train()
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")

        for i, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            soft_targets = soft_labels[i * 64:(i + 1) * 64].to(device)

            outputs = model_student(images)

            ce_loss = criterion(outputs, labels)
            kd_loss = nn.KLDivLoss(reduction='batchmean')(
                torch.log_softmax(outputs / T, dim=1), soft_targets
            )
            loss = 0.85 * ce_loss + 0.15 * (T ** 2) * kd_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('Train/Loss', loss.item(), epoch * len(train_loader) + i)

        # Validation
        model_student.eval()
        all_preds, all_gts = [], []

        with torch.no_grad():
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model_student(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_gts.extend(labels.cpu().numpy())

        acc = accuracy_score(all_gts, all_preds)
        writer.add_scalar('Val/Accuracy', acc, epoch)

        # Save model
        checkpoint = {
            'epoch': epoch + 1,
            'model': model_student.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint, f'{pathsave_model}/last_cnn.pt')

        if acc > best_accuracy:
            torch.save(checkpoint, f'{pathsave_model}/best_cnn.pt')
            best_accuracy = acc

        print(f"Epoch {epoch + 1} | Accuracy: {acc:.4f} | Best: {best_accuracy:.4f}")
from torchvision import models
def make_dataset(type_data):
  if type_data == 'train':
    train_transform = Compose([
          # Biến đổi hình học
          RandomAffine(
              degrees=(-5, 5),
              translate=(0.05, 0.05),
              scale=(0.85, 1.15),  # scale quanh 1, tránh thu nhỏ quá nhiều
              shear=5              # shear nhẹ
          ),
          # Biến đổi màu sắc
          ColorJitter(
              brightness=0.1,
              contrast=0.2,
              saturation=0.2,
              hue=0.05
          ),
          Resize((224, 224)),
          ToTensor(),
          # Chuẩn hóa theo ImageNet
          Normalize(
              mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]
          )
    ])
  else:
    train_transform = Compose([
          Resize((224, 224)),
          ToTensor(),
          # Chuẩn hóa theo ImageNet
          Normalize(
              mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]
          )
    ])


  dataset = Falling_Dataset4Clss(
      root='D:\BTL_ThucTapCS\dataset_classification3',
      type_data = type_data,
      transform = train_transform
  )

  return dataset

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_vit = models.vit_b_16(pretrained=False)
    model_vit.heads.head = nn.Linear(model_vit.heads.head.in_features, 2)

    # 2. Load checkpoint và trích xuất state_dict
    checkpoint = torch.load('D:\BTL_ThucTapCS\VIT_Finetuning/best_cnn.pt', map_location=device)
    model_vit.load_state_dict(checkpoint['model'])
    train_dataset = make_dataset('train')
    # 4. Đưa model sang thiết bị và chuyển sang eval mode
    model_vit = model_vit.to(device)
    model_vit.eval()
    # compute_and_save_soft_labels(model_vit, train_dataset,
    #                              'D:\BTL_ThucTapCS/soft_labels.pt', batch_size=64,
    #                              T=2)

    test_dataset = make_dataset('valid')
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        num_workers=4,
        drop_last=False,
        pin_memory=True

    )
    model_effi = My_EfficientNet_B0(num_classes=2)
    checkpoint = torch.load('D:\BTL_ThucTapCS\Model_Classification/no_KD\efficientnet/best_cnn.pt',
                            map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model_effi.load_state_dict(checkpoint['model'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model_effi.to(device)
    model_effi.eval()
    train_with_KD_soft_labels(
        model_student = model_effi,
        train_dataset = train_dataset,
        test_dataloader = test_dataloader,
        soft_labels_path ='D:\BTL_ThucTapCS/soft_labels.pt',
        check_point = None,
        num_epochs = 20,
        pathsave_model = 'D:\BTL_ThucTapCS\Model_Classification\KD',
        model_name = 'Effi_with_KD',
        tensorboard_name= 'run1',
        T=2
    )