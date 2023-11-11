import os
import shutil

import math

import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader


# # 데이터 분할을 위한 폴더 생성 # #
classes_list = os.listdir('./dataset')

base_dir = './splitted'  # 나눈 데이터를 저장할 폴더
os.mkdir(base_dir)

# 나누고 나서 각 데이터를 저장할 하위 폴더들
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
valid_dir = os.path.join(base_dir, 'val')
os.mkdir(valid_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# train, val, test 폴더 내에 클래스 폴더 목록 생성
for c in classes_list:
  os.mkdir(os.path.join(train_dir, c))
  os.mkdir(os.path.join(valid_dir, c))
  os.mkdir(os.path.join(test_dir, c))

# # 데이터 분할 & 클래스별 데이터 수 확인 # #
for c in classes_list:
  path = os.path.join.(original_dataset_dir, c)
  fnames = os.listdir(path)  # path 위치의 모든 이미지 파일 목록을 저장

  # train, validation, test 비율
  train_size = math.floor(len(fnames) * 0.6)
  valid_size = math.floor(len(fnames) * 0.2)
  test_size = math.floor(len(fnames) * 0.2)

  # train 데이터
  train_fnames = fnames[:train_size]  # train_fnames에 저장
  print("Train size(", c, "): ", len(train_fnames))
  for fname in train_fnames:
    src = os.path.join(path, fname)  # 복사할 파일의 경로
    dst = os.path.join(os.path.join(train_dir, c), fname)  # 복사한 후 저장할 경로
    shutil.copyfile(src, dst)  # src 경로의 파일을 dst 경로에 저장

  validation_fnames = fnames[train_size: (validation_size + train_size)]
  print("Validation size(", c, "): ", len(validation_fnames))
  for fname in validation_fnames:
    src = os.path.join(path, fname)
    dst = os.path.join(os.path.join(valid_dir, c), fname)
    shutil.copyfile(src, dst)

  test_fnames = fnames[(train_size + validation_size): (validation_size + train_size + test_size)]
  print("Test size(", c, "): ", len(test_fnames))
  for fname in test_fnames:
    src = os.path.join(path, fname)
    dst = os.path.join(os.path.join(test_dir, c), fname)
    shutil.copyfile(src, dst)

# # 베이스라인 모델 학습을 위한 준비 # #
if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

batch_size = 256
n_epochs = 30

transform_base = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])  # 이미지 데이터 전처리, augmentation 등
# 참고: transforms.ToTensor(): 이미지를 텐서 형태로 변환하고 모든 값을 0~1 사이로 정규화

train_dataset = ImageFolder(root='./splitted/train', transform=transform_base)  # 클래스 하나가 폴더 하나에 대응되는 구조인 데이터셋 불러오기
val_dataset = ImageFolder(root='./splitted/val', transform=transform_base)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)  # 불러온 이미지를 주어진 조건에 따라 미니 배치 단위로 분리
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# # 베이스라인 모델 설계 # #
class ConvNet(nn.Module):
  def __init__(self):  # 모델에서 사용할 레이어 정의
    super(ConvNet, self).__init__()  # nn.Module의 메서드를 상속받아 사용하기

    self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

    self.fc1 = nn.Linear(4096, 512)  # 입력: 64x64 이미지 펼친 값
    self.fc2 = nn.Linear(512, 33)  # 입력: fc1의 출력 채널 수, 출력: 클래스 수

  def forward(self, x):  # forward propagation
    x = self.conv1(x)
    x = F.relu(x)
    x = self.pool(x)
    x = F.dropout(x, p=0.25, training=self.training)  # 'training=self.training': 학습 모드와 검증 모드 구분

    x = self.conv2(x)
    x = F.relu(x)
    x = self.pool(x)
    x = F.dropout(x, p=0.25, training=self.training)

    x = self.conv3(x)
    x = F.relu(x)
    x = self.pool(x)
    x = F.dropout(x, p=0.25, training=self.training)

    x = x.view(-1, 4096)  # 특징맵을 1차원으로 펼치기 (flatten)
    x = self.fc1(x)
    x = F.relu(x)
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.fc2(x)

    return F.log_softmax(x, dim=1)  # 각 클래스에 속할 확률을 softmax로 계산

model = ConvNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습 함수
def train(model, train_loader, optimizer):
  model.train()

  for batch_idx, (data, target) in enumerate(train_loader):  # train_loader에 (data, target) 형태가 미니 배치 단위로 묶여있음
    data, target = data.to(device), target.to(device)

    optimizer.zero_grad()  # optimizer 초기화: 이전 배치의 기울기 값이 optimizer에 저장되어있기 때문

    output = model(data)

    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()

# 모델 평가 함수

def evaluate(model, test_loader):
  model.eval()

  test_loss = 0  # 미니 배치별로 손실값을 합산해서 저장
  correct = 0  # 올바르게 예측한 데이터 개수

  with torch.no_grad():
    for test_data, test_target in test_loader:
      test_data, test_target = test_data.to(device), test_target.to(device)
      output = model(test_data)

      test_loss += F.cross_entropy(output, test_target, reduction='sum').item()

      test_prediction = output.max(1, keepdim=True)[1]  # 테스트 데이터가 33개 클래스에 속할 각각의 확률값 중 가장 높은 값을 가진 인덱스를 예측값으로
      correct += test_prediction.eq(test_target.view_as(test_prediction)).sum().item()

  test_loss /= len(test_loader.dataset)
  test_accuracy = 100.0 * correct / len(test_loader.dataset())

  return test_loss, test_accuracy

# 모델 학습 실행
def train_baseline(model, train_loader, val_loader, optimizer, num_epochs = 30):
  best_acc = 0.0  # 정확도가 가장 높은 모댈의 정확도
  best_model_weights = copy.deepcopy(model.state_dict())  # 정확도가 가장 높은 모델 저장

  for epoch in range(1, num_epochs + 1):
    since = time.time()  # 한 epoch당 소요되는 시간 측정

    train(model, train_loader, optimizer)  # 모델 학습
    # 해당 epoch에서의 손실값과 정확도
    train_loss, train_acc = evaluate(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)

    if val_acc > best_acc:  # 현재 epoch의 검증 정확도가 최고 정확도보다 높으면
      best_acc = val_acc  # 최고 정확도를 현재 epoch의 정확도로 업데이트
      best_model_weights = copy.deepcopy(model.state_dict())  # 그 epoch의 모델을 저장

    time_elapsed = time.time() - since  # 한 epoch당 소요된 시간: 해당 epoch이 시작할 때의 시각 - 그 epoch이 끝날 때의 시각

    print("---------------- epoch {} ----------------".format(epoch))
    print("train loss: {:.4f}, accuracy: {:.2f}%".format(train_loss, train_acc))
    print("val loss: {:.4f}, accuracy: {:.2f}%".format(val_loss, val_acc))
    print("Completed in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))

  model.load_state_dict(best_model_weights)  # 최종적으로 정확도가 가장 높은 모델 불러오기

  return model


base = train_baseline(model_base, train_loader, val_loader, optimizer, n_epochs)  # baseline 모델 학습

torch.save(base, 'baseline.pt')  # 학습된 모델 저장
