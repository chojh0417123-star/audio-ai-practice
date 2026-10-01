import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# 1. librosa로 스펙트로그램 추출
y, sr = librosa.load(librosa.ex('nutcracker'))
D = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

print(f"원본 NumPy Spectrogram Shape: {S_db.shape}") # (Frequency, Time)

# 2. PyTorch Tensor 변환 (CNN 입력 형태: [Channel, Freq, Time])
# float32 타입 텐서로 변환 후 채널 차원(Unsqueeze) 추가
tensor_spec = torch.from_numpy(S_db).float().unsqueeze(0)
print(f"PyTorch Tensor Shape (C, F, T): {tensor_spec.shape}")

# 3. 간단한 PyTorch Dataset 구축 예시
class AudioDataset(Dataset):
    def __init__(self, spec_tensor):
        self.data = spec_tensor

    def __len__(self):
        return 10  # 가상의 샘플 개수

    def __getitem__(self, idx):
        return self.data

# 4. DataLoader로 배치(Batch) 만들기
dataset = AudioDataset(tensor_spec)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch_idx, batch_data in enumerate(dataloader):
    print(f"Batch {batch_idx+1} Shape (Batch, Channel, Freq, Time): {batch_data.shape}")
    break

print("\nPyTorch 텐서 변환 및 DataLoader 세팅 성공!")
# 5. 스펙트로그램을 처리할 2D CNN 모델 정의
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        
        # 첫 번째 합성곱 레이어: [Channel 1 -> 16]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # Freq, Time 크기를 절반으로 축소
        )
        
        # 두 번째 합성곱 레이어: [Channel 16 -> 32]
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # Freq, Time 크기를 다시 절반으로 축소
        )

    def forward(self, x):
        # x shape: [Batch, 1, Freq, Time]
        x = self.conv1(x)
        x = self.conv2(x)
        return x

# 6. 모델 생성 및 데이터 통과 테스트 (Forward Pass)
model = AudioCNN()

print("\n--- 딥러닝 모델 데이터 통과 테스트 ---")
for batch_idx, batch_data in enumerate(dataloader):
    print(f"입력 데이터 Shape  : {batch_data.shape}")
    
    # 모델에 텐서 데이터 통과시키기
    output = model(batch_data)
    
    print(f"모델 출력 데이터 Shape: {output.shape}")
    break

print("\n모델 정의 및 Forward Pass 성공!")

print("\n--- 오차 계산 및 학습(Backpropagation) 테스트 ---")

# 1. 가상의 정답(Target) 데이터 생성 (동일한 크기의 텐서)
# 예: 원본을 복원하거나 특정 스펙트로그램을 예측하는 과제라고 가정
target = torch.randn_like(output) 

# 2. 손실 함수(Loss Function) 및 최적화 알고리즘(Optimizer) 정의
criterion = nn.MSELoss()  # 평균 제곱 오차 (스펙트로그램 차이 계산)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 학습 절차 1회 수행
optimizer.zero_grad()      # 1) 기울기 초기화
loss = criterion(output, target) # 2) 예측값과 정답 간의 오차(Loss) 계산
loss.backward()            # 3) 역전파로 기울기(Gradient) 계산
optimizer.step()           # 4) 모델 가중치 업데이트

print(f"계산된 Loss 값: {loss.item():.4f}")
print("역전파 및 가중치 업데이트 완료! 완벽한 PyTorch 파이프라인 완성입니다.")