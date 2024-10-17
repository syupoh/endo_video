import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 예시 Transformer 기반 모델
class VideoAnomalyModel(nn.Module):
    def __init__(self, input_dim=512, num_heads=8, num_layers=4, hidden_dim=1024):
        super(VideoAnomalyModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.transformer(x)  # (seq_length, batch_size, input_dim)
        x = self.fc(x[-1])  # (batch_size, 1) - 마지막 시퀀스에서 예측
        return x


# MIL 기반 손실 함수
def mil_loss_function(outputs, labels):
    # MIL: 각 프레임의 출력을 통해 anomaly 여부를 예측
    max_output, _ = torch.max(outputs, dim=1)  # 각 세그먼트에서 가장 높은 점수를 선택
    loss = F.binary_cross_entropy_with_logits(max_output, labels)
    return loss

# 모든 프레임이 동일한 라벨을 가진 경우의 손실 함수
def frame_consistent_loss_function(outputs, labels):
    # 모든 프레임에 대해 동일한 라벨 적용
    loss = F.binary_cross_entropy_with_logits(outputs, labels)
    return loss


def test_and_ensemble(model_mil, model_strong, test_dataloader):
    model_mil.eval()
    model_strong.eval()
    
    all_outputs = []
    with torch.no_grad():
        for inputs in test_dataloader:
            # 모델 각각에서 예측 수행
            outputs_mil = torch.sigmoid(model_mil(inputs))
            outputs_strong = torch.sigmoid(model_strong(inputs))
            
            # 앙상블: 평균
            final_output = (outputs_mil + outputs_strong) / 2.0
            
            all_outputs.append(final_output)
    
    return torch.cat(all_outputs)


# segment label 이 일치 할때
class SegmentAnomalyDetector(nn.Module):
    def __init__(self, input_channels=3, embed_dim=512, lstm_hidden_size=256, num_layers=2):
        super(SegmentAnomalyDetector, self).__init__()

        # 3D CNN for spatial feature extraction
        self.conv3d = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(128, embed_dim, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(),
            nn.AdaptiveMaxPool3d((None, 7, 7))  # Reduce spatial dimensions to 7x7
        )

        # LSTM for temporal feature extraction
        self.lstm = nn.LSTM(embed_dim, lstm_hidden_size, num_layers=num_layers, batch_first=True)

        # Fully connected layer for segment-level classification
        self.fc = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_length, channels, height, width)
        batch_size, seq_length, c, h, w = x.size()

        # Apply 3D CNN to extract spatial features
        x = x.permute(0, 2, 1, 3, 4)  # (batch_size, channels, seq_length, height, width)
        x = self.conv3d(x)  # (batch_size, embed_dim, seq_length, 7, 7)
        x = x.flatten(3)  # Flatten spatial dimensions: (batch_size, embed_dim, seq_length, 49)
        x = x.permute(0, 2, 1).flatten(2)  # (batch_size, seq_length, embed_dim)

        # Apply LSTM to capture temporal relationships
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_length, lstm_hidden_size)

        # Use the output corresponding to the last sequence element for segment-level prediction
        out = self.fc(lstm_out[:, -1, :])  # (batch_size, 1)

        return out


# segment label 이 일치 할때
class VideoTransformer(nn.Module):
    def __init__(self, input_channels=3, embed_dim=512, num_heads=8, num_layers=4, hidden_dim=1024):
        super(VideoTransformer, self).__init__()

        # 3D CNN for spatial feature extraction
        self.conv3d = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(128, embed_dim, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(),
            nn.AdaptiveMaxPool3d((None, 7, 7))  # Reduce spatial dimensions to 7x7
        )

        # Positional encoding to capture temporal position information
        self.positional_encoding = nn.Parameter(torch.zeros(1000, embed_dim))  # Assuming max 1000 frames

        # Transformer Encoder for temporal feature extraction
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected layer for classification
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_length, channels, height, width)
        batch_size, seq_length, c, h, w = x.size()

        # Apply 3D CNN to extract spatial features
        x = x.permute(0, 2, 1, 3, 4)  # (batch_size, channels, seq_length, height, width)
        x = self.conv3d(x)  # (batch_size, embed_dim, seq_length, 7, 7)
        x = x.flatten(3)  # Flatten spatial dimensions: (batch_size, embed_dim, seq_length, 49)
        x = x.permute(2, 0, 1, 3).flatten(2)  # (seq_length, batch_size, embed_dim)

        # Add positional encoding to capture temporal position information
        x = x + self.positional_encoding[:seq_length, :]

        # Apply Transformer to capture temporal relationships
        x = self.transformer(x)  # (seq_length, batch_size, embed_dim)

        # Use the output corresponding to the last sequence element
        out = x[-1]  # (batch_size, embed_dim)

        # Classification layer
        out = self.fc(out)  # (batch_size, 1)

        return out


def main():

    # 모델 초기화
    model_mil = VideoAnomalyModel()
    model_strong = VideoAnomalyModel()

    # 옵티마이저
    optimizer_mil = torch.optim.Adam(model_mil.parameters(), lr=0.001)
    optimizer_strong = torch.optim.Adam(model_strong.parameters(), lr=0.001)

    # 훈련 루프
    num_epochs = 10

    for epoch in range(num_epochs):
        model_mil.train()
        model_strong.train()

        # MIL 모델 학습
        for inputs, labels in train_dataloader_weak:
            optimizer_mil.zero_grad()
            outputs = model_mil(inputs)
            loss = mil_loss_function(outputs, labels)
            loss.backward()
            optimizer_mil.step()

        # 모든 프레임 동일 라벨 모델 학습
        for inputs, labels in train_dataloader_strong:
            optimizer_strong.zero_grad()
            outputs = model_strong(inputs)
            loss = frame_consistent_loss_function(outputs, labels)
            loss.backward()
            optimizer_strong.step()

    # 테스트 데이터 로드 및 앙상블
    test_outputs = test_and_ensemble(model_mil, model_strong, test_dataloader)

    # 최종 이상 탐지 결과
    anomaly_predictions = (test_outputs > 0.5).float()  # 임계값 0.5로 이상 탐지



# Test the model with random input
if __name__ == "__main__":
    main()


