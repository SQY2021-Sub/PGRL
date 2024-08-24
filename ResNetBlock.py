import torch
import torch.nn as nn

# 定义ResNet块
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += shortcut
        x = self.relu(x)
        return x

# 定义模型
class ResNetLSTMModel(nn.Module):
    def __init__(self, input_size, resnet_channels, lstm_hidden_size, output_size):
        super(ResNetLSTMModel, self).__init__()
        self.resnet_x1 = ResNetBlock(input_size, resnet_channels)
        self.resnet_x2 = ResNetBlock(input_size, resnet_channels)
        self.lstm_x1 = nn.LSTM(resnet_channels, lstm_hidden_size, batch_first=True)
        self.lstm_x2 = nn.LSTM(resnet_channels, lstm_hidden_size, batch_first=True)
        self.fc_lstm = nn.Linear(lstm_hidden_size, 2)  # LSTM的输出连接到全连接层，输出2个值
        self.fc_x1_y1 = nn.Linear(3, 1)  # x1输出和y1通过全连接层
        self.fc_x2_y2 = nn.Linear(3, 1)  # x2输出和y2通过全连接层

    def forward(self, x1, x2, y1, y2):
        x1 = self.resnet_x1(x1)
        x2 = self.resnet_x2(x2)
        x1 = x1.permute(0, 2, 1)  # 调整输入形状为 (batch_size, sequence_length, channels)
        x2 = x2.permute(0, 2, 1)  # 调整输入形状为 (batch_size, sequence_length, channels)
        x1, _ = self.lstm_x1(x1)
        x2, _ = self.lstm_x2(x2)
        x1 = x1[:, -1, :]  # 取最后一个时间步的输出
        x2 = x2[:, -1, :]  # 取最后一个时间步的输出

        lstm_output_x1 = self.fc_lstm(x1)
        lstm_output_x2 = self.fc_lstm(x2)

        x1_pred_input = torch.cat((lstm_output_x1, y1), dim=1)
        x2_pred_input = torch.cat((lstm_output_x2, y2), dim=1)

        x1_pred = self.fc_x1_y1(x1_pred_input)
        x2_pred = self.fc_x2_y2(x2_pred_input)

        return torch.cat((x1_pred, x2_pred), dim=1)
