import torch
import torch.nn as nn


class ConvolutionalClassifier(nn.Module):
    def __init__(self, channel_in, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=channel_in,
            out_channels=128,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(32)

        self.maxpool = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(
            in_channels=128,
            out_channels=64,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )

        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=32,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )

        self.conv4 = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )

        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = x.mean(dim=2)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Multi-Scale CNN
# ---------------------------------------------------------------------------

class ResBlock1d(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))

class ResBlock2d(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2)
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(
                channels, channels, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2)
            ),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))

class MultiScaleCNN(nn.Module):
    """Multi-branch CNN with residual blocks operating on sensor-specific resolutions.

    Inputs:
        x_bvp      (B, 1, 1920)  — BVP at 64 Hz
        x_acc      (B, 3, 17, 59)   — Spectogram of ACC at 32 Hz
        x_eda_temp (B, 2, 120)   — EDA+Temp at 4 Hz
        x_hr       (B, 30)       — HR at 1 Hz
    Output: (B, 5) class logits
    """

    def __init__(self):
        super().__init__()

        self.bvp_path = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=13, stride=2, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            ResBlock1d(32),
            nn.Conv1d(32, 32, kernel_size=15, stride=2, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            ResBlock1d(32),
            # nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            # nn.MaxPool1d(2),
            # ResBlock1d(128),
        )  # (B, 128, 30)

        # input (B, 3, 17, 59)
        self.acc_path = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(1, 13), stride=(1, 2), padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            ResBlock2d(32),
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 2), padding=(0, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            ResBlock2d(32),
        )  # (B, 32, 17, 33)

        # input (B, 2, 120)
        self.eda_temp_path = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, stride=2, padding=1),  # →60
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),  # →30
        )  # (B, 16, 30) — too shallow for residuals

        self.fc = nn.Sequential(
            nn.Linear(32 * 17 * 33 + 300, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 5),
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x_bvp, x_acc, x_eda_temp, x_hr):
        x_hr = x_hr
        out_bvp = self.bvp_path(x_bvp).flatten(1)
        out_acc = self.acc_path(x_acc).flatten(1)
        # out_eda_temp = self.eda_temp_path(x_eda_temp).flatten(1)
        merged = torch.cat([out_acc, x_hr], dim=1)
        return self.fc(merged)


# ---------------------------------------------------------------------------
# DeepConvLSTM
# ---------------------------------------------------------------------------

class _CNNBranch(nn.Module):
    def __init__(self, in_channels, n_filters=32, n_layers=2):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers += [
                nn.Conv1d(
                    in_channels if i == 0 else n_filters,
                    n_filters,
                    kernel_size=5,
                    padding="same",
                ),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
            ]
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.cnn(x)
        x = self.pool(x)
        return x.squeeze(-1)


class _LSTMBranch(nn.Module):
    def __init__(self, in_channels, n_filters=32, n_lstm_layers=2, hidden_size=64):
        super().__init__()
        layers = []
        for i in range(4):
            layers += [
                nn.Conv1d(
                    in_channels if i == 0 else n_filters,
                    n_filters,
                    kernel_size=5,
                    padding="same",
                ),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
            ]
        self.cnn = nn.Sequential(*layers)
        self.lstms = nn.ModuleList()
        for i in range(n_lstm_layers):
            input_size = n_filters if i == 0 else hidden_size
            self.lstms.append(
                nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
            )
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        for lstm in self.lstms:
            x, _ = lstm(x)
        return x[:, -1, :]


class DeepConvLSTM(nn.Module):
    """Reimplementation of DeepConvLSTM for multimodal wearable sleep staging.

    Reference:
        F. J. Ordóñez and D. Roggen, "Deep Convolutional and LSTM Recurrent
        Neural Networks for Multimodal Wearable Activity Recognition," Sensors,
        vol. 16, no. 1, Jan. 2016, doi: 10.3390/s16010115.

    Inputs:
        x_bvp      (B, 1, 1920)
        x_acc      (B, 3, 960)
        x_eda_temp (B, 2, 120)
        x_hr       (B, 30)
    Output: (B, n_classes) logits
    """

    def __init__(self, n_classes=5, n_filters=32, hidden_size=64):
        super().__init__()
        self.bvp_branch = _CNNBranch(in_channels=1, n_filters=n_filters)
        self.acc_branch = _LSTMBranch(in_channels=3, n_filters=n_filters, hidden_size=hidden_size)
        self.eda_branch = _CNNBranch(in_channels=2, n_filters=n_filters)

        merged_size = n_filters * 2 + hidden_size + 1
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Sequential(
            nn.Linear(merged_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x_bvp, x_acc, x_eda_temp, x_hr):
        bvp_feat = self.bvp_branch(x_bvp)
        acc_feat = self.acc_branch(x_acc)
        eda_feat = self.eda_branch(x_eda_temp)
        hr_feat = x_hr[:, 0].unsqueeze(1)
        merged = torch.cat([bvp_feat, acc_feat, eda_feat, hr_feat], dim=1)
        merged = self.dropout(merged)
        return self.dense(merged)


# ---------------------------------------------------------------------------
# MultiTCN
# ---------------------------------------------------------------------------

class Residual(nn.Module):
    """Dilated causal TCN residual block."""

    def __init__(self, input_size, hidden_size, kernel_size=3, dilation_depth=3):
        super().__init__()
        self.blocks = nn.ModuleList()
        original_input_size = input_size
        for i in range(dilation_depth):
            dilation = 2**i
            self.blocks.append(
                nn.Sequential(
                    nn.ZeroPad1d(padding=((kernel_size - 1) * dilation, 0)),
                    nn.Conv1d(
                        in_channels=input_size,
                        out_channels=hidden_size,
                        kernel_size=kernel_size,
                        dilation=dilation,
                    ),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                )
            )
            input_size = hidden_size

        self.dropout = nn.Dropout1d(0.5)
        self.align = (
            nn.Conv1d(original_input_size, hidden_size, 1)
            if original_input_size != hidden_size
            else nn.Identity()
        )

    def forward(self, x):
        residual = x
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x)
        return self.align(residual) + x


class MultiTCN(nn.Module):
    """Multi-branch Temporal Convolutional Network for multimodal sleep staging.

    Inputs:
        x_bvp      (B, 1, 1920)
        x_acc      (B, 3, 960)
        x_eda_temp (B, 2, 120)
        x_hr       (B, 30)
    Output: (B, 5) logits
    """

    _TARGET_LENGTH = 120

    def __init__(self):
        super().__init__()
        self.bc_bvp = nn.BatchNorm1d(1)
        self.bc_acc = nn.BatchNorm1d(3)
        self.bc_temp = nn.BatchNorm1d(2)
        self.bc_hr = nn.BatchNorm1d(1)

        self.bvp_block = Residual(1, 16, dilation_depth=10)
        self.acc_block = Residual(3, 16, dilation_depth=9)
        self.eda_temp_block = Residual(2, 16, dilation_depth=6)

        self.avg_bvp = nn.AvgPool1d(kernel_size=1920 // self._TARGET_LENGTH)
        self.avg_acc = nn.AvgPool1d(kernel_size=960 // self._TARGET_LENGTH)

        self.fc = nn.Sequential(
            nn.Linear(16 * self._TARGET_LENGTH * 3 + 30, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5),
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x_bvp, x_acc, x_eda_temp, x_hr):
        x_hr = x_hr.unsqueeze(1)
        x_bvp = self.bc_bvp(x_bvp)
        x_acc = self.bc_acc(x_acc)
        x_eda_temp = self.bc_temp(x_eda_temp)
        x_hr = self.bc_hr(x_hr)

        out_bvp = self.avg_bvp(self.bvp_block(x_bvp)).flatten(start_dim=1)
        out_acc = self.avg_acc(self.acc_block(x_acc)).flatten(start_dim=1)
        out_eda_temp = self.eda_temp_block(x_eda_temp).flatten(start_dim=1)

        merged = torch.cat([out_bvp, out_acc, out_eda_temp, x_hr.squeeze(1)], dim=1)
        return self.fc(merged)


class CNN(nn.Module):
    """simple CNN with a multihead attention layer for the MIT-BIH arrhythmia detection task

    Inputs:
        X (B, 1, window_len)
        RR (B, 4)
    Outputs: (B, 3) logits
    """

    def __init__(self):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=64, num_heads=4, dropout=0.2, batch_first=True
        )
        self.norm = nn.LayerNorm(64)

        self.linears = nn.Sequential(
            nn.Linear(64 * 8 + 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3),
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, X, rr):
        out = self.convolutions(X)
        out = out.permute(0, 2, 1)
        attn_out, _ = self.attention(out, out, out)
        out = self.norm(out + attn_out)
        out = torch.cat([out.flatten(1), rr], dim=1)
        return self.linears(out)
