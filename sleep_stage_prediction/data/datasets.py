import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda


class DreamtDataset(Dataset):
    DEFAULT_TRANSFORM = Compose(
        [
            torch.FloatTensor,
            Lambda(lambda x: x.permute([1, 0])),
        ]
    )

    def __init__(self, X, y, transform=DEFAULT_TRANSFORM, target_transform=None):
        super().__init__()
        self.X = X
        self.y = y
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]
        if self.transform:
            x = self.transform(self.X[index])

        if self.target_transform:
            y = self.target_transform(self.y[index])

        return x, y

    def __len__(self):
        return self.X.shape[0]


class MultiModalDreamtDataset(Dataset):
    """Dataset for multi-modal DREAMT data with sensor-specific inputs.

    Expects arrays in (C, T) format (already permuted by the preprocessing step).
    Returns (x_bvp, x_acc, x_eda_temp, x_hr, y) tuples.
    """

    def __init__(self, X_bvp, X_acc, X_eda_temp, X_hr, y, n_fft):
        super().__init__()
        self.X_bvp = X_bvp
        self.X_acc = X_acc
        self.X_eda_temp = X_eda_temp
        self.X_hr = X_hr
        self.y = y
        self.n_fft = n_fft
        self.hann_window = torch.hann_window(n_fft)

    def __getitem__(self, index):
        acc_stft = (
            torch.stft(
                torch.FloatTensor(self.X_acc[index]),
                n_fft=self.n_fft,
                hop_length=self.n_fft // 2,
                center=False,
                normalized=True,
                return_complex=True,
            ).abs()
            ** 2
        )
        log_acc_stft = 10 * torch.log(acc_stft + 1e-8)
        return (
            torch.FloatTensor(self.X_bvp[index]),
            log_acc_stft,
            torch.FloatTensor(self.X_eda_temp[index]),
            torch.FloatTensor(self.X_hr[index]),
            torch.tensor(int(self.y[index]), dtype=torch.long),
        )

    def __len__(self):
        return len(self.y)