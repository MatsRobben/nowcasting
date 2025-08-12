import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L  # modern import

# Dataset that returns [[tensor1, tensor2]]
class RandomLightingDataset(Dataset):
    def __init__(self, 
                 in_shape=(1, 4, 128, 128), 
                 out_shape=(1, 20, 128, 128), 
                 include_timestamps=False,
                 length=100):
        self.in_shape = tuple(in_shape)
        self.out_shape = tuple(out_shape)
        self.include_timestamps = include_timestamps

        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input = torch.randn(self.in_shape, dtype=torch.float32)
        output = torch.randn(self.out_shape, dtype=torch.float32)
        if self.include_timestamps:
            C, T, H, W = input.shape
            timesteps = torch.randn((T))
            return [[input, timesteps]], output
        return input, output

# Lightning DataModule
class RandomDataModule(L.LightningDataModule):
    def __init__(self, 
                 in_shape=(1, 4, 128, 128), 
                 out_shape=(1, 20, 128, 128), 
                 include_timestamps=False, 
                 batch_size=8, length=32000,
                 num_workers=0):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.include_timestamps = include_timestamps

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.length = length

    def setup(self, stage=None):
        self.train_dataset = RandomLightingDataset(self.in_shape, self.out_shape, self.include_timestamps, self.length)
        self.val_dataset   = RandomLightingDataset(self.in_shape, self.out_shape, self.include_timestamps, self.length // 5)
        self.test_dataset  = RandomLightingDataset(self.in_shape, self.out_shape, self.include_timestamps, self.length // 5)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=self.num_workers > 0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=self.num_workers > 0)

# Example usage
if __name__ == "__main__":
    dm = RandomDataModule(in_shape=(1, 4, 128, 128), 
                          out_shape=(1, 20, 128, 128), 
                          include_timestamps=False,  
                          batch_size=8)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    print(f"Outer type: {type(batch)}")
    print(f"Inner type: {type(batch[0])}")
    print(f"Shapes: {batch[0][0].shape}, {batch[0][1].shape}")
