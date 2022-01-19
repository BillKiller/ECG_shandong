#%%
import pickle
from torch.utils.data import DataLoader
from dataset import ECGDataset
from config import config

val_dataset = ECGDataset(data_path=config.train_data, train=False)
val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4)
for inputs, target in val_dataloader:
        print("shape", inputs.shape)
# %%
