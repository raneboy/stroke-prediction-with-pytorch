
from torch import optim, nn
from preprocess_data import *
from neural_network import NeuralNetwork

# --------- Prepare Training Data -------------#
processed_data = process_data_to_array('./data/healthcare-dataset-stroke-data.csv')
train_set, test_set = split_dataset(processed_data)

train_dataset = CustomStrokeDataset(train_set)
test_dataset = CustomStrokeDataset(test_set)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)


# --------- Build Model -------------#

print(torch.cuda.is_available())


# --------- Hyper parameter setting -------------#
batch_size = 64
learning_rate = 1e-2
loss_fn = nn.CrossEntropyLoss()
