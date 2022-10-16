import torch
from torch import optim, nn
from preprocess_data import *
from neural_network import NeuralNetwork
from train import *


# Set up GPU Environment
devive = "cuda:0"if torch.cuda.is_available else "cpu"

# --------- Prepare Training Data -------------#
processed_data = process_data_to_array('Stroke Prediction\data\healthcare-dataset-stroke-data.csv')
train_set, test_set = split_dataset(processed_data)

train_dataset = CustomStrokeDataset(train_set)
test_dataset = CustomStrokeDataset(test_set)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# --------- Build Model -------------#
model = NeuralNetwork().to(device=devive)

# --------- Hyper parameter setting -------------#
batch_size = 64
learning_rate = 1e-2
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
epochs = 1000


# --------- Train and Test -------------#

epoch_num = []
loss = []
for epoch in range(epochs):
    train_loop(train_dataloader, model, loss_fn, optimizer, epoch)
    epoch_loss = test_loop(test_dataloader, model, loss_fn)

    epoch_num.append(epoch)
    loss.append(epoch_loss)
print("Done!")



# Demonstrate and used to debug the training process
def demonstrate_train_test_process():
    for i, (x, y) in enumerate(train_dataloader):
        
        # Train
        # value = model(x)
        # y = y.reshape(-1, 1)
        # loss = loss_fn(value, y)
    
        # Test
        test_loss, correct = 0, 0
        with torch.no_grad():
            y = y.reshape(-1, 1)
            prediction = model(x)
            loss = loss_fn(prediction, y)
            test_loss += loss_fn(prediction, y).item()
            
        print(prediction)
        for value in zip(y, prediction):
            correct_answer = value[0].item()
            predict_value = value[1].item()
            if predict_value > 0.7:
                predicted_ans = 1
            else:
                predicted_ans = 0

            if predicted_ans == correct_answer:
                correct += 1
            
        print(correct)
            
        

        if i == 0:
            break
    
# demonstrate_train_test_process()
