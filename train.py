import torch

# Set up GPU Environment
devive = "cuda:0"if torch.cuda.is_available else "cpu"

def train_loop(dataloader, model, loss_fn, optimizer, train_epoch):

    num_batches = len(dataloader) -1
    for batch, (x, y) in enumerate(dataloader):
        y = y.reshape(-1, 1)

        # Make prediction and compute loss
        prediction = model(x) 
        loss = loss_fn(prediction, y)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if train_epoch % 2 == 0 and batch == num_batches:
            print(f"Loss :  {loss.item():>7f}   Current Epoch : {train_epoch}")
            

            

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in dataloader:

            y = y.reshape(-1, 1)
            prediction = model(x)
            test_loss += loss_fn(prediction, y).item()

            for value in zip(y, prediction):
                correct_answer = value[0].item()
                predict_value = value[1].item()
                if predict_value > 0.7:
                    predicted_ans = 1
                else:
                    predicted_ans = 0

                if predicted_ans == correct_answer:
                    correct += 1

    total_loss = test_loss/num_batches
    accuracy = correct/size
    print(f"Test_loss : {total_loss:>7f}   |   Accuracy : {accuracy}\n")
    return test_loss

