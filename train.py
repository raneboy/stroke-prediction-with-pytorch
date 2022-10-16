
def train_loop(dataloader, model, loss_fn, optimizer):

    for batch, (x, y) in enumerate(dataloader):

        # Make prediction and compute loss
        prediction = model(x)
        loss = loss_fn(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


