# function to train the model
import torch
def train_util(model, trainloader,validloader, optimizer, criterion, epochs=5, device='cuda'):
  steps =0
  print_every = 60

  #moving model to gpu
  if(device=="cuda"):
    model.to(device)

  #training phase
  for epoch in range(epochs):
    running_loss=0
    for images , labels in trainloader:
      #Training Loop
      #counting batches passed
      steps+= 1
      # moving inputs to device
      images , labels = images.to(device) , labels.to(device)

      # removing gradients
      optimizer.zero_grad()
      # feeding into model
      logps=model(images)
      #calculating loss
      loss = criterion(logps, labels)
      #backpropagation
      loss.backward()
      #optimizing the weights
      optimizer.step()
      #calculating total loss of training data
      running_loss +=loss.item()


      if steps % print_every == 0:

        #evalluation loop
        #turning on eval mode
        model.eval()
        valid_loss = 0
        accuracy = 0

        for v_images , v_labels in validloader:

            v_images , v_labels = v_images.to(device) , v_labels.to(device)

            logps = model(v_images)
            loss = criterion(logps , v_labels)
            valid_loss += loss.item()

            # calculate our accuracy
            ps = torch.exp(logps)
            #getting to top ps and class and dim=1 looks along the columns
            top_ps , top_class = ps.topk(1 , dim=1)
            # checking for prediction is correct or not
            # comparing from labels array resizeing to match dimension of prediction
            equality = top_class == v_labels.view(*top_class.shape)
            accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

        print(
            f"Epoch { epoch+1}/{epochs}.."
            f" Train loss: {running_loss/print_every:.3f}.."
            f"valid Loss: {valid_loss/len(validloader):.3f}.."
            f"valid accuracy: {accuracy / len(validloader):.3f}"
        )
        model.train()