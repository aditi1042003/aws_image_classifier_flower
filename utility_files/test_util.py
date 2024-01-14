import torch
#testing my model
def test_util(testloader ,model, device='cuda'):
  test_loss = 0
  accuracy = 0
  if(device=="cuda"):
    model.to(device)

  #testing model without gradients
  with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to("cuda"), labels.to("cuda")
        logps = model(inputs)
        loss = criterion(logps, labels)

        test_loss += loss.item()

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(testloader):.3f}.. "
    f"Test accuracy: {accuracy/len(testloader):.3f}")