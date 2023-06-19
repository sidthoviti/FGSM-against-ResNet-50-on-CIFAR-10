import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import random

def load_dataset(batch_size):
    # Set dataset path
    dataset_path = './data/cifar10'

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=2)

    # Class names for CIFAR-10 dataset
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    return trainset, trainloader, testset, testloader, classes

def train(model, trainloader, criterion, optimizer, device):
    train_loss = 0.0
    train_total = 0
    train_correct = 0

    # Switch to train mode
    model.train()

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update training loss
        train_loss += loss.item() * inputs.size(0)

        # Compute training accuracy
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    # Compute average training loss and accuracy
    train_loss = train_loss / len(trainloader.dataset)
    train_accuracy = 100.0 * train_correct / train_total

    return model, train_loss, train_accuracy

def test(model, testloader, criterion, device):
    test_loss = 0.0
    test_total = 0
    test_correct = 0

    # Switch to evaluation mode
    model.eval()

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Update test loss
            test_loss += loss.item() * inputs.size(0)

            # Compute test accuracy
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    # Compute average test loss and accuracy
    test_loss = test_loss / len(testloader.dataset)
    test_accuracy = 100.0 * test_correct / test_total

    return test_loss, test_accuracy

def train_epochs(model, trainloader, testloader, criterion, optimizer, device, num_epochs, save_interval=5):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        model, train_loss, train_accuracy = train(model, trainloader, criterion, optimizer, device)
        test_loss, test_accuracy = test(model, testloader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f'Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.2f}%')
        print(f'Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%')
        print()

        # Save the model if the current test accuracy is higher than the best accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            checkpoint = {
                'epoch' : epoch,
                'model_state_dict' : model.state_dict(),
                #'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy' : test_accuracy
            }
            torch.save(checkpoint, 'best_model.pth')

    return model, train_losses, train_accuracies, test_losses, test_accuracies

def fgsm_attack(model, criterion, images, labels, device, epsilon):
    images.requires_grad_(True)
    outputs = model(images)
    loss = criterion(outputs, labels).to(device)
    model.zero_grad()
    loss.backward()

    gradient = images.grad.data
    perturbations = epsilon * torch.sign(gradient)
    adversarial_images = images + perturbations
    adversarial_images = torch.clamp(adversarial_images, 0, 1)

    return adversarial_images, perturbations

def test_adversarial(model, testloader, criterion, device, epsilon):
    adversarial_correct = 0
    attack_success = 0
    total = 0

    model.eval()

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        adversarial_images, _ = fgsm_attack(model, criterion, images, labels, device, epsilon)

        adversarial_outputs = model(adversarial_images)

        _, adversarial_predicted = torch.max(adversarial_outputs.data, 1)

        adversarial_correct += (adversarial_predicted == labels).sum().item()
        attack_success += (adversarial_predicted != labels).sum().item()
        total += labels.size(0)

    adversarial_accuracy = 100.0 * adversarial_correct / total
    attack_success_rate = 100.0 * attack_success / total
    print(f'Epsilon = {epsilon}:')
    print(f'Adversarial Accuracy: {adversarial_accuracy:.2f}%')
    print(f'Attack Success Rate: {attack_success_rate:.2f}%')
    print('------------------------------------------------------')
    return adversarial_accuracy, attack_success_rate

def plot_loss(train_losses, test_losses):
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(test_losses)), test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()


def plot_accuracy(train_accuracies, test_accuracies):
    plt.figure()
    plt.plot(range(len(train_accuracies)), train_accuracies, label='Training Accuracy')
    plt.plot(range(len(test_accuracies)), test_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.show()

def plot_image(dataset, model, classes, device):
    idx = random.randint(0, len(dataset))
    label = dataset[idx][1]
    img = dataset[idx][0].unsqueeze(0).to(device)  # Move the input image tensor to the GPU
    model.eval()
    output = model(img)
    _, predicted = torch.max(output.data, 1)
    # Convert the image and show it
    img = img.squeeze().permute(1, 2, 0).cpu()  # Move the image tensor back to the CPU and adjust dimensions
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted: {classes[predicted]}, True: {classes[label]}')
    plt.savefig('predicted_image.png')
    plt.show()
    print("Predicted label: ", classes[predicted[0].item()])
    print("Actual label: ", classes[label])

def plot_adv_images(dataset, model, criterion, classes, device, epsilon):
    num_images = 4

    clean_images = []
    clean_labels = []
    for _ in range(num_images):
        index = random.randint(0, len(dataset))
        image, label = dataset[index]
        clean_images.append(image)
        clean_labels.append(label)

    clean_images = torch.stack(clean_images).to(device)
    clean_labels = torch.tensor(clean_labels).to(device)

    adversarial_images, perturbations = fgsm_attack(model, criterion, clean_images, clean_labels, device, epsilon)

    fig, axes = plt.subplots(num_images, 5, figsize=(15, 10))

    for i in range(num_images):
        clean_img = clean_images[i].cpu().permute(1, 2, 0).detach().numpy()
        perturbation = perturbations[i].cpu().permute(1, 2, 0).detach().numpy()
        adversarial_img = adversarial_images[i].cpu().permute(1, 2, 0).detach().numpy()

        axes[i, 0].imshow(clean_img)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f'Clean: {classes[clean_labels[i]]}', fontweight='bold', color='green')

        axes[i, 1].axis('off')
        axes[i, 1].text(0.5, 0.5, '+', fontsize=40, ha='center', va='center')
        axes[i, 1].set_title('')

        axes[i, 2].imshow(perturbation)
        axes[i, 2].axis('off')
        axes[i, 2].set_title('Perturbation')

        axes[i, 3].axis('off')
        axes[i, 3].text(0.5, 0.5, '=', fontsize=40, ha='center', va='center')
        axes[i, 3].set_title('')

        axes[i, 4].imshow(adversarial_img)
        axes[i, 4].axis('off')
        axes[i, 4].set_title(f'Adversarial: {classes[model(adversarial_images[i].unsqueeze(0)).argmax().item()]}', fontweight='bold', color='red')

    plt.tight_layout()
    plt.title('Results of Generated Adversarial Examples')
    plt.savefig('Generated_Adversarial_examples.png')
    plt.show()

def epsilon_compare(epsilon_values, adversarial_accuracies, attack_success_rates):
    if len(epsilon_values) != len(adversarial_accuracies) or len(epsilon_values) != len(attack_success_rates):
        print("Error: Input lists have different lengths.")
        return
    plt.figure(figsize=(10, 6))

    plt.plot(epsilon_values, adversarial_accuracies, 'o-', label='Adversarial Accuracy')
    plt.plot(epsilon_values, attack_success_rates, 'o-', label='Attack Success Rate')

    for i in range(len(epsilon_values)):
        plt.text(epsilon_values[i], adversarial_accuracies[i], f"{adversarial_accuracies[i]:.2f}", ha='center', va='bottom')
        plt.text(epsilon_values[i], attack_success_rates[i], f"{attack_success_rates[i]:.2f}", ha='center', va='bottom')

    plt.xlabel('Epsilon')
    plt.ylabel('Percentage')
    plt.title('Comparison of Adversarial Accuracies and Attack Success Rates')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ep_list_results')
    plt.show()

def main(train_model, epsilon_list):

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load the dataset
    num_classes = 10
    batch_size = 64
    trainset, trainloader, testset, testloader, classes = load_dataset(batch_size)

    # Load the pre-trained model
    model = models.resnet50(pretrained=True)
    # Modify conv1 to suit CIFAR-10
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Modify the final fully connected layer according to the number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Set hyperparameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_epochs = 60
    epsilon = 0.3
    epsilon_values = [0.01, 0.03, 0.07, 0.1, 0.3, 0.5]

    if train_model:
      print("Training the model...")
      # Train the model
      model, train_losses, train_accuracies, test_losses, test_accuracies = train_epochs(
          model, trainloader, testloader, criterion, optimizer, device, num_epochs)

      # Plot the loss and accuracy curves
      plot_loss(train_losses, test_losses)
      plot_accuracy(train_accuracies, test_accuracies)
      # Plot and save an example image
      plot_image(testset, model, classes, device)
      # Visualize some adversarial examples
      print("Generating Visualization Plot")
      plot_adv_images(testset, model, criterion, classes, device, epsilon)  
    else:
      # Load the best model
      best_model = models.resnet50(pretrained=True)
      # Modify conv1 to suit CIFAR-10
      best_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
      best_model.fc = nn.Linear(num_features, num_classes)
      # Load checkpoints
      checkpoint = torch.load('best_model.pth')
      best_model.load_state_dict(checkpoint['model_state_dict'])
      epoch = checkpoint['epoch']
      test_accuracy = checkpoint['test_accuracy']
      best_model = best_model.to(device)
      print("Best Trained Model Loaded!")
      print(f"Checkpoint at Epoch {epoch+1} with accuracy of {test_accuracy}%")

      # Test the best model on adversarial examples
      if epsilon_list:
        # Evaluate adversarial attacks for each epsilon value
        adversarial_accuracies = []
        attack_success_rates = []
        print("Testing with clean data again to compare with checkpoint accuracy...")
        _, clean_test_accuracy = test(best_model, testloader, criterion, device)
        print(f"Clean Adv Accuracy: {clean_test_accuracy:.2f}%\nClean Attack Success Rate: {100-clean_test_accuracy:.2f}%")
        if(clean_test_accuracy==test_accuracy):
          print("Matches with the Checkpoint Accuracy!")
        print('-----------------------------')
        print("Testing with adversarial examples...")
        for epsilon in epsilon_values:
          adversarial_accuracy, attack_success_rate = test_adversarial(best_model, testloader, criterion, device, epsilon)
          adversarial_accuracies.append(adversarial_accuracy)
          attack_success_rates.append(attack_success_rate)
        epsilon_compare(epsilon_values, adversarial_accuracies, attack_success_rates)
      else:
        clean_adversarial_accuracy, clean_attack_success_rate = test_adversarial(best_model, testloader, criterion, device, epsilon)
        print(f"Clean Adv Accuracy: {clean_adversarial_accuracy}\nClean Attack Success Rate: {clean_attack_success_rate}")
        # Visualize some adversarial examples
        print("Generating Visualization Plot")
        plot_adv_images(testset, best_model, criterion, classes, device, epsilon)



if __name__ == '__main__':
    main(train_model=True, epsilon_list=False)
    main(train_model=False, epsilon_list=True)