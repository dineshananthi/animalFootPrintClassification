# import os
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms, models
# from torchvision.models.vision_transformer import ViT_B_16_Weights
#
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ]),
#     'train': transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ]),
# }
#
# data_dir = '/content/animal_fpc'
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                           data_transforms[x])
#                   for x in ['train', 'train']}
# dataloaders = {x: DataLoader(image_datasets[x], batch_size=4,
#                              shuffle=True)
#                for x in ['train', 'train']}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'train']}
# class_names = image_datasets['train'].classes
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# weights = ViT_B_16_Weights.DEFAULT
# model = models.vit_b_16(weights=weights)
# model = model.to(device)
#
# num_ftrs = model.heads[0].in_features
# model.heads[0] = nn.Linear(num_ftrs, 13)
# model = model.to(device)
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
#
#
# # Training function
# def train_and_validate_model(model, criterion, optimizer, num_epochs=50):
#     best_model_wts = model.state_dict()
#     best_acc = 0.0
#
#     for epoch in range(num_epochs):
#         for phase in ['train', 'train']:
#             if phase == 'train':
#                 model.train()  # Set model to training mode
#                 dataloader = dataloaders['train']
#             else:
#                 model.eval()  # Set model to evaluate mode
#                 dataloader = dataloaders['train']
#
#             running_loss = 0.0
#             running_corrects = 0
#
#             # Iterate over data.
#             for inputs, labels in dataloader:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#
#                 # zero the parameter gradients
#                 optimizer.zero_grad()
#
#                 # forward
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)
#
#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()
#
#                 # statistics
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
#
#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]
#
#             print(f'{phase.capitalize()} Epoch {epoch}/{num_epochs - 1} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
#
#             # Deep copy the model if it performs better on the validation set
#             if phase == 'test' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = model.state_dict()
#
#         print()
#
#     # Save the best model
#     torch.save(best_model_wts, 'vistra_animal_foot_print_class_1.pth')
#     print('vistra_animal_foot_print_class model saved.')
#
#     return model
#
#
# # Train the model
# model_trained = train_and_validate_model(model, criterion, optimizer, num_epochs=25)
