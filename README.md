# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Image classification is a fundamental task in computer vision where an input image is assigned to one of several predefined classes. The objective of this experiment is to build and train a Convolutional Neural Network (CNN) using a labeled image dataset and evaluate its performance using accuracy, confusion matrix, and classification report.

## Neural Network Model
<img width="1165" height="737" alt="image" src="https://github.com/user-attachments/assets/147ee17a-ac80-4711-a877-f7807f3777de" />


## DESIGN STEPS
## STEP 1:

Collect and preprocess the image dataset.

## STEP 2:

Import required deep learning libraries.

## STEP 3:

Build the CNN architecture.

## STEP 4:

Train the CNN model using training data.

## STEP 5:

Evaluate the model performance using test data.

## STEP 6:

Test the model with new images and verify predictions.




## PROGRAM

### Name:Rabin R

### Register Number:212224230213

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):

        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        x = self.fc3(x)

        return x


# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Model
def train_model(model, train_loader, num_epochs=3):
   for epoch in range(num_epochs):
      model.train()
      running_loss = 0.0
      for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()


      print('Name: MOHAMMED ASHFAQ NADEEM A')
      print('Register Number:212224230166 ')
      print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
        
        
        

````
`````
### OUTPUT




## Confusion Matrix

<img width="998" height="861" alt="image" src="https://github.com/user-attachments/assets/f46a5c5b-28a9-44b4-811b-fc7fdde2d8c8" />


## Classification Report
<img width="818" height="462" alt="image" src="https://github.com/user-attachments/assets/6dddeadb-6afc-4572-84b0-0a170a300f48" />
### New Sample Data Prediction
<img width="763" height="710" alt="image" src="https://github.com/user-attachments/assets/64f09457-2221-4bdf-bfea-07ad20266614" />


## RESULT
The CNN model was successfully trained and tested, achieving accurate image classification for new input images.
