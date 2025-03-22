## Introduction
For a long time, I’ve believed that one of the most fascinating applications of machine vision lies in its potential to revolutionize food production through automation. Recently, I came across a Swiss company called [MycoSense](https://www.mycosense.ch/), which is doing exactly that. 

MycoSense develops technologies to optimize the growth and harvesting of mushrooms, aiming to enhance the efficiency of human labor. Their system involves equipping mushroom crates with cameras and using computer vision to segment mushrooms from images. This enables them to perform critical analyses, such as disease detection, determining the optimal time for harvesting, and forecasting mushroom growth. While human pickers are still employed, this technology significantly improves the efficiency of mushroom cultivation. I see this as a crucial first step toward fully automated food production. In the future, the machine vision systems developed by MycoSense could potentially guide robotic arms to pick mushrooms, eliminating the need for human labor altogether.

Inspired by this innovative use of machine vision in agriculture, I decided to train a deep learning model capable of segmenting mushrooms from images. Accurate identification and segmentation of mushrooms could enable automated systems to handle tasks like sorting, quality control, and harvesting. Such advancements have the potential to dramatically improve efficiency, reduce labor costs, and enhance food safety across the agricultural industry.

## Goal and Rationale
The primary goal of this project is to develop a robust model capable of segmenting mushrooms from images. Segmentation involves classifying each pixel in an image as either belonging to a mushroom or the background. This task is crucial for automating food production because it allows machines to "see" and interact with individual mushrooms in a controlled environment.

For example, in a mushroom farm, an automated system could use segmentation to:

1. Sort mushrooms by size and quality: By identifying the boundaries of each mushroom, the system can measure their dimensions and classify them accordingly.

2. Detect defects or contamination: Segmentation can help identify damaged or contaminated mushrooms, ensuring only high-quality produce reaches consumers.

3. Automate harvesting: Robots equipped with cameras and segmentation models could precisely locate and harvest mushrooms without human intervention.

## The dataset
To be able to train a model we need data. In fact we need data were someone has already gone through the pain-staking task of annotating the location of every mushroom in the image. A quick search on google returned this [dataset](https://universe.roboflow.com/pennsylvania-state-university-sofmp/mushroom-detection-smart-bounding-boxes), from pennsylvania state university. In this dataset, they have labeled multiple images of mushrooms by hand, with a mask for each mushroom in the image. 

Let's take a look at what some of these images look like:

{{< figure src="./images_of_mushrooms.png" title="" >}}

And here are the same images with the human-annotated labels overlayed in red. ie the red colour indicates a pixel where a mushroom is present:

{{< figure src="./images_of_mushrooms_w_masks.png" title="" >}}

## Transfer learning

One major downside of this dataset is that it is very small, with only 35 images - this could make it difficult for a model to achieve accurate and generalizable performance beyond the training data. With small datasets like this deep learning models tend to "overfit" the data, meaning that they perform well on the training data but poorly on the test data.

One way that we can get around this problem is using transfer learning. Transfer learning is a technique where a model developed for one task is reused as the starting point for a model on a second task. This approach is particularly useful when working with small datasets, as it allows the model to leverage knowledge learned from a larger, more diverse dataset. For example, a pre-trained model like ResNet or VGG, which has been trained on millions of images from the ImageNet dataset, can be fine-tuned on the smaller dataset. By doing so, the model can benefit from the general features (e.g., edges, textures, shapes) learned during the initial training, reducing the risk of overfitting and improving generalization on the smaller dataset.

In addition, transfer learning is beneficial because:

1. It reduces the need for large amounts of labeled data, which can be expensive and time-consuming to collect.

2. It allows the model to start with a strong understanding of general image features, such as edges and textures, which are also relevant for mushroom segmentation.

3. It accelerates training and improves generalization, especially when working with limited datasets.

## Model Architecture: U-Net with Transfer Learning
The project employs a U-Net architecture, a popular convolutional neural network (CNN) designed for biomedical image segmentation. U-Net is particularly well-suited for this task because of its ability to capture fine-grained details and produce high-resolution segmentation masks. The network consists of an encoder (which extracts features from the input image) and a decoder (which reconstructs the segmentation mask from the extracted features).

To improve performance, the U-Net is initialized with a pre-trained ResNet34 backbone. 

Here’s how the U-Net model is implemented in the code:

```python
import segmentation_models_pytorch as smp

# Create a U-Net with a pre-trained ResNet34 backbone
model = smp.Unet(
    encoder_name="resnet34",        # Pre-trained backbone
    encoder_weights="imagenet",     # Use ImageNet pre-trained weights
    in_channels=3,                  # Input channels (e.g., RGB)
    classes=1,                      # Output classes (e.g., binary segmentation)
    activation=None,                # No activation for BCEWithLogitsLoss
).to(device)  # Move model to the device
```

## Dataset and Training Strategy
The dataset is divided into three subsets: training, validation, and test sets. This division is critical for ensuring the model's reliability and generalizability:

1. Training Set: Used to train the model by adjusting its weights to minimize the loss function.

2. Validation Set: Used to monitor the model's performance during training and prevent overfitting. Overfitting occurs when the model learns to perform well on the training data but fails to generalize to new, unseen data.

3. Test Set: Used to evaluate the model's final performance after training. This provides an unbiased estimate of how well the model will perform in real-world scenarios.

The dataset is loaded and split into these subsets as follows:

```python
# Load dataset

train_dataset = CocoDataset(root='./data/train/', annotation='./data/train/_annotations.coco.json', transform=transforms.ToTensor())
val_dataset = CocoDataset(root='./data/valid/', annotation='./data/valid/_annotations.coco.json', transform=transforms.ToTensor())
test_dataset = CocoDataset(root='./data/test/', annotation='./data/test/_annotations.coco.json', transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
```
The model is trained using the Binary Cross-Entropy Loss (BCEWithLogitsLoss), which is suitable for binary segmentation tasks. The Adam optimizer is used to update the model's weights, with a learning rate of 0.001:


```python
# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss with logits
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

## Evaluation Metric: Intersection over Union (IoU)
The performance of the segmentation model is evaluated using the Intersection over Union (IoU) metric. IoU measures the overlap between the predicted segmentation mask and the ground truth mask. It is calculated as:



{{< math >}}
$$ IOU = \frac{Area of overlap}{Area of union}$$
{{< /math >}}
​
An IoU score of 1 indicates perfect overlap, while a score of 0 indicates no overlap. This metric is particularly useful for segmentation tasks because it provides a clear measure of how well the model is capturing the target objects.

Here’s how the IoU is calculated in the code:

```python
# Function to calculate IoU
def calculate_iou(preds, targets):
    preds = torch.sigmoid(preds) > 0.5  # Apply sigmoid before thresholding
    targets = targets > 0.5
    intersection = (preds & targets).float().sum((1, 2))
    union = (preds | targets).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6) # add very small amount to avoid divide by zero
    return iou.mean()
```

## Training and Validation
The training loop involves iterating over the dataset for a fixed number of epochs, updating the model's weights, and evaluating its performance on the validation set. The training and validation losses, as well as the IoU scores, are recorded for each epoch:

```python
# Training loop
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_iou = 0
    
    # Training phase
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device).float()
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        iou = calculate_iou(outputs, masks)
        train_loss += loss.item()
        train_iou += iou.item()
    
    # Validation phase
    model.eval()
    val_loss = 0
    val_iou = 0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            masks = masks.float()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            iou = calculate_iou(outputs, masks)
            
            val_loss += loss.item()
            val_iou += iou.item()
    
    # Print metrics for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}")
```
## Does it actually work?
It turns out that the model performs really well. In fact after just 30 epochs of training we were able to get an average IOU score of .92 on the test set. To visualise this we can look at the segmented masks on top of the image along side the ground truth labels:

{{< figure src="./images_of_mushrooms_w_predications.png" title="" >}}

In fact it looks like our model is actually able to detect some of the smaller mushrooms that were not annotated! This may mean that the model is able to generalise to out of domain images quite well.

## Future Applications
The successful segmentation of mushrooms is a step toward automating food production. In the future, this technology could be integrated into robotic systems for tasks such as:

1. Automated harvesting: Robots could use segmentation models to identify and pick mushrooms with precision.

2. Quality control: Automated systems could inspect mushrooms for defects or contamination, ensuring only high-quality produce is packaged.

3. Yield estimation: By analyzing segmentation results, farmers could estimate crop yields and optimize their production processes.


## Conclusion
One important limitation of our model is that we are only predicting whether each pixel contians a mushroom or not. To really be able to precit mushroom growth we would want to be able to assign a different label for each mushroom. This is what is called instance segmentation as opposed to semantic segmentation. One simple way to do this would be to perform watershed segmentation on the binarised model predictions to give us an individual label for each mushroom. Alternatively, we could have used a R-CNN or visual transformer such as meta's SAM-2 model which is better adapted to preforming instance segmentation.

This project demonstrates the potential of deep learning for automating food production through the segmentation of mushrooms. By leveraging a U-Net architecture with transfer learning, we achieved accurate segmentation results, as measured by the IoU metric. The use of training, validation, and test sets ensured the model's reliability and generalizability.  As the technology matures, it could revolutionize the agricultural industry, making food production more efficient, cost-effective, and sustainable. 
