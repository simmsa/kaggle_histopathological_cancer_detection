import time

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


from torch.utils.data import Dataset, DataLoader

import argparse

parser = argparse.ArgumentParser(description="Train a neural network")
parser.add_argument("--epochs", type=int, default=100, help="Number of Epochs")
parser.add_argument("--crop", type=int, default=None, help="Crop in Px")
parser.add_argument("--model", type=str, default="VGNet", help="Model")
args = parser.parse_args()

BATCH_SIZE = 256
NUM_EPOCHS = args.epochs
LOSS_RATE_CHANGE_WINDOW = 5
PADDING = 1
STRIDE = 1
# We want to exit early if the difference between epochs is small
LOSS_RATE_IMPROVEMENT_THRESHOLD = 0.00025

MODEL_TYPE = "VGNet"

if args.model == "InceptionNet":
    MODEL_TYPE = "InceptionNet"
elif args.model == "ResNet":
    MODEL_TYPE = "ResNet"


# It seems that complex doesn't help make the model more accurate, so keep this typical
if args.crop == 32:
    TRANSFORM_TYPE = "32PX_CROP"
    INPUT_SIZE = 32
elif args.crop == 48:
    TRANSFORM_TYPE = "48PX_CROP"
    INPUT_SIZE = 48
else:
    TRANSFORM_TYPE = "NONE"
    INPUT_SIZE = 96

MODEL_NAME = f"v3.model_type_{MODEL_TYPE}.transform_{TRANSFORM_TYPE}.padding_{PADDING}.stride_{STRIDE}.epochs_spec_{NUM_EPOCHS}"
print(MODEL_NAME)


MODEL_OUTPUT_PATH = Path("./final_model_output/")
MODEL_OUTPUT_PATH.mkdir(exist_ok=True)
MODEL_OUTPUT_PATH = Path(MODEL_OUTPUT_PATH, f"{MODEL_NAME}_FINAL.pth")

WORKING_MODEL_OUTPUT_PATH = Path("./working_model_output/")
WORKING_MODEL_OUTPUT_PATH.mkdir(exist_ok=True)

METRICS_OUTPUT_PATH = Path("./model_metrics_output")
METRICS_OUTPUT_PATH.mkdir(exist_ok=True)
METRICS_OUTPUT_PATH = Path(f"./model_metrics_output/{MODEL_NAME}.parquet")

TEST_CSV_OUTPUT_PATH = Path("./model_test_output")
TEST_CSV_OUTPUT_PATH.mkdir(exist_ok=True)
TEST_CSV_OUTPUT_PATH = Path(f"./model_test_output/{MODEL_NAME}.csv")


#  Device Definition ----------------------------------------------------{{{

device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)
print("Using device: ", device)

#  End Device Definition ------------------------------------------------}}}
#  Dataset Definition --------------------------------------------------{{{

# Load dataset
train_df = pd.read_csv(
    "./data/cancer_detection/histopathologic-cancer-detection/train_labels.csv"
)
# train_df = train_df.sample(n=10000, random_state=42)  # Adjust 'n' as needed
train_img_folder = Path(
    "./data/cancer_detection/histopathologic-cancer-detection/train/"
)

if TRANSFORM_TYPE == "NONE":
    transform = transforms.Compose([transforms.ToTensor()])
elif TRANSFORM_TYPE == "48PX_CROP":
    transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(48)])
elif TRANSFORM_TYPE == "32PX_CROP":
    transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(32)])
elif TRANSFORM_TYPE == "TYPICAL":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
elif TRANSFORM_TYPE == "COMPLEX":
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),  # Augmentation: random horizontal flip
            transforms.RandomRotation(15),  # Augmentation: random rotation
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2
            ),  # Augmentation: jitter
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
elif TRANSFORM_TYPE == "PCAM_SPECIFIC":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Standard normalization
            transforms.RandomHorizontalFlip(),  # Random flips
            transforms.RandomVerticalFlip(),  # Random flips
            transforms.CenterCrop(32),  # Focus on the central 32x32px region
        ]
    )
elif TRANSFORM_TYPE == "CENTER_CROP":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            # ),  # Standard normalization
            # transforms.RandomHorizontalFlip(),  # Random flips
            # transforms.RandomVerticalFlip(),  # Random flips
            transforms.CenterCrop(32),  # Focus on the central 32x32px region
        ]
    )
elif TRANSFORM_TYPE == "CENTER_CROP_NORMALIZE":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(),  # Random flips
            # transforms.RandomVerticalFlip(),  # Random flips
            transforms.CenterCrop(32),  # Focus on the central 32x32px region
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Standard normalization
        ]
    )
elif TRANSFORM_TYPE == "CENTER_CROP_NORMALIZE_COMPLEX":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.CenterCrop(32),  # Focus on the central 32x32px region
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Standard normalization
            transforms.RandomHorizontalFlip(),  # Random flips
            transforms.RandomVerticalFlip(),  # Random flips
        ]
    )

elif TRANSFORM_TYPE == "NONE":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
else:
    raise ValueError(f"Unexpected TRANSFORM_TYPE: {TRANSFORM_TYPE}")


class CancerDataset(Dataset):
    def __init__(self, dataframe, img_folder, transform=None):
        self.dataframe = dataframe
        self.img_folder = img_folder
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_id = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, 1]
        img_path = self.img_folder / f"{img_id}.tif"
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label


# Initialize dataset and dataloader
print("Creating Cancer Dataset...")
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=1234)
train_dataset = CancerDataset(train_df, train_img_folder, transform)
val_dataset = CancerDataset(val_df, train_img_folder, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print("Loading Cancer Data...")
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
)


#  End Dataset Definition ----------------------------------------------}}}
#  Model Definiton -------------------------------------------------------{{{

#  VGNet ----------------------------------------------------------------{{{


class VGNet(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super(VGNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.input_size = input_size
        self.feature_map_size = self._get_feature_map_size(input_size)

        self.fc1 = nn.Linear(self.feature_map_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_classes)

    def _get_feature_map_size(self, input_size):
        # Simulate the size reduction through layers to calculate the feature map size
        size = input_size // 2  # After pool1
        size = size // 2  # After pool2
        size = size // 2  # After pool3
        return 256 * size * size  # 256 channels in the last conv layer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#  End VGNet ------------------------------------------------------------}}}
#  InceptionNet --------------------------------------------------------{{{


class InceptionBlock(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlock, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 64, kernel_size=1)

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 64, kernel_size=1),
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        return torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], dim=1)


class InceptionNetLike(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super(InceptionNetLike, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(3, 2, padding=1)

        self.inception1 = InceptionBlock(64)
        self.inception2 = InceptionBlock(320)
        self.pool2 = nn.MaxPool2d(3, 2, padding=1)

        # Dynamically compute the flattened size for the fully connected layer
        self.input_size = input_size
        self.feature_map_size = self._get_feature_map_size(input_size)
        self.fc = nn.Linear(self.feature_map_size, num_classes)

    def _get_feature_map_size(self, input_size):
        # Simulate the size reduction through layers to calculate the feature map size
        size = (input_size + 2 * 3 - 7) // 2 + 1  # After conv1
        size = (size + 2 * 1 - 3) // 2 + 1  # After pool1
        size = (size + 2 * 1 - 3) // 2 + 1  # After pool2
        return 320 * size * size  # 320 channels in the last InceptionBlock

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.inception1(x)
        x = self.inception2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


#  End InceptionNet ----------------------------------------------------}}}
#  ResNet --------------------------------------------------------------{{{


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetLike(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super(ResNetLike, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)

        # Dynamically compute feature map size
        # self.feature_map_size = self._get_feature_map_size(input_size)
        if input_size == 32:
            # 16384
            self.feature_map_size = 32 * 512
        elif input_size == 48:
            # 36864
            self.feature_map_size = 48 * 768
        else:
            # 1478956
            self.feature_map_size = 96 * 1536

        self.fc = nn.Linear(self.feature_map_size, num_classes)

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels, 1),
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


#  End ResNet ----------------------------------------------------------}}}


#  End Model Definiton -------------------------------------------------}}}
#  Model Initialization -------------------------------------------------{{{

print(MODEL_TYPE)

if MODEL_TYPE == "VGNet":
    model = VGNet(INPUT_SIZE).to(device)
elif MODEL_TYPE == "InceptionNet":
    model = InceptionNetLike(INPUT_SIZE).to(device)
elif MODEL_TYPE == "ResNet":
    model = ResNetLike(INPUT_SIZE).to(device)
else:
    raise ValueError(f"Unexpected MODEL_TYPE: {MODEL_TYPE}")


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)

#  End Model Initialization ---------------------------------------------}}}
#  Model Training ------------------------------------------------------{{{


# def validate(model, val_loader, criterion):
#     model.eval()  # Set model to evaluation mode
#     val_loss = 0.0
#     correct = 0
#     total = 0

#     with torch.no_grad():  # No need to track gradients during evaluation
#         for images, labels in val_loader:
#             images, labels = images.to(device), labels.to(device)
#             labels = labels.float()
#             outputs = model(images)
#             loss = criterion(outputs.squeeze(), labels)
#             val_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     accuracy = 100 * correct / total
#     avg_loss = val_loss / len(val_loader)
#     print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")
#     return avg_loss, accuracy


# def validate_model(model, val_loader):
def validate(model, val_loader):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    y_pred = []
    y_actual = []

    with torch.no_grad():  # No need to track gradients during evaluation
        for images, labels in val_loader:
            y_actual.extend(labels.numpy())
            images, labels = images.to(device), labels.to(device)
            labels = labels.float()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_pred.extend(predicted.cpu().numpy())

    # Metrics
    val_accuracy = accuracy_score(y_actual, y_pred)
    val_precision = precision_score(y_actual, y_pred)
    val_recall = recall_score(y_actual, y_pred)
    val_f1 = f1_score(y_actual, y_pred)

    # Classification Report and Confusion Matrix
    class_report = classification_report(
        y_actual, y_pred, target_names=["Class 0", "Class 1"], output_dict=True
    )
    conf_matrix = confusion_matrix(y_actual, y_pred)

    # Print Metrics
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")
    print(f"Validation F1 Score: {val_f1:.4f}")

    # Print Confusion Matrix
    print("\nConfusion Matrix:")
    conf_matrix_df = pd.DataFrame(
        conf_matrix, columns=["Predicted 0", "Predicted 1"], index=["True 0", "True 1"]
    )
    print(conf_matrix_df)

    conf_matrix_df.to_parquet("./week_3_report/week_3_sample_confusion_matrix.parquet")

    # Print Classification Report as DataFrame
    class_report_df = pd.DataFrame(class_report).transpose()
    print("\nClassification Report:")
    print(class_report_df)

    class_report_df.to_parquet(
        "./week_3_report/week_3_sample_classification_report.parquet"
    )
    exit()


# Check if a prebuilt model exists
if MODEL_OUTPUT_PATH.exists():
    model.load_state_dict(
        torch.load(MODEL_OUTPUT_PATH, map_location=device, weights_only=True)
    )
    print("Prebuilt model loaded.")
else:
    print(f"Prebuilt model not found, training model {MODEL_NAME}...")

    # Arrays to track metrics
    metrics = {
        "epoch": [],
        "learning_rate": [],
        "execution_time": [],
        "training_loss": [],
        # "validation_average_loss": [],
        # "validation_accuracy": [],
    }

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        model.train()

        start_time = time.time()  # Track execution time
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float()
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs.squeeze(), labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # validate_avg_loss, validate_accuracy = validate(model, val_loader, criterion)

        # scheduler.step(validate_avg_loss)  # Update learning rate

        # End time for this epoch
        end_time = time.time()
        epoch_time = end_time - start_time

        # Calculate average training loss
        avg_loss = running_loss / len(train_loader)

        # Track metrics
        metrics["epoch"].append(epoch + 1)
        metrics["learning_rate"].append(optimizer.param_groups[0]["lr"])
        metrics["execution_time"].append(epoch_time)
        metrics["training_loss"].append(avg_loss)
        # metrics["validation_average_loss"].append(validate_avg_loss)
        # metrics["validation_accuracy"].append(validate_accuracy)

        # Check for exit conditions
        loss_rate_avg_diff = np.nan
        if len(metrics["training_loss"]) >= LOSS_RATE_CHANGE_WINDOW:
            # Get the most recent training losses in the change window
            loss_rates = metrics["training_loss"][-LOSS_RATE_CHANGE_WINDOW:]
            loss_rate_diff = np.diff(loss_rates)
            loss_rate_avg_diff = np.mean(np.abs(loss_rate_diff))

        print(
            f"Time: {epoch_time:.2f}s, "
            f"Epoch [{epoch+1:03}/{NUM_EPOCHS:03}], "
            f"Loss: {avg_loss:.4f}, "
            # f"Validate Loss: {validate_avg_loss:.4f}, "
            f"Loss Rate Avg Diff: {loss_rate_avg_diff:.8f}, "
            f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}"
        )
        this_model_output_path = Path(
            WORKING_MODEL_OUTPUT_PATH, f"{MODEL_NAME}_working_epoch_{epoch}.pth"
        )
        torch.save(model.state_dict(), this_model_output_path)

        # if loss_rate_avg_diff < LOSS_RATE_IMPROVEMENT_THRESHOLD:
        #     print("Model is below the improvement threshold")
        #     print(
        #         f"Windowed loss rate diff of {loss_rate_avg_diff} is > {LOSS_RATE_IMPROVEMENT_THRESHOLD}"
        #     )
        #     print(f"Exiting early at epoch {epoch}!")
        #     break

    # Save the trained model
    torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
    print("Model saved after training.")

    # Save metrics to a Parquet file
    metrics_df = pd.DataFrame(metrics)
    metrics_df["model"] = MODEL_NAME
    metrics_df.to_parquet(METRICS_OUTPUT_PATH)
    print(f"Training metrics saved to {METRICS_OUTPUT_PATH}")

print("Model is ready to use.")


#  End Model Training --------------------------------------------------}}}


# Split into training and validation


# validate(model, val_loader, criterion)
validate(model, val_loader)

exit()

model.eval()  # Set model to evaluation mode

# Prepare a list to store test results
results = []

test_img_folder = Path("./data/cancer_detection/histopathologic-cancer-detection/test/")

# Loop over each test image
for img_path in sorted(list(test_img_folder.rglob("*.tif"))):
    # Load image and apply transformations
    image = Image.open(img_path)
    image = (
        transform(image).unsqueeze(0).to(device)
    )  # Add batch dimension and send to device

    # Run the model to get prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)  # Get the predicted class (0 or 1)

    # Append the filename (without extension) and prediction to results
    results.append({"id": img_path.stem, "label": predicted.item()})

# Convert results to a DataFrame and save to CSV
# output_csv_path = Path(f"./all_v2_test_results_{num_epochs}_epochs.csv").resolve()
output_csv_path = TEST_CSV_OUTPUT_PATH
test_df = pd.DataFrame(results)
test_df.to_csv(output_csv_path, index=False)

print(f"Test results saved to {output_csv_path}")
