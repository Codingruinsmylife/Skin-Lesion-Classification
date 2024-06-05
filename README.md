# Skin-Lesion-Classification
Theis project on skin lesion classification using deep learning involves developing a robust model to accurately diagnose various skin conditions from image inputs. The primary objective is to classify skin lesions into multiple categories, including both benign and malignant types, using a deep learning techniques. The project entails collecting and preprocessing a comprehensive dataset of labeled skin lesion images, followed by implementing a deep learning architecture, such as ResNet-50, to learn and distinguish between different lesion types. Advanced techniques like data augmentation and class balancing are employed to enhance model performance and generalization. The model is trained and fine-tuned to achieve high accuracy, and its efficacy is evaluated through rigorous testing. The ultimate goal is to provide a reliable tool for dermatologists, aiding in early and precise diagnosis of skin diseases, potentially improving patient outcomes through timely medical intervention.

## Overview
The skin lesion classification project aims to develop a sophisticated deep learning model to accurately diagnose various skin conditions from image inputs. Utilizing a convolutional neural network (CNN) architecture, specifically ResNet-50, the model is trained on a diverse dataset of labeled skin lesion images to classify lesions into multiple categories, including benign and malignant types. The project involves extensive data preparation, including combining images from different sources, data augmentation, and class balancing, to enhance the model's performance. The workflow encompasses importing necessary libraries, mounting Google Drive for data access, preprocessing images, and training the model with fine-tuning for optimal accuracy. The ultimate goal is to provide a reliable and automated tool for dermatologists, aiding in the early detection and precise diagnosis of skin diseases, thereby improving diagnostic efficiency and patient outcomes in clinical settings.

## Project Objectives
The following are the main objectives of this project:
1. **Develop a High-Accuracy Deep Learning Model for Skin Lesion Classification**<br>
The primary objective of this project is to create a deep learning model, specifically using the ResNet-50 architecture, to accurately classify skin lesions into various categories, such as benign and malignant types. This involves extensive training on a labeled dataset of skin lesion images, applying advanced techniques like data augmentation, and optimizing the model through fine-tuning. Achieving high accuracy is crucial to ensure the model's reliability and effectiveness in clinical settings, where it can assist dermatologists in making informed diagnostic decisions.
2. **Implement Comprehensive Data Preparation and Augmentation Techniques**<br>
Another key objective is to meticulously prepare the dataset to enhance the model's performance. This includes combining images from multiple sources into a single dataset, handling class imbalances through sampling methods, and applying various data augmentation techniques to increase the diversity of the training data. Proper data preparation ensures that the model can generalize well to new, unseen data, which is essential for its practical application in real-world scenarios.
3. **Provide a Reliable Diagnostic Tool to Assist Dermatologists**<br>
The final objective is to develop an automated tool that can be used by dermatologists to aid in the early detection and diagnosis of skin diseases. By integrating the trained model into a user-friendly interface, the tool can analyze skin lesion images and provide diagnostic suggestions, thereby enhancing diagnostic efficiency and accuracy. This tool aims to support dermatologists in clinical settings, reduce diagnostic errors, and ultimately improve patient outcomes by facilitating timely and accurate treatment interventions.

## Key Components
1. **Google Drive Mounting and Directory Setup (Only applicable to Google Colab)**<br>
This section of the code is responsible for connecting to Google Drive, which provides a convenient way to store and access files in a cloud environment. Mounting Google Drive allows the Colab notebook to interact with files stored in the user’s Google Drive account, making it possible to read datasets and save models directly to a persistent storage location. The drive.mount('/content/drive') command mounts the drive, and the cd /content/drive/MyDrive/UCCD3074_Labs/Assignment2 command changes the working directory to a specific folder where the project files are located. This setup ensures that subsequent file operations are correctly directed to the desired directory, facilitating seamless data management and storage.
2. **Library Imports**<br>
This section imports all the necessary libraries required for various stages of the project. Libraries such as pandas and numpy are essential for data manipulation and numerical operations. PyTorch (torch, torch.nn, torch.optim, torchvision.models) provides the deep learning framework needed for building, training, and evaluating the neural network. matplotlib.pyplot and seaborn are used for data visualization, helping to understand the data distribution and model performance. The sklearn.metrics module provides tools for evaluating the model, such as generating classification reports and confusion matrices. The PIL library is used for image processing tasks. Importing these libraries at the beginning ensures that all the necessary functions and classes are available for the subsequent code.
3. **Data Preparation**<br>
The data preparation phase involves several critical steps to ensure that the dataset is ready for training the model. The combine_folders function is used to consolidate images from different source directories into a single destination directory. This step is crucial for organizing the dataset in a manner that facilitates easy access and processing. After combining the images, the code checks for and handles any missing or 'unknown' values in the dataset. Dropping rows with null values in the 'age' column ensures that the data fed into the model is complete and reliable. Additionally, rows with 'unknown' values in any attribute are removed to maintain data quality. These preprocessing steps are fundamental to ensuring the integrity and consistency of the dataset, which directly impacts the model's performance.
4. **Class Balancing**<br>
Class imbalance is a common issue in classification tasks where some classes have significantly more samples than others. This can lead to biased model performance, where the model performs well on majority classes but poorly on minority classes. To address this, the code resamples the dataset to ensure that each class has an equal number of samples. The SAMPLE_PER_GROUP variable defines the target number of samples per class. The groupby method is used to group the DataFrame by the 'dx' column (which represents different skin lesion classes), and the apply method is used to resample each group to the desired size, with replacement if necessary. This class balancing step is essential for creating a more uniform dataset, which helps the model learn to distinguish between all classes more effectively and improves overall classification performance.
5. **Model Definition and Training**<br>
In this component, the ResNet-50 model is defined and customized for the specific classification task. ResNet-50, a deep convolutional neural network, is known for its effectiveness in image classification tasks. The models.resnet50(pretrained=False) command initializes the model without pre-trained weights, allowing for training from scratch on the specific dataset. The final fully connected layer is modified to match the number of classes in the dataset (in this case, 7 classes). The training process involves defining the loss function (e.g., cross-entropy loss) and the optimizer (e.g., Adam or SGD). The model is set to training mode using model.train(), and an iterative process updates the model weights based on the training data. Each epoch involves feeding batches of images through the model, calculating the loss, and backpropagating the error to adjust the weights. This training loop continues until the model converges to an optimal set of weights that minimize the loss function.
6. **Model Saving**<br>
After training the model, it is essential to save the trained model's state so that it can be reused without retraining from scratch. The torch.save(model.state_dict(), save_path) command saves the model's state dictionary, which contains all the parameters (weights and biases) of the model. The save_path specifies the location where the model will be stored, ensuring that the trained model can be easily accessed and loaded later. Saving the model is a critical step in the workflow, as it allows for persistence of the training effort and facilitates future use in inference or further training.

## Why ResNet-50?
ResNet-50 is a widely used convolutional neural network architecture known for its significant depth and effectiveness in image classification tasks. Introduced by Microsoft Research, ResNet-50 is part of the Residual Network (ResNet) family, characterized by its use of residual blocks with skip connections. These connections allow the model to learn residual functions instead of direct mappings, effectively addressing the vanishing gradient problem and enabling the training of very deep networks. ResNet-50 comprises 50 layers, including convolutional layers, batch normalization, ReLU activation functions, and pooling layers, culminating in a fully connected layer for classification. Its ability to maintain high accuracy and reduce computational complexity has made ResNet-50 a popular choice for various computer vision applications, including image recognition and feature extraction.
1. **Depth and Learning Capacity**<br>
ResNet-50, with its 50 layers, offers a substantial depth that allows it to learn complex features from images. This depth is critical for capturing intricate patterns and variations in data, which is essential for tasks like skin lesion classification that require fine-grained analysis to distinguish between benign and malignant lesions. The residual blocks in ResNet-50 facilitate the training of such a deep network by mitigating the vanishing gradient problem, ensuring that the network can learn effectively even as it grows deeper. This balance between depth and effective training makes ResNet-50 particularly suitable for complex image classification tasks.
2. **Residual Learning and Skip Connections**<br>
The key innovation of ResNet-50 is its use of residual learning through skip connections. These connections allow the model to bypass one or more layers, effectively enabling the network to learn residuals (differences) rather than direct mappings. This approach helps in maintaining the flow of gradients during backpropagation, which is crucial for training very deep networks. By addressing the degradation problem (where adding more layers to a deep network leads to higher training error), ResNet-50 ensures more stable and faster convergence, leading to better performance and generalization on new data compared to traditional deep networks without residual connections.
3. **Proven Performance and Versatility**<br>
ResNet-50 has been extensively tested and validated in numerous studies and competitions, consistently demonstrating high performance in various image classification benchmarks, including the ImageNet Large Scale Visual Recognition Challenge (ILSVRC). Its architecture has become a standard baseline in computer vision due to its robustness and ability to generalize well across different datasets and tasks. For skin lesion classification, a task requiring high precision and accuracy, ResNet-50's proven track record provides confidence in its ability to deliver reliable results. Furthermore, its architecture is versatile and can be fine-tuned for specific applications, making it an adaptable choice for various image-related tasks beyond classification, such as object detection and segmentation.

## Getting Started
To get started with the Skin Lesion Classification project, follow these steps to set up the environment, prepare the data, and train the model.
### Prerequisites
1. **Google Colab**: This project uses Google Colab for its computational resources. Ensure you have a Google account to access Colab. You may use other cell-based IDE such as Jupyter Notebook as well.
2. **Google Drive**: The project leverages Google Drive for data storage and access. Make sure you have sufficient space in your Google Drive.
3. **Python Libraries**: The necessary libraries include pandas, numpy, torch, os, shutil, matplotlib, seaborn, sklearn, torchvision, and PIL. These will be installed as part of the Colab environment setup.
### Setting Up the Environment
1. **Clone the Repository**: Start by cloning the project repository to your local machine or Google Drive.
```
git clone https://github.com/yourusername/skin-lesion-classification.git
```
2. **Open Google Colab**: Navigate to Google Colab and upload the cloned repository's notebook file (skin_lesion_classification.ipynb).
3. **Mount Google Drive**: Mount your Google Drive to access and store datasets and models.
```
from google.colab import drive
drive.mount('/content/drive')
```
4. **Set the Working Directory**: Change the working directory to the project folder in your Google Drive.
```
cd /content/drive/MyDrive/UCCD3074_Labs/Assignment2
```

## Contributing
We appreciate your interest in contributing to the Skin Lesion Classification project. Whether you are offering feedback, reporting issues, or proposing new features, your contributions are invaluable. Here's how you can get involved:
### How to Contribute
1. **Issue Reporting**
   * If you encounter any issues or unexpected behavior, please open an issue on the project.
   * Provide detailed information about the problem, including steps to reproduce it.
2. **Feature Requests**
   * Share your ideas for enhancements or new features by opening a feature request on GitHub.
   * Clearly articulate the rationale and potential benefits of the proposed feature.
3. **Pull Requests**
   * If you have a fix or an enhancement to contribute, submit a pull request.
   * Ensure your changes align with the project's coding standards and conventions.
   * Include a detailed description of your changes.
  
## License
The Skin Lesion Classification project is open-source and licensed under the [MIT License](LISENCE). By contributing to this project, you agree that your contributions will be licensed under this license. Thank you for considering contributing to our project. Your involvement helps make this project better for everyone. <br><br>
**Have Fun!** 🚀
