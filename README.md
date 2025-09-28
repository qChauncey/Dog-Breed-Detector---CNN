# Definition Project Overview

> This project is to solve the dog breed classification problem by
> creating CNN (Convolutional Neural Network) models as the detectors.
> 
><img width="710" height="67" alt="image" src="https://github.com/user-attachments/assets/e034d0bf-4bc3-4cb1-9c09-fab92db5433e" />

> (Graph 1: Basic Flowchart)
>
> Above is the basic flowchart of the solution that will be provided.
> Each step requires applying different knowledge together to achieve
> one goal.
>
> The project is conducted based on the dataset provided by Udacity -
> 13,233 human images and 8,351 dog images.

# Problem Statement

> The project is related to the dog breed classification -- it is hard
> for the majority to identify dog breeds as this requires sufficient
> knowledge in this area.
>
> Computer Vision (CV) technique is one of the best solutions to solve
> this kind of issue. And Convolutional Neural Network(CNN) is a
> powerful tool that already solves various image classification
> problems.

# Metrics

> In the project, we will determine a model's performance by its
> prediction accuracy -- the percentage of correct images identified
> among the whole dataset.

# Analysis Data Exploration

> The project is based on the Human and Dog picture dataset provided by
> Udacity, which includes 13,233 human images and 8,351 dog images. The
> images will be transformed and resized as 224\*224 for the program to
> read and proceed. For pre-trained models (VGG16, ResNet18) are trained
> based on ImageNet, which is a worldwide famous visual database with
> more than 14 million images that have been labeled (ImageNet, 2022).

# Exploratory Visualization

> As the dataset is mainly images, we don't plan for any data
> exploratory visualization, but one of the images as below for
> reference.

<img width="342" height="357" alt="image" src="https://github.com/user-attachments/assets/02d58696-32f0-4645-980e-a318681cd898" />


> （Image 1: Example of 1 human image from the dataset）

# Algorithm and Techniques

> 3 CNN models are provided as the solutions to the problem. 1^st^ CNN
> model, which is based on VGG16, is a pre-trained VGG16 model for
> determining whether the picture provided is a dog; 2^nd^ CNN model,
> which is a simple self-designed model with 2 convolutional layers;
> 3^rd^ CNN model is based on pre-trained ResNet18 model, then applied
> transfer learning for the dog breeds classification.

# Benchmark

> For model scratch, we expect to create a simple model with 10% or
> higher accuracy to identify dog images from humans.
>
> For the transfer learning model, we expect to create a model with 60%
> or higher accuracy to identify dog images from humans.
>
> At last, we expect we can identify as many dogs' breeds as we can.

# Methodology Data Preprocessing

> To allow the algorithm to read the information of any image, necessary
> procedures and related functions are created: 1) Images decoding into
> data flow; 2) Images data resize to unify and normalize.
>
> OpenCV and Torchvision are two popular libraries available, with
> several functions aiming to decode the image into data for the
> algorithm to read.

>
> For dog identification and breeds classification, the "*Transform*"
> function provided in the *Torchvision* project is applied for image
> pre-process and normalization.

| Function                           | Description                                         |
|------------------------------------|-----------------------------------------------------|
| Resize()/CenterCrop()              | To resize and unify image size to avoid any error   |
| To.Tensor()                        | Change to Tensor format                             |
| Normalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) | Normalization, the parameter is recommended by Pytorch |


> (Table 1: Torchvision method for image transform and pre-process)

# Implementation

## Detector Function

> The detector functions are separated to turn the data prediction into
> verbal and meaningful outcomes.
>
> In this project, two detectors are created -- one human detector and
> one dog detector.

## Pre-trained VGG16 model

> The project imports the VGG16 model, which is well-trained with the
> ImageNet dataset.
>
> In this part, a pre-trained model is selected so that I can have a
> basic understanding on what's the process and architecture of a
> Computer Vision
>
> detector like A data processer function, a model as the core, and a
> "result decoder" function.

## Scratch Model

> Next will go into the details of the model's architecture. There shall
> be at least the Convolutional layers, Pooling layers, and fully
> connected for any CNN model.
>
> Two convolutional layers and one max-pooling layer are created for
> forecasting in the self-designed model. As the model's final aim is to
> classify different dog breeds, the number of classes should be 133.

## Pre-trained ResNet18 model and Transfer Learning

> Transfer learning is also a powerful tool to leverage the experience
> of the well-trained model to solve different problems. In this part, I
> mainly follow the example on Pytorch to apply the ResNet18 model and
> then makes some adjustment.
>
> Transfer learning requires the last layer's parameters to change for
> the task.

# Refinement

## Scratch Model

> CrossEntropyLoss() is selected as the loss function and Adam as the
> optimizer. These are two well-recognized choices for the task.

## Pre-trained ResNet18 model and Transfer Learning

> In this part, the CrossEntropyLoss function is the loss function, and
> SGD is the optimizer. After 25 epochs of training, the final model is
> identified.

# Result

> **Model Evaluation and Validation**

## Pre-trained VGG16 model

> As a result, the model identified 100% dog images as required.
>
><img width="146" height="61" alt="image" src="https://github.com/user-attachments/assets/c2a1633b-7899-4d57-9eee-9e7609096e76" />

> (Image2: Result of VGG16 model)

## Scratch Model

> With 100 epochs' of training, the final model's validation loss is
> 4.09 and finally identified 86 out of 825 pictures, with more than 10%
> accuracy.

<img width="292" height="97" alt="image" src="https://github.com/user-attachments/assets/34e75d03-5cd0-401b-a7c2-111bd23ed1d0" />

> (Image 3: Result of Scratch model)

## Pre-trained ResNet18 model and Transfer Learning

> As a result, the transfer learning model identified 512 out of 836
> pictures, which is 61% accuracy that meets the expectation.

<img width="303" height="97" alt="image" src="https://github.com/user-attachments/assets/547bbd7c-82b5-45f1-beaa-775c988c17a7" />

> （Image 4: Result of Transfer Learning model）

## Final Dog Breeds Classification

> Finally, we test the final classification tool with 3 dogs images and
> 3 human images. The tools successfully identify all human and dogs
> images, and classify the dog breeds correctly.

<img width="241" height="151" alt="image" src="https://github.com/user-attachments/assets/09815de5-4282-4987-9ed9-6bcd89d30bb9" />


> （Image 5: Result of Dog Breeds classification）
>
> **Justification**

# Reflection

> The project is entirely meaningful as it provides such an opportunity
> for me to try and go through creating the CNN model and turn it into a
> workable solution to a real problem. It gives me a sense of the
> difficulty of the CNN model and what level of accuracy we expect on
> different types of models.
>
> This project is valuable as the solution created can be applied to any
> other kind of image classification problem. The potential benefits of
> leveraging the knowledge might be even more precious.

# Improvement

> With the successful experience applying the CNN model to identify dogs
> images, it is possible to expand the outcome and problem range -- to
> identify other animals' classes or names.
>
> Also, in the future study, will consider creating a more complex CNN
> model for higher accuracy. And it is recommended to try different
> pre-trained models for different problems set for better prediction
> outcome.
