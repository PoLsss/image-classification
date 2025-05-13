## Simple CNN network to classify images of dogs and cats
### Dataset: Cats and Dogs Image Classification
Data Source:
The dataset used in this project is collected from Roboflow Universe: [Cats and Dogs Image Classification (Roboflow)](https://universe.roboflow.com/workspace1-aalti/cats-and-dogs-image-classification/dataset/1)


The dataset consists of images belonging to two classification classes: "Cat" and "Dog", serving the image classification problem.


Summary:
 - Total images: 2.000
 - Image format: .jpg, .png.

Data directory structure: make sure it has the following structure

```bash
    data
    ├── train/
    │   ├── cats/
    │   │   ├── 001.jpg
    │   │   └── ...
    │   └── dogs/
    │       ├── 101.jpg
    │       └── ...
    ├── valid/
        ├── cats/
        │   ├── 001.jpg
        │   └── ...
        └── dogs/
            ├── 101.jpg
            └── ...
  
```
### How to run
## Installation
1. Clone this repository:
      
       git clone https://github.com/PoLsss/image-classification.git
       cd image-classification
      
2. Install the necessary libraries using the command below:
   
       pip install -r requirements.txt

3. Trainning:
   
       python training.py --root <you data folder>

4. Tesing:

4.1 Tesing a image:
   
       python test_One.py -im <your image dir>

4.2 Test many image:


       test many: python Test_Many.py --image-dir <your folder dir>

Your folder dir should have this type:

```bash

data_test/
 ├── cats/
 │   ├── 001.jpg
 │   └── ...
 └── dogs/
     ├── 101.jpg
     └── ...
...
