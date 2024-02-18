# petnet

A deep neural network for the semantic segmentation of cats and dogs images and classifying their breed. We will refer to this network as `petnet`.

This network has the following objectives:

1. Identify the presence of a cat or a dog or both in the image.
2. Identify the breed of the animal present in the image.
3. Provide a binary mask presenting which pixel features a pet (cat or dog) and which does not.

This network is trained on 7270 images of cats and dogs, and it is tested on 1818 images. Each images contain maxium one cat and one dog. 

### Network Architecture
The network is based on the U-Net architecture. It uses a pretrained RESNET-18 network as the backbone. It has two output heads. One head is for multi-label classification (pet type and breed type, 39 labels) and the other head is for semantic segmentation. The network is trained to detect maxium one cat and one dog in the image.

### Model Training
To train the model first download the dataset from [here](https://github.com/harrison-ai/hai-tech-tasks/releases/download/v0.1/cats_and_dogs.zip). Keep the `data` folder and the `pets_dataset_info.csv` file within the `dataset` folder of the project.

Build the docker image using the following command:
```bash
docker build --platform linux/amd64 -t petnet:latest .
```

After building the docker image, run the following command to train the model:
```bash
python3 src/train.py
```

You can tune the hyperparameters in the `src/config.py` file. The best model will be saved as `weights/model.pth`.





### Model Evaluation
Train and testing loss and accuracy are logged in Weights and Biases. We log the mean classification precision, recall and segmentation iou for the validation set. We also log the training and validation classification and segmentation loss. The report can be found [here](https://api.wandb.ai/links/qmaruf/48qzjuz9). 



### Inference
We have created a REST api to serve the model using FastAPI. To access the api, there is an UI based on Streamlit. To run the api, first build the docker image and then run the `docker-compose.yml` file.

```bash
docker build --platform linux/amd64 -t petnet:latest .
docker compose up
```

To access the UI go to `http://0.0.0.0:9002` in your browser. You can upload an image and get the prediction. The prediction will include the pet type (cat, dog or both), the breed of the pet and the segmentation mask.

The UI will look like this:

<img src="imgs/inf.png" width="300">

### Test
Use `pytest` from the root directory to run the tests.