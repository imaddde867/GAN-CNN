# GAN-CNN-gangang

plz download the dataset from : 

https://storage.googleapis.com/kaggle-data-sets/2701470/4649306/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20251009%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20251009T113525Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=3bacb4f03d78b3fb7fe986b16d7e558a784390d3c13d484fe6a66594b2a5cb79b9ebd570278f6ddf722d45519d4a98506e8eb104b59e290fc84a2d3a7e7ef432f24061e9543843df739677b7e46d16248ca04f8d4dbc8a484c04b7f19563282013bbac4f0d0846d021d8f04ace0166e0fd368915236ce5ef3f37562cbb0dc97b356caf6162f93e82f7155ebf2fb093b5ce8a7ac8b31e7e9aa2c28aa22fb3519716c1a05e918de8a3ee0839cc28af494b94cc1fcdc1c4d925e3f70a18cc804f2f217c3408b7185fff4be23a6a25b924a4abaa8bc2c736d093797f734b6e7136dacbdce66f180959efc113bc5ab56ee51bc1c98c4305891db45570c5cee1adcf02

rename the folder into "data" and move it to the repo : it should be like this : 

data / 
train/images/  <- actual .jpg/.png files
train/labels/  <- .txt files with bounding box coordinates
valid/images/  <- validation images
valid/labels/  <- validation annotations

ALSO : the class number for cars is 5
also : the format of the labels is class + x_center + y_center + width + height