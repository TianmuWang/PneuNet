# PneuNet

## The model along with Dataset cannot be used for clinical dignosis on COVID-19 without the agreement from clinician.

## Dataset access
The open accessed Dataset can be obtained from 
  1.Kaggle https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia 
and from 
  2.https://github.com/ieee8023/COVID-chestxray-dataset
  
## Multi-category classification
In this github, the PneuNet is build for 3-category classification, from CXR images of non pneumonia, normal pneumonia and COVID-19. The dataset used in this project can be fine divided into 4 categories which are non pneumonia, bacterial pneumonia, viral pneumonia and COVID-19 but not givven in this Github repo.
To change the model to realzie multi-category classification, you need to both edit the shape of layers in PneuNet.ipynb but also make changes on file under utils/Image_reader.py that matches the shape.
