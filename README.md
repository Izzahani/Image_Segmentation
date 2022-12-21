 ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

# Semantic Segmentation for Images Containing Cell Neuclei by using TensorFlow
 
 ## Summary
<p>To speed up research on a variety of illnesses, such as cancer, heart disease, and uncommon ailments, an algorithm that can automatically recognise nuclei is needed. The development of remedies for a variety of illnesses, including chronic obstructive pulmonary disease, Alzheimer's, diabetes, and even the common cold, might be accelerated greatly with the help of such a tool.</p>

<p>Therefore, identifying cell nuclei is an important first step in many research investigations because it enables researchers to examine the DNA present in the nucleus, which contains the genetic information that governs the activity of each cell. Researchers can investigate how cells react to various treatments and learn more about the underlying biological mechanisms at work by identifying the nuclei of cells. It may be possible to speed up drug testing and shorten the period before new medications are made available to the general public with an automated AI model for detecting nuclei.</p>

<p>There several steps need to be complete to build the model.</p>
<p>1. Data Loading</p>
  <ol>- Jn this project, I used operating system (os) to load the data by joining the dataset with train and test.</ol>
  <ol>- In order to upload the images of inputs and masks in the train and test dataset, I used opencv. In the coding, I also included the image's colour. For inputs, I set it to RGB and grayscale for masks. </ol>
  <ol>- I did the same procedure for testing data.</ol>
  
<p>2. Data Pre-processing/p>
   <ol>- The images was in list. Therefore, I converted the images to numpy array.</ol>
   <ol>- Then, I checked some examples. Image belows show the images in training data. </ol>
   <p align="center"><img src="example_img.PNG" alt="examaple img">
   <ol>- I expanded the mask dimension for training and testing data and check the mask output.</ol>
   <ol>- Then, I converted the masks value for training and testing data into class labels.</ol>
   <ol>- Train-validation split is being performed.</ol>
   <ol>- The numpy array for training data, validation data and testing data was then converted into tensor.</ol>
   <ol>Then, combine all images and masks using zip.</ol>
   

<p>3. Data Cleaning</p>
   <ol>- Data cleaning is important to increase overall productivity and allow for the highest quality information in your decision-making.</ol>
   <ol>- I used Regex to remove unwanted words which then leave only the words with alphabets A-Z</ol>
   <ol>- The alphabets are then all converted in lower case.</ol>
   <ol>- All of the duplicated data has been removed in this part as well.</ol>
   <ol>- I then did data augmentation.</ol>
   <ol>- The data is converted into prefetch dataset.</ol>
   <ol>- Then, I checked some examples. Image belows show the images in training data after prefetch the dataset. </ol>
   <p align="center"><img src="visualize_some_example.png" alt="visualizeexamaple img">
 <ol>- The image was divided into two types, Inputs and Masks. Based on the image, the picture on the right, represent masks of the inputs on the left.</ol>

<p>3. Model Development</p>
   <ol>- Image segmentation model was created.</ol>
   <ol>- Then, I used pretrained model as the feature extraction layers.</ol>
   <ol>- I list down the activation layers, feature extraction model, unsampling path and the output layer.</ol>
<ol>- The model was used to construct the entire U-Net.</ol>
   
 <p>4. Model Evaluation</p>
 <ol>- In this section, the prediction is displayed.</ol>
  <p align="center"><img src="prediction_before.png" alt="prediction before_img">
 <ol>- Based on the image above, the predicted image is not clear. Thus, the model need to be trained to get a better prediction.</ol>
 <ol>- I applied early stopping and tensorboard to improve the prediction</ol>
 
 <p>5. Model Training</p>
 <ol>- After that, the model was trained. </ol>
 <p align="center"><img src="model_training.png" alt="prediction img">
 <p align="center"><img src="epoch_acc.png" alt="epoch acc">
  <p align="center"><img src="epoch_loss.png" alt="epoch loss">
 
 
 <p> Then, the project is being compiled. The result as shown in the picture below:</p>
  <p align="center"><img src="https://github.com/Izzahani/Article_Classification/blob/main/prediction.png" alt="descriptive text">
 <div align="center"><ol>The f1-score for accuracy based on the image is 81%. However, eliminating stop words can increase accuracy. Additionally, adding another dense layer might increase accuracy as well.</ol></div>
 
 <p>For Epoch Acc</p>
  <p align="center"><img src="https://github.com/Izzahani/Article_Classification/blob/main/epoch_acc.png" alt="epoch acc">
 <div align="center"><ol>According to the graph, the train model was coloured green. Between 1.5 and 2, the model started to get overfitted. Perhaps the Bidrectional LSTM layer is at cause. But when it reached to 2, it started to get close to optimum. So maybe the dropout layer is the reason. But at 3, it became overfitted again. I might need to add another dropout layer to reached the good git graph.</ol></div>

 <p>For Epoch Loss</p>
  <p align="center"><img src="https://github.com/Izzahani/Article_Classification/blob/main/epoch_loss.png" alt="epoch loss">
 
## Acknowledgement
Special thanks to [(https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv)](https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv) :smile:

