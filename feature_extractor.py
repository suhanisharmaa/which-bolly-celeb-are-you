# import os
# import pickle
#
# actors = os.listdir('Bollywood_celeb_face_localized')
# print(actors)
#
# filenames = []
#
# for actor in actors:
#     for file in os.listdir(os.path.join('Bollywood_celeb_face_localized',actor)):
#         filenames.append(os.path.join('Bollywood_celeb_face_localized',actor,file))
#
# # print(filenames)
# # print(len(filenames))
#
# pickle.dump(filenames,open('filenames.pkl','wb'))
# dont need the above as we already got the pkl file

from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from tqdm import tqdm

filenames = pickle.load(open('filenames.pkl','rb'))

# create model
model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

def feature_extractor(img_path,model):
    img = image.load_img(img_path,target_size=(224,224)) #load and resize img
    img_array = image.img_to_array(img) #convert img to numpy array for numerical computation
    expanded_img = np.expand_dims(img_array,axis=0) #adding extra dimension, models will accept imgs as batches even if it is a single img
    preprocessed_img = preprocess_input(expanded_img) #normalization and scaling is done to img
    result = model.predict(preprocessed_img).flatten() #run the model on img flatten from 2d to 1d
    return result

# Scaling	     Resize values to a new range (like 0–1)	        [0, 1] or [-1, 1]
# Normalization	 Often means subtracting mean or dividing by std	varies

# Stage	                     Shape	            Notes
# Image loaded	             (224, 224, 3)	    3D (RGB image)
# Expanded for batch	     (1, 224, 224, 3)	4D (model input)
# Model output (avg pooled)	 (1, 2048)	        2D (batch of feature vectors)
# After flatten()	         (2048,)	        1D (just the features for 1 image)

features = []

#tqdm is used to get a realtime progressbar in run env
for file in tqdm(filenames):
    features.append(feature_extractor(file,model)) #every img will give a vector of 2048

pickle.dump(features,open('embedding.pkl','wb'))
