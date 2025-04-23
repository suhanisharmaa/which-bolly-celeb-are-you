# steps:
# load img detect face and extract features
# find cosine distance of curr img with all other 8655 features
# recommend

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from mtcnn import MTCNN
from PIL import Image

feature_list = pickle.load(open('embedding.pkl','rb'))
feature_list = np.array(feature_list)
filenames = pickle.load(open('filenames.pkl','rb'))

model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

#LOAD IMG AND FACE DET
detector = MTCNN() #mtcnn object Multi-task Cascaded Convolutional Networks used for face detection

sample_img = cv2.imread('sampleimgs/ranbir_si.jpg')
results = detector.detect_faces(sample_img)
# print("Image shape:", sample_img.shape)

x,y,width,height = results[0]['box']

face = sample_img[y:y+height,x:x+width]

# cv2.imshow('output',face)
# cv2.waitKey(0)

#EXTRACT FEATURES
image = Image.fromarray(face) #PIL image obj
image = image.resize((224,224)) # ✅ One tuple argument with width and height
face_array = np.asarray(image)
face_array = face_array.astype('float32')
expanded_img = np.expand_dims(face_array,axis=0)
preprocessed_img = preprocess_input(expanded_img)
result = model.predict(preprocessed_img).flatten()
# print(result)
# print(result.shape)

# compare [vector result with 0,2048] with [vector feature_list which has 8655,2048]

# print(type(feature_list[0]), feature_list[0].shape)

#comparing just 1st vector
# print(cosine_similarity(result.reshape(1,-1),feature_list[0].reshape(1,-1))[0][0]) #reshaping as function needs 2d vectors

similarity = []
for i in range(len(feature_list)):
    similarity.append(cosine_similarity(result.reshape(1,-1),feature_list[i].reshape(1,-1))[0][0])

# print(len(similarity))
# print(list(enumerate(similarity))) #to preserve order
# print(sorted(list(enumerate(similarity)),reverse=True,key=lambda x:x[1]))
index_pos = sorted(list(enumerate(similarity)),reverse=True,key=lambda x:x[1])[0][0]
temp_img = cv2.imread(filenames[index_pos])
cv2.imshow('output',temp_img)
cv2.waitKey(0)