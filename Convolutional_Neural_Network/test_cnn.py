import pickle
import numpy as np
from keras.preprocessing import image

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

test_image = image.load_img('car.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)

#Dictionary of actual class labels
labels = {0:'Aeroplane', 1:'Bike', 2:'Bus', 3:'Car', 4:'Train'}

pred_label = result[0].argmax()

print('Predicted class is : {}'.format(labels[pred_label]))