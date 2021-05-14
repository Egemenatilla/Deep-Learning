# Libraries
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,
                             horizontal_flip=True,vertical_flip=True,fill_mode='nearest')
# Upload pictures

img = load_img('araguler.jpg')
x = img_to_array(img)
x = x.reshape((1,)+x.shape)

# Get 50 images from a single image

i = 0
for batch in datagen.flow(x,batch_size = 1,save_to_dir=('Done'),save_format = 'jpeg'):
    i += 1
    if i>50:
        break
