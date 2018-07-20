import numpy as np
import scipy as sp
import keras
import keras.backend as K
import matplotlib.pyplot as plt
from tqdm import tqdm

def show_image(image, show_shape=False):
	if show_shape:
		plt.title(image.shape)
	plt.imshow(image)
	plt.xticks([])
	plt.yticks([])
	plt.show()

def preprocess(img):
    img = img.copy()                   # copy so that we don't mess the original data
    img = img.astype('float64')        # make sure it's float type
    img = np.expand_dims(img, axis=0)  # change 3-D to 4-D.  the first dimension is the record index
    return keras.applications.vgg16.preprocess_input(img)

def initialize_variables(content, style, h, w):
		content_img   = K.variable(content)
		style_img     = K.variable(style)
		generated_img = K.placeholder(content.shape)
		input_tensor = K.concatenate([content_img, style_img, generated_img], axis=0)

		return content_img, style_img, generated_img, input_tensor


def content_loss(layer_dict, content_layer_names):
    loss = 0
    for name in content_layer_names:
        layer = layer_dict[name]
        content_features   = layer.output[0, :, :, :]  # content features
        generated_features = layer.output[2, :, :, :]  # generated features
        loss += K.sum(K.square(generated_features - content_features)) # keep the similarity between them
    return loss / len(content_layer_names)


def gram_matrix(x):    
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1))) # flatten per filter
    gram = K.dot(features, K.transpose(features)) # calculate the correlation between filters
    return gram

def style_loss(style_features, generated_features):
    S = gram_matrix(style_features)
    G = gram_matrix(generated_features)
    channels = 3
    size = TARGET_SIZE[0]*TARGET_SIZE[1]
    return K.sum(K.square(S - G)) / (4. * (channels**2) * (size**2))

def calc_style_loss(layer_dict, style_layer_names):
    loss = 0
    for name in style_layer_names:
        layer = layer_dict[name]
        style_features     = layer.output[1, :, :, :] # style features
        generated_features = layer.output[2, :, :, :] # generated features
        loss += get_style_loss(style_features, generated_features) 
    return loss / len(style_layer_names)
def calc_variation_loss(x):
    row_diff = K.square(x[:, :-1, :-1, :] - x[:, 1:,    :-1, :])
    col_diff = K.square(x[:, :-1, :-1, :] - x[:,  :-1, 1:,   :])
    return K.sum(K.pow(row_diff + col_diff, 1.25))


content = plt.imread('./original.jpeg') # the image source is the reference [4]
show_image(content, show_shape=True)

style = plt.imread('./datasets.jpg')      
show_image(style, show_shape=True)

TARGET_SIZE = content.shape[:2]

style = sp.misc.imresize(style, TARGET_SIZE) # resize the style image to the content image size


content_input, style_input, generated_input, input_tensor = make_inputs(cat_img, hokusai_img)
model = keras.applications.vgg16.VGG16(input_tensor=input_tensor, include_top=False)

style_loss = calc_style_loss(
    layer_dict,
    ['block1_conv1',
     'block2_conv1',
     'block3_conv1',
     'block4_conv1', 
     'block5_conv1'])
variation_loss = calc_variation_loss(generated_input)

loss = 0.8 * content_loss + \
       1.0 * style_loss   + \
       0.1 * variation_loss
        
grads = K.gradients(loss, generated_input)[0]

calculate = K.function([generated_input], [loss, grads])

generated_data = preprocess(cat_img)

for i in tqdm(range(10)):
    _, grads_value = calculate([generated_data])
    generated_data -= grads_value * 0.001





def deprocess(img):
    img = img.copy()                   # copy so that we don't mess the original data
    img = img[0]                       # take the 3-D image from the 4-D record    
    img[:, :, 0] += 103.939            # these are average color intensities used 
    img[:, :, 1] += 116.779            # by VGG16 which are subtracted from 
    img[:, :, 2] += 123.68             # the content image in the preprocessing
    img = img[:, :, ::-1]              # BGR -> RGB
    img = np.clip(img, 0, 255)         # clip the value within the image intensity range
    return img.astype('uint8')         # convert it to uin8 just like a normal image data
generated_image = deprocess(generated_data)

show_image(generated_image, figsize=(10,20))
