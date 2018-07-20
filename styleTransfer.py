import numpy as np
import scipy as sp
import keras
import keras.backend as K
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf


def show_image(image, figsize=None, show_shape=False):
    if figsize is not None:
        plt.figure(figsize=figsize)
    if show_shape:
        plt.title(image.shape)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def preprocess(img):    
	img = img.copy()              
	img = img.astype('float64')
	img = img[:,:,:3]         
	img = np.expand_dims(img, axis=0) 
	return keras.applications.vgg16.preprocess_input(img)

def initialize_variables(content, style):
  content_img   = K.variable(preprocess(content))
  style_img     = K.variable(preprocess(style))
  generated_img = K.placeholder(content_img.shape)
  

  return content_img, style_img, generated_img


def calc_content_loss(layer_dict, content_layer_names):
    loss = 0
    for name in content_layer_names:
        layer = layer_dict[name]
        content_features   = layer.output[0, :, :, :] 
        generated_features = layer.output[2, :, :, :]  
        loss += K.sum(K.square(generated_features - content_features)) 
    return loss / len(content_layer_names)


def gram_matrix(x):    
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features)) 
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
        style_features     = layer.output[1, :, :, :] 
        generated_features = layer.output[2, :, :, :] 
        loss += style_loss(style_features, generated_features) 
    return loss / len(style_layer_names)
def calc_variation_loss(x):
    row_diff = K.square(x[:, :-1, :-1, :] - x[:, 1:,    :-1, :])
    col_diff = K.square(x[:, :-1, :-1, :] - x[:,  :-1, 1:,   :])
    return K.sum(K.pow(row_diff + col_diff, 1.25))


def transfer_style(content_img, 
                   style_img,
                   content_layer_names, 
                   style_layer_names,
                   content_loss_ratio, 
                   style_loss_ratio, 
                   variation_loss_ratio,
                   start_img=None, 
                   steps=10,
                   learning_rate=0.001,
                   show_generated_image=True,
                   figsize=(10,20)):
    # clear the previous session if any
    K.clear_session()
    
    # by default start with the content image
    if start_img is None:
        start_img = content_img

    # prepare inputs and the model
    content_input, style_input, generated_input = initialize_variables(content_img, style_img)
    input_tensor = K.concatenate([content_input, style_input, generated_input], axis=0)
    model = keras.applications.vgg16.VGG16(input_tensor=input_tensor, include_top=False)
    
    # calculate various loss
    layer_dict = {layer.name:layer for layer in model.layers}
    content_loss = calc_content_loss(layer_dict, content_layer_names)
    style_loss = calc_style_loss(layer_dict, style_layer_names)
    variation_loss = calc_variation_loss(generated_input)
    
    # calculate the gradients
    loss = content_loss_ratio   * content_loss   + \
           style_loss_ratio     * style_loss     + \
           variation_loss_ratio * variation_loss

    grads = K.gradients(loss, generated_input)[0]
    calculate = K.function([generated_input], [loss, grads])

    # nudge the generated image to apply the style while keeping the content
    generated_data = preprocess(start_img)
    for i in tqdm(range(steps)):
        _, grads_value = calculate([generated_data])
        generated_data -= grads_value * learning_rate
        
    # reverse the preprocessing
    generated_img = deprocess(generated_data)
    
    if show_generated_image:
        show_image(generated_img, figsize=(10,20))
        
    return generated_img




def deprocess(img):
    img = img.copy()                   
    img = img[0]                       
    img[:, :, 0] += 103.939            # these are average color intensities used 
    img[:, :, 1] += 116.779            # by VGG16 which are subtracted from 
    img[:, :, 2] += 123.68             # the content image in the preprocessing
    img = img[:, :, ::-1]              
    img = np.clip(img, 0, 255)         
    return img.astype('uint8')         
  
  
content = plt.imread('./content.jpg') 
show_image(content, show_shape=True)

generated = plt.imread('./style.jpg')      
show_image(generated, show_shape=True)

TARGET_SIZE = content.shape[:2]
generated = sp.misc.imresize(generated, TARGET_SIZE) 
show_image(generated, show_shape=True)


generated = transfer_style(
    content, 
    generated,
    ['block5_conv2'], 
    ['block1_conv1',
     'block2_conv1',
     'block3_conv1',
     'block4_conv1', 
     'block5_conv1'],
    content_loss_ratio=0.05, 
    style_loss_ratio=1.0, 
    variation_loss_ratio=0.3,               
    steps=1,
    learning_rate=0.01)
plt.imsave(fname="output.jpg", arr = generated)