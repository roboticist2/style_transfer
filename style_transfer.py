import os
import tensorflow as tf

# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools

content_image_path = 'image/rhino.jpg'
style_image_path = 'image/mondrian.jpg'

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)

content_image = load_img(content_image_path)
style_image = load_img(style_image_path)

# plt.subplot(1, 2, 1)
# imshow(content_image, 'Content Image')
# 
# plt.subplot(1, 2, 2)
# imshow(style_image, 'Style Image')
# 
# plt.show()


#VGG19 모델 다운
#x = tf.keras.applications.vgg19.preprocess_input(style_image*255)
#x = tf.image.resize(x, (224, 224))
#vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
#prediction_probabilities = vgg(x)
#prediction_probabilities.shape

#VGG19 분류기 테스트
#predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
#print([(class_name, prob) for (number, class_name, prob) in predicted_top_5])


#content와 style 레이어 정의
content_layers = ['block5_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# 중간층들의 결과물을 배열로 출력하는 VGG19 모델 반환
def vgg_layers(layer_names):
  """ Creates a VGG model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on ImageNet data
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False

  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model

#style 레이어 정의
style_extractor = vgg_layers(style_layers)
#style 출력 정의
style_outputs = style_extractor(style_image*255)

#content는 중간층의 feature map값으로 표현
#style은 feature map의 평균과 feature map 사이의 상관관계로 표현
#gram matrix는 각 위치의 feature vector의 outer product 수행 및 평균 계산
def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

#style, content의 tensor 반환 모델
#입력 이미지에 대해 gram matrix 출력
class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg = vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name: value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name: value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content': content_dict, 'style': style_dict}

extractor = StyleContentModel(style_layers, content_layers)

#results = extractor(tf.constant(content_image))

#style, content의 타겟 정의
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

#최적화할 이미지 정의 및 content 이미지 활용하여 초기화, content 이미지와 크기 동일함
image = tf.Variable(content_image)

# 픽셀값이 실수이므로 0~1 사이로 cliping
def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

# 최적화 함수 정의 # LBFGS or Adam
opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

style_weight=1e-2
content_weight=1e4

# 오차 정의
def style_content_loss(outputs):
  style_outputs = outputs['style']
  content_outputs = outputs['content']
  style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                         for name in style_outputs.keys()])
  style_loss *= style_weight / num_style_layers

  content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                           for name in content_outputs.keys()])
  content_loss *= content_weight / num_content_layers
  loss = style_loss + content_loss
  return loss

total_variation_weight=30

# tf.GradientTape로 이미지 업데이트
@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)
    loss += total_variation_weight*tf.image.total_variation(image)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))

# #학습 테스트
# train_step(image)
# train_step(image)
# train_step(image)
# tensor_to_image(image)
#
# imshow(image, 'Style_transfered Image')
# plt.show()

import time
start = time.time()

epochs = 10
steps_per_epoch = 2

step = 0
for n in range(epochs):
  for m in range(steps_per_epoch):
    step += 1
    train_step(image)
    print(".", end='', flush=True)
  #display.clear_output(wait=True)
  #display.display(tensor_to_image(image))
  plt.imshow(tensor_to_image(image))
  plt.show()
  print("Train step: {}".format(step))

end = time.time()
print("Total time: {:.1f}".format(end-start))