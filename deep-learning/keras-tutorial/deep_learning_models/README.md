# Trained image classification models for Keras

This repository contains code for the following Keras models:

- VGG16
- VGG19
- ResNet50

We plan on adding Inception v3 soon.

All architectures are compatible with both TensorFlow and Theano, and upon instantiation the models will be built according to the image dimension ordering set in your Keras configuration file at ~/.keras/keras.json. For instance, if you have set image_dim_ordering=tf, then any model loaded from this repository will get built according to the TensorFlow dimension ordering convention, "Width-Height-Depth".

Weights can be automatically loaded upon instantiation (weights='imagenet' argument in model constructor). Weights are automatically downloaded if necessary, and cached locally in ~/.keras/models/.

Note that using these models requires the latest version of Keras (from the Github repo, not PyPI).

## Examples

### Classify images

```python
from resnet50 import ResNet50
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions

model = ResNet50(weights='imagenet')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))
# print: [[u'n02504458', u'African_elephant']]
```

### Extract features from images

```python
from vgg16 import VGG16
from keras.preprocessing import image
from imagenet_utils import preprocess_input

model = VGG16(weights='imagenet', include_top=False)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
```

### Extract features from an arbitrary intermediate layer

```python
from vgg19 import VGG19
from keras.preprocessing import image
from imagenet_utils import preprocess_input
from keras.models import Model

base_model = VGG19(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('block4_pool').output)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

block4_pool_features = model.predict(x)
```

## References

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) - please cite this paper if you use the VGG models in your work.
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) - please cite this paper if you use the ResNet model in your work.

Additionally, don't forget to [cite Keras](https://keras.io/getting-started/faq/#how-should-i-cite-keras) if you use these models.

## License

- All code in this repository is under the MIT license as specified by the LICENSE file.
- The ResNet50 weights are ported from the ones released by Kaiming He under the MIT license.
- The VGG16 and VGG19 weights are ported from the ones released by VGG at Oxford under the Creative Commons Attribution License.

## Maintainer

This repository is maintained by Shobhit Dixit, an AI/ML Engineer specializing in building scalable machine learning and generative AI solutions. With extensive experience in Python, PyTorch, and TensorFlow, Shobhit focuses on developing production-grade ML pipelines and real-time inference systems.

For inquiries or contributions, please contact:
- Email: iamshobhit98@gmail.com
- Role: AI/ML Engineer