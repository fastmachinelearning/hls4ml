import tensorflow as tf
import numpy as np

# Sources:
# https://www.geeksforgeeks.org/python-tensorflow-tf-keras-layers-conv2d-function/
# https://jiafulow.github.io/blog/2021/02/17/simple-fully-connected-nn-firmware-using-hls4ml/
# https://stackoverflow.com/questions/51930312/how-to-include-a-custom-filter-in-a-keras-based-cnn

# Create custom kernel
# NOTE: This kernel is random and purely for testing small examples
def random_kernel(shape=(3,3,1), dtype=None):

    f = np.array([
            [[[1]], [[-1]], [[1]]],
            [[[-1]], [[1]], [[-1]]],
            [[[1]], [[-1]], [[1]]]
        ])
    assert f.shape == shape
    return f

# Create model with one conv2d layer for small example
def create_model():
	# Create a model
	model = tf.keras.Sequential()
				
	# First layer args.:
	# filters: Number of output filters.
	# kernel_size: Convolution window size width and height.
	# strides: Stride of the convolution.
	# padding: "same" adds padding if needed to ensure output dimensions are equal to input dimensions. "valid" means no padding.
	# activation: Non-linear functions (i.e. relu).
	# use_bias: Boolean or bias vectors.
	# dilation_rate: Dilation rate for dilated convolutions.
	# kernel_initializer: Default is glorot_uniform, meaning it initializes acrossed an uniform distribution.
	# bias_initializer: Initializer for bias vectors.
	# kernel_constraint: Constraint function for the kernel.
	# bias_constraint: Constraint function for the bias vectors. 

	# NOTE: Input size indicates a 5x5 pixel image (matrix) with one channel (i.e. just the red channel from RGB).
	# Image (matrix) size is equal to kernel size since this is a very small example.
	model.add(tf.keras.layers.Conv2D(1, 3, 1, padding="valid", kernel_initializer=random_kernel, input_shape=(5, 5, 1)))
				
	return model

# Save model to forms for hls4ml
def save_model(model, name=None):
	# Save as model.h5, model_weights.h5, and model.json
	if name is None:
		name = model.name
	model.save(name + '.h5')
	model.save_weights(name + '_weights.h5')
	with open(name + '.json', 'w') as outfile:
		outfile.write(model.to_json())
	return

if __name__ == '__main__':
    model = create_model()
    save_model(model, name='conv2d_1layer')

    # Image Matrix
    image_mat = np.array([
	[ [1], [2], [3], [4], [5] ],
	[ [6], [7], [8], [9], [10] ],
	[ [11], [12], [13], [14], [15] ],
        [ [16], [17], [18], [19], [20] ],
        [ [21], [22], [23], [24], [25] ]
    ])

    image_mat = image_mat.reshape((1, 5, 5, 1))

    # Get prediction
    prediction = model.predict(image_mat)
    print("Image Matrix\n")
    print(image_mat)
    print("Prediction\n")
    print(prediction)

    image_mat2 = np.array([
	[ [1], [2], [3], [4], [5] ],
	[ [5], [1], [2], [3], [4] ],
	[ [4], [5], [1], [2], [3] ],
        [ [3], [4], [5], [1], [2] ],
        [ [2], [3], [4], [5], [1] ]
    ])

    image_mat2 = image_mat2.reshape((1, 5, 5, 1))

    # Get prediction
    prediction = model.predict(image_mat2)
    print("Image Matrix\n")
    print(image_mat2)
    print("Prediction\n")
    print(prediction)


