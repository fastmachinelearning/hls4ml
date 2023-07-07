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
	input_layer = tf.keras.Input(shape=(5,5))
	conv1 = tf.keras.layers.Conv1D(filters=1,
               kernel_size=3,
               strides=1,
               padding='valid')(input_layer)
	model = tf.keras.models.Model(inputs=input_layer, outputs=conv1)
			
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
    save_model(model, name='conv1d_1layer')

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


