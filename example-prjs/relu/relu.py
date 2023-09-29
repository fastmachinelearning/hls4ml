import tensorflow as tf
import numpy as np

# Create a relu 1layer that takes in a 25 element array
def create_model():
	# Create a model
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.InputLayer(input_shape=(25,)))
	model.add(tf.keras.layers.ReLU())
				
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
    save_model(model, name='relu')

    # Image Matrix
    image_mat = np.array([
	[ 1, -2, 3, -4, 5, -5, 1, -2, 3, -4, 4, -5, 1, -2, 3, -3, 4, -5, 1, -2, 2, -3, 4, -5, 1 ]
    ])

    # Get prediction
    prediction = model.predict(image_mat)
    print("Image Matrix\n")
    print(image_mat)
    print("Prediction\n")
    print(prediction)

    image_mat2 = np.array([
	[ -1, 2, -3, 4, -5, -6, 7, -8, 9, -10, -11, 12, -13, 14, -15, -16, 17, -18, 19, -20, -21, 22, -23, 24, -25 ]
    ])

    # Get prediction
    prediction = model.predict(image_mat2)
    print("Image Matrix\n")
    print(image_mat2)
    print("Prediction\n")
    print(prediction)


