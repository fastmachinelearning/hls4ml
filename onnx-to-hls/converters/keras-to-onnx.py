import onnxmltools
from onnx import checker, shape_inference, optimizer
from onnx.utils import polish_model
from keras.models import Model
from keras.models import model_from_json
import argparse

def keras_to_onnx():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-m", action='store', dest='model',
                        help="Keras model file (.json).")
    parser.add_argument("-w", action='store', dest='weights',
                        help="Keras model weights (.h5).")
    parser.add_argument("-o", action='store', dest='output',
                        help="Output file name (.onnx).")
    args = parser.parse_args()
    
    if not args.model: parser.error('Model file needs to be specified.')
    if not args.weights: parser.error('Weights file needs to be specified.')
    if not args.weights: parser.error('Output file needs to be specified.')
    
    # Load Keras model and its weights
    with open(args.model, 'r') as json_file:
        keras_model = model_from_json(json_file.read())
    
    keras_model.load_weights(args.weights)
    #keras_model.summary()

    # Save to ONNX format
    onnx_model = onnxmltools.convert_keras(keras_model)

    # Check model
    checker.check_model(onnx_model)

    # Infer shape
    onnx_model = shape_inference.infer_shapes(onnx_model)

    passes = ['fuse_matmul_add_bias_into_gemm', 'fuse_consecutive_transposes', 'fuse_transpose_into_gemm']
    onnx_model = optimizer.optimize(onnx_model, passes)
    onnx_model = polish_model(onnx_model)
    onnxmltools.utils.save_model(onnx_model, args.output)


if __name__ == "__main__":
    keras_to_onnx()
