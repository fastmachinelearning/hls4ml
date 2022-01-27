from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.layers import Transpose

class RemoveUselessTranspose(OptimizerPass):
    def match(self, node):
        is_match = isinstance(node, Transpose) and\
                   node.get_attr('perm') == [0] #Useless transpose
        return is_match
    
    def transform(self, model, node):
        """
        Remove a transpose layer if it doesn't do anything. i.e 1D input and perm = [0]
        """
        print("Unnessary {} in the model, optimizing ...".format(node.name))
        if not node.get_output_nodes():
            print("WARNING: {} is the output layer! No rewiring performed.".format(node.name))
            model.remove_node(node, rewire=False) #Don't rewire if there is no output layer
        else:
            model.remove_node(node, rewire=True)
       
        return True