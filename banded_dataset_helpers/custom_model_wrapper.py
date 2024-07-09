import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import backend as K
from tcav.model import ModelWrapper

class Shapes_Classifier_Wrapper(ModelWrapper):
    """ModelWrapper for a custom Keras model."""
    
    def __init__(self, model_path, labels):
        """Initialize the wrapper with the model path and labels.
        
        Args:
          model_path: Path to the saved Keras model.
          labels: List of class labels.
        """
        super(Shapes_Classifier_Wrapper, self).__init__()
        print(f"Loading model from {model_path}")
        self.model = load_model(model_path)
        self.labels = labels
        self.model_name = 'Shapes_Classifier'
        
        print("Constructing gradient ops...")
        # Construct gradient ops. Defaults to using the model's output layer
        self.y_input = tf.compat.v1.placeholder(tf.int64, shape=[None])
        self.loss = tf.reduce_mean(
            input_tensor=tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.stop_gradient(tf.one_hot(self.y_input, self.model.output.shape[-1])),
                logits=self.model.output),
            axis=0)
        self._make_gradient_tensors()
    
    def get_image_shape(self):
        """Returns the shape of an input image."""
        print(f"Model input shape: {self.model.input.shape[1:]}")
        return self.model.input.shape[1:]
    
    def get_predictions(self, examples):
        """Get prediction of the examples.
        
        Args:
          examples: array of examples to get predictions
        
        Returns:
          array of predictions
        """
        print(f"Predicting on examples with shape: {examples.shape}")
        predictions = self.model.predict(examples)
        print(f"Predictions shape: {predictions.shape}")
        return predictions
    
    def id_to_label(self, idx):
        """Convert index in the logit layer (id) to label (string)."""
        return self.labels[idx]
    
    def label_to_id(self, label):
        """Convert label (string) to index in the logit layer (id)."""
        return self.labels.index(label)
    
    def reshape_activations(self, layer_acts):
        """Reshapes layer activations as needed to feed through the model network.
        
        Args:
          layer_acts: Activations as returned by run_examples.
        
        Returns:
          Activations in model-dependent form; the default is a squeezed array (i.e.
          at most one dimension of size 1).
        """
        reshaped_acts = np.asarray(layer_acts).squeeze()
        print(f"Reshaped activations to: {reshaped_acts.shape}")
        return reshaped_acts
    
    def run_examples(self, examples, bottleneck_name):
        """Get activations at a bottleneck for provided examples.
        
        Args:
          examples: example data to feed into network.
          bottleneck_name: string, should be key of self.bottlenecks_tensors
        
        Returns:
          Activations in the given layer.
        """
        print(f"Running examples through bottleneck: {bottleneck_name}")
        print(f"Original examples shape: {examples.shape}")

        # Ensure the examples are grayscale and normalized
        if examples.shape[-1] == 3:
            examples = np.mean(examples, axis=-1, keepdims=True)
        examples = examples / 255.0

        print(f"Processed examples shape: {examples.shape}")
        
        bottleneck_tensor = self.model.get_layer(bottleneck_name).output
        intermediate_model = tf.keras.models.Model(inputs=self.model.input, outputs=bottleneck_tensor)
        activations = intermediate_model.predict(examples)
        print(f"Activations shape: {activations.shape}")
        return activations
    
    def _make_gradient_tensors(self):
        """Makes gradient tensors for all bottleneck tensors."""
        self.bottlenecks_tensors = {layer.name: layer.output for layer in self.model.layers if 'input' not in layer.name}
        self.bottlenecks_gradients = {}
        for bn in self.bottlenecks_tensors:
            grads = K.gradients(self.loss, self.bottlenecks_tensors[bn])[0]
            self.bottlenecks_gradients[bn] = grads
        print(f"Bottleneck tensors: {list(self.bottlenecks_tensors.keys())}")
    
    def get_gradient(self, acts, y, bottleneck_name, example):
        """Return the gradient of the loss with respect to the bottleneck_name.
        
        Args:
          acts: activation of the bottleneck
          y: index of the logit layer
          bottleneck_name: name of the bottleneck to get gradient wrt.
          example: input example. Unused by default. Necessary for getting gradients
            from certain models, such as BERT.
        
        Returns:
          the gradient array.
        """
        
        # Ensure the example is shaped as a batch
        if example.ndim == 3:
            example = np.expand_dims(example, axis=0)
        

        # Ensure the example is grayscale
        if example.shape[-1] == 3:
            example = np.mean(example, axis=-1, keepdims=True)
        example = example / 255.0

        print(f"Processed example shape for gradient: {example.shape}")
        
        # Ensure y is shaped as a batch
        if isinstance(y, int):
            y = [y]
        
        y = np.array(y)
        
        input_tensors = [self.model.input, self.y_input]
        grads = self.bottlenecks_gradients[bottleneck_name]
        get_grads = K.function(inputs=input_tensors, outputs=[grads])
        gradient_values = get_grads([example, y])[0]
        return gradient_values


