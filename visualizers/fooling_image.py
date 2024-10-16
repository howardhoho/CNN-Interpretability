import torch
from torch.autograd import Variable

class FoolingImage:
    def make_fooling_image(self, X, target_y, model):
        """
        Generate a fooling image that is close to X, but that the model classifies
        as target_y.

        Inputs:
        - X: Input image; Tensor of shape (1, 3, 224, 224)
        - target_y: An integer in the range [0, 1000)
        - model: A pretrained CNN

        Returns:
        - X_fooling: An image that is close to X, but that is classifed as target_y
        by the model.
        """

        model.eval()

        # Initialize our fooling image to the input image, and wrap it in a Variable.
        X_fooling = X.clone()
        X_fooling_var = Variable(X_fooling, requires_grad=True)

        # We will fix these parameters for everyone so that there will be
        # comparable outputs

        learning_rate = 10
        max_iter = 100  # maximum number of iterations

        for it in range(max_iter):
            
            # Forward pass: Compute the scores
            scores = model(X_fooling_var)  # Get the logits from the model
            

            # Compute the score for the target class
            target_score = scores[0, target_y]
            
            # Backward pass: Compute gradients with respect to the input image
            model.zero_grad()
            target_score.backward()
            
            # Get the gradient of the input image
            grad = X_fooling_var.grad.data
            
            # Normalize the gradient
            grad_norm = grad.norm()
            dX = learning_rate * grad / grad_norm
            
            # Update the image to maximize the target class score
            X_fooling_var.data += dX
            
            # Clear the gradients for the next iteration
            X_fooling_var.grad.zero_()
            

        X_fooling = X_fooling_var.data

        return X_fooling
