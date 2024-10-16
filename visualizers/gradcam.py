import torch
from torch.autograd import Function as TorchFunc
import numpy as np
from PIL import Image


class CustomReLU(TorchFunc):
    """
    Define the custom change to the standard ReLU function necessary to perform guided backpropagation.
    We have already implemented the forward pass for you, as this is the same as a normal ReLU function.
    """

    @staticmethod
    def forward(self, x):
        output = torch.addcmul(torch.zeros(x.size()), x, (x > 0).type_as(x))
        self.save_for_backward(x, output)
        return output

    @staticmethod
    def backward(self, dout):

        x, _ = self.saved_tensors  # Retrieve tensors saved from the forward pass
        dx = dout.clone()  # Clone the upstream gradient
        dx[x <= 0] = 0  # Zero out gradients where input was <= 0
        dx[dout <= 0] = 0  # Zero out gradients where upstream gradient was <= 0
        return dx



class GradCam:
    def guided_backprop(self, X_tensor, y_tensor, gc_model):
        """
        Compute a guided backprop visualization using gc_model for images X_tensor and 
        labels y_tensor.

        Input:
        - X_tensor: Input images; Tensor of shape (N, 3, H, W)
        - y: Labels for X; LongTensor of shape (N,)
        - model: A pretrained CNN that will be used to compute the guided backprop.

        Returns:
        - guided backprop: A numpy of shape (N, H, W, 3) giving the guided backprop for 
        the input images.
        """

        for param in gc_model.parameters():
            param.requires_grad = True

        for idx, module in gc_model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                gc_model.features._modules[idx] = CustomReLU.apply
            elif module.__class__.__name__ == 'Fire':
                for idx_c, child in gc_model.features[int(idx)].named_children():
                    if child.__class__.__name__ == 'ReLU':
                        gc_model.features[int(idx)]._modules[idx_c] = CustomReLU.apply


        X_tensor.requires_grad = True
        output = gc_model(X_tensor)

        # Compute loss as the correct class score
        correct_class_scores = output.gather(1, y_tensor.view(-1, 1)).squeeze()

        # Perform backward pass (guided backpropagation)
        correct_class_scores.backward(torch.ones_like(correct_class_scores))

        # Get the gradient with respect to input image
        grad = X_tensor.grad.data

        # Return gradient as numpy array for visualization
        return grad.permute(0, 2, 3, 1).cpu().numpy()  # Return (N, H, W, 3)


    def grad_cam(self, X_tensor, y_tensor, gc_model):
        """
        Input:
        - X_tensor: Input images; Tensor of shape (N, 3, H, W)
        - y: Labels for X; LongTensor of shape (N,)
        - model: A pretrained CNN that will be used to compute the gradcam.
        """
        conv_module = gc_model.features[12]
        self.gradient_value = None  # Stores gradient of the module you chose above during a backwards pass.
        self.activation_value = None  # Stores the activation of the module you chose above during a forwards pass.

        def gradient_hook(a, b, gradient):
            self.gradient_value = gradient[0]

        def activation_hook(a, b, activation):
            self.activation_value = activation

        conv_module.register_forward_hook(activation_hook)
        conv_module.register_backward_hook(gradient_hook)

        # Perform forward pass
        output = gc_model(X_tensor)
        
        # Compute the correct class scores
        correct_class_scores = output.gather(1, y_tensor.view(-1, 1)).squeeze()

        # Perform backward pass to compute gradients
        gc_model.zero_grad()
        correct_class_scores.backward(torch.ones_like(correct_class_scores))

        # Get the captured gradients and activations
        gradients = self.gradient_value  # Gradient from backward pass
        activations = self.activation_value  # Activation from forward pass

        # Compute the weights as the average of the gradients over the spatial dimensions
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)  # Compute weights for each channel

        # Compute the weighted combination of activations and gradients (Grad-CAM)
        grad_cam = torch.sum(weights * activations, dim=1).clamp(min=0)  # ReLU on weighted sum
        
        # Convert to numpy and return (N, K, K)
        cam = grad_cam.detach().cpu().numpy()


        # Rescale GradCam output to fit image.
        cam_scaled = []
        for i in range(cam.shape[0]):
            cam_scaled.append(np.array(Image.fromarray(cam[i]).resize(X_tensor[i, 0, :, :].shape, Image.BICUBIC)))
        cam = np.array(cam_scaled)
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam
