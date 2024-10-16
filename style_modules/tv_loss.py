import torch
import torch.nn as nn

class TotalVariationLoss(nn.Module):
    def forward(self, img, tv_weight):
        """
            Compute total variation loss.

            Inputs:
            - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
            - tv_weight: Scalar giving the weight w_t to use for the TV loss.

            Returns:
            - loss: PyTorch Variable holding a scalar giving the total variation loss
              for img weighted by tv_weight.
            """

        ##############################################################################
        # TODO: Implement total variation loss function                              #
        # Use torch tensor math function or else you will run into issues later      #
        # where the computational graph is broken and appropriate gradients cannot   #
        # be computed.                                                               #
        ##############################################################################

        diff_x = torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2)
        
        # Calculate the difference between adjacent pixel values in the y-direction (vertical)
        diff_y = torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2)
        
        # Sum the squared differences and apply the total variation weight
        loss = tv_weight * (diff_x.sum() + diff_y.sum())
        
        return loss

        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################