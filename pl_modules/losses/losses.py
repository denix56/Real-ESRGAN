import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class PerceptualLoss2(nn.Module):
    """Perceptual loss with commonly used style loss.
    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1',
                 style_criterion=None,
                 norm_gram=False,
                 eig=False):
        super(PerceptualLoss2, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')
        
        self.norm_gram = norm_gram
        self.eig = eig
        
        self.style_criterion_type = style_criterion
        if style_criterion is None:
            self.style_criterion = self.criterion
            self.style_criterion_type = self.criterion_type
        else:
            if self.style_criterion_type == 'l1':
                self.style_criterion = torch.nn.L1Loss()
            elif self.style_criterion_type == 'l2':
                self.style_criterion = torch.nn.MSELoss()
            elif self.style_criterion_type == 'fro':
                self.style_criterion = None
            else:
                raise NotImplementedError(f'{style_criterion} criterion has not been supported.')
        

    def forward(self, x, gt):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.style_criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k], norm=self.norm_gram, eig=self.eig) - self._gram_mat(gt_features[k], norm=self.norm_gram, eig=self.eig), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.style_criterion(self._gram_mat(x_features[k], img=x, norm=self.norm_gram, eig=self.eig), self._gram_mat(
                        gt_features[k], img=gt, norm=self.norm_gram, eig=self.eig)) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x, img=None, norm = False, eig=False):
        """Calculate Gram matrix.
        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).
        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        if norm:
            features = F.normalize(features, dim=2)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t)
        if norm:
            #norm1 = torch.norm(features, dim=-1, keepdim=True)
            #norm2 = norm1.transpose(1, 2)
            #eps = 1e-8
            #norm12 = norm1*norm2
            #gram = gram / torch.max(norm12, eps*torch.ones_like(norm12))
            
            if eig:
                try:
                    gram = torch.linalg.eigvalsh(gram)
                except:
                    data = {}
                    if img is not None:
                        data['org'] = img
                    data['x'] = x
                    data['gram'] = gram
                    torch.save(data, 'data_err.pth')
                        
                gram = gram[..., -1]
            
        else:
            gram = gram / (c * h * w)
        return gram
