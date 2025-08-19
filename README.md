# YOLO-RD



CSAF module
```python
class ESELayer(nn.Module):
    """ Effective Squeeze-Excitation
    """
    def __init__(self, channels, act='hardsigmoid'):
        super(ESELayer, self).__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)


# Conv-SPD Attention Fusion
class CSAF(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  
        compress_c = 16 
        # Conv2
        self.cv1 = Conv(c1, c_, 3, 2)
        # SPD
        self.cv2 = Conv(4*c1, c_, 1, 1)
        self.conv_level = Conv(c_, compress_c, 1, 1)
        self.spd_level = Conv(c_, compress_c, 1, 1)
    
        self.eca = ESELayer(2*compress_c)
        # Attention
        self.weight = nn.Conv2d(2*compress_c, 2, kernel_size=1, stride=1, padding=0)
        # Expand
        self.cv3 = nn.Conv2d(c_, c2, 1, 1, bias=False)

    def forward(self, x):
        # Conv
        conv_output = self.cv1(x)
        # SPD
        spd = torch.cat([x[..., ::2, ::2],  x[..., 1::2, ::2], 
                         x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        spd_output = self.cv2(spd)
        
        conv_level = self.conv_level(conv_output)
        spd_level = self.spd_level(spd_output)
        levels_weight_v = torch.cat((conv_level, spd_level), 1)
         
        # ECA
        levels_features_erca = self.eca(levels_weight_v)
        levels_weight = self.weight(levels_features_erca)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out = conv_output * levels_weight[:,0:1,:,:] + \
                    spd_output * levels_weight[:,1:2,:,:]

        out = self.cv3(fused_out)
        return out
```
