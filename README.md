# YOLO-RD



## 1. CSAF module
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


## 2. LGECA mechanism
```python
class LGECA(nn.Module):
    def __init__(self, c1, c2, local_size=5, global_k_size=7, local_k_size=3, local_weight=0.5):
        super(LGECA, self).__init__()
        # global attention
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.global_conv = nn.Conv1d(1, 1, kernel_size=global_k_size, padding=(global_k_size-1) // 2, bias=False) 
        # local attention
        self.local_avg_pool = nn.AdaptiveAvgPool2d(local_size)
        self.local_max_pool = nn.AdaptiveMaxPool2d(local_size)
        self.local_conv = nn.Conv1d(1, 1, kernel_size=local_k_size, padding=(local_k_size-1) // 2, bias=False) 
        
        self.local_weight = local_weight
        
    def forward(self, x):
        b,c,h,w = x.shape
        # global attention
        global_y = self.global_avg_pool(x) + self.global_max_pool(x)
        bg,cg,hg,wg = global_y.shape
        gloabl_y_temp = self.global_conv(global_y.view(bg, cg, -1).transpose(-1,-2))
        global_y = gloabl_y_temp.view(bg, -1).unsqueeze(-1).unsqueeze(-1)
        att_global = global_y.sigmoid()
        # local attention
        local_y = self.local_avg_pool(x) + self.local_max_pool(x)
        bl,cl,hl,wl = local_y.shape
        local_y_temp = self.local_conv(local_y.view(bl, cl, -1).transpose(-1, -2).reshape(bl, 1, -1))
        local_y = local_y_temp.reshape(bl, hl*wl, cl).transpose(-1,-2).view(bl,cl,hl,wl)
        att_local = local_y.sigmoid()
        # adaptive weighting
        att_global = F.adaptive_avg_pool2d(att_global, [hl, wl])
        att_all = F.adaptive_avg_pool2d((att_local*self.local_weight + att_global*(1-self.local_weight)), [h, w])
        x = x * att_all
        return x
```

## 3. SR-WBCE loss
```python
count_balance = [math.pow(it, 1/4) for it in counter_per_cls]
cls_weights = [max(count_balance)/it if it != 0 else 1 for it in count_balance]
class_loss = (self.BCEcls(pred_scores, target_scores.to(dtype))*torch.Tensor(cls_weights).cuda()).sum() / target_scores_sum  
```
