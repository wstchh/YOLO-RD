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