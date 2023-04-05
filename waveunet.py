import torch
import torch.nn as nn
# Adapted from https://github.com/f90/Wave-U-Net-Pytorch

class InterpolationLayer(nn.Module):
    def __init__(self, n_ch, interp_kernel_size=5):
        super(InterpolationLayer, self).__init__()
        self.interp_kernel_size = interp_kernel_size
        self.interp_filter = nn.Conv1d(n_ch, n_ch, self.interp_kernel_size, padding='same')

    def forward(self, x):
        x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        out = self.interp_filter(x)
        return out

    
class Waveunet(nn.Module):
    def __init__(self, n_src, n_inputs=1, kernel_size=15, merge_kernel_size=5, long_kernel_size=101, k_neurons=24, depth=12, n_first_filter=42):
        super(Waveunet, self).__init__()
        self.num_inputs = n_inputs
        self.num_outputs = n_src
        self.depth = depth
        
        self.downsample_convs = nn.ModuleList([nn.Conv1d(self.num_inputs, n_first_filter*k_neurons, long_kernel_size, padding='same')] +
                                              [nn.Conv1d(n_first_filter*k_neurons, 2*k_neurons, kernel_size, padding='same')] +
                                              [nn.Conv1d((i+2)*k_neurons, (i+3)*k_neurons, kernel_size, padding='same') for i in range(depth - 2)])

        self.latent_conv = nn.Conv1d((depth)*k_neurons, (depth+1)*k_neurons, kernel_size, padding='same')

        self.upsample_convs = nn.ModuleList([InterpolationLayer((i+1)*k_neurons) for i in range(depth, 0, -1)])
        self.post_shortcut_convs = nn.ModuleList([nn.Conv1d((2*i+1)*k_neurons, (i)*k_neurons, merge_kernel_size, padding='same') for i in range(depth, 1, -1)] + 
                                                [nn.Conv1d((2+n_first_filter)*k_neurons, k_neurons, merge_kernel_size, padding='same')])
        
        self.output_conv = nn.Conv1d(k_neurons+1, self.num_outputs, 1, padding='same')
        
    def forward(self, x_in, inst=None):
        x = x_in
        
        all_shortcuts = []
        # DOWNSAMPLING
        for conv in self.downsample_convs:
            # SHORTCUT FEATURES
            shortcut = nn.LeakyReLU(0.3)(conv(x))
            all_shortcuts.append(shortcut)
            # DECIMATION
            x = shortcut
            x = x[:,:,::2]
            
        x = nn.LeakyReLU(0.3)(self.latent_conv(x))
        
        for k in range(len(all_shortcuts)):
            upsampled = x
            upconv = self.upsample_convs[k]
            upsampled = upconv(upsampled)
            conv = self.post_shortcut_convs[k]
            combined = nn.LeakyReLU(0.3)(conv(torch.cat([upsampled, all_shortcuts[-(k+1)]], dim=1)))
            x = combined
        
        x = torch.cat([x, x_in], dim=1)
        x_out = self.output_conv(x)
        return x_out