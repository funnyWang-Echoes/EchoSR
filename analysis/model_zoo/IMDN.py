import numpy as np
import torch.nn as nn
import tqdm

from analysis.model_zoo.IMDN_block import IMDModule_Large
import IMDN_block as B
import torch




# For any upscale factors
class IMDN_AS(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_modules=6, out_nc=3, upscale=4):
        super(IMDN_AS, self).__init__()

        self.fea_conv = nn.Sequential(B.conv_layer(in_nc, nf, kernel_size=3, stride=2),
                                      nn.LeakyReLU(0.05),
                                      B.conv_layer(nf, nf, kernel_size=3, stride=2))

        # IMDBs
        self.IMDB1 = B.IMDModule(in_channels=nf)
        self.IMDB2 = B.IMDModule(in_channels=nf)
        self.IMDB3 = B.IMDModule(in_channels=nf)
        self.IMDB4 = B.IMDModule(in_channels=nf)
        self.IMDB5 = B.IMDModule(in_channels=nf)
        self.IMDB6 = B.IMDModule(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)


    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.IMDB1(out_fea)
        out_B2 = self.IMDB2(out_B1)
        out_B3 = self.IMDB3(out_B2)
        out_B4 = self.IMDB4(out_B3)
        out_B5 = self.IMDB5(out_B4)
        out_B6 = self.IMDB6(out_B5)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr)
        return output

class IMDN(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_modules=6, out_nc=3, upscale=4):
        super(IMDN, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)

        # IMDBs
        self.IMDB1 = B.IMDModule(in_channels=nf)
        self.IMDB2 = B.IMDModule(in_channels=nf)
        self.IMDB3 = B.IMDModule(in_channels=nf)
        self.IMDB4 = B.IMDModule(in_channels=nf)
        self.IMDB5 = B.IMDModule(in_channels=nf)
        self.IMDB6 = B.IMDModule(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)


    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.IMDB1(out_fea)
        out_B2 = self.IMDB2(out_B1)
        out_B3 = self.IMDB3(out_B2)
        out_B4 = self.IMDB4(out_B3)
        out_B5 = self.IMDB5(out_B4)
        out_B6 = self.IMDB6(out_B5)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr)
        return output

# AI in RTC Image Super-Resolution Algorithm Performance Comparison Challenge (Winner solution)
class IMDN_RTC(nn.Module):
    def __init__(self, in_nc=3, nf=12, num_modules=5, out_nc=3, upscale=2):
        super(IMDN_RTC, self).__init__()

        fea_conv = [B.conv_layer(in_nc, nf, kernel_size=3)]
        rb_blocks = [B.IMDModule_speed(in_channels=nf) for _ in range(num_modules)]
        LR_conv = B.conv_layer(nf, nf, kernel_size=1)

        upsample_block = B.pixelshuffle_block
        upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)

        self.model = B.sequential(*fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),
                                  *upsampler)

    def forward(self, input):
        output = self.model(input)
        return output


class IMDN_RTE(nn.Module):
    def __init__(self, upscale=2, in_nc=3, nf=20, out_nc=3):
        super(IMDN_RTE, self).__init__()
        self.upscale = upscale
        self.fea_conv = nn.Sequential(B.conv_layer(in_nc, nf, 3),
                                      nn.ReLU(inplace=True),
                                      B.conv_layer(nf, nf, 3, stride=2, bias=False))

        self.block1 = IMDModule_Large(nf)
        self.block2 = IMDModule_Large(nf)
        self.block3 = IMDModule_Large(nf)
        self.block4 = IMDModule_Large(nf)
        self.block5 = IMDModule_Large(nf)
        self.block6 = IMDModule_Large(nf)

        self.LR_conv = B.conv_layer(nf, nf, 1, bias=False)

        self.upsampler = B.pixelshuffle_block(nf, out_nc, upscale_factor=upscale**2)

    def forward(self, input):

        fea = self.fea_conv(input)
        out_b1 = self.block1(fea)
        out_b2 = self.block2(out_b1)
        out_b3 = self.block3(out_b2)
        out_b4 = self.block4(out_b3)
        out_b5 = self.block5(out_b4)
        out_b6 = self.block6(out_b5)

        out_lr = self.LR_conv(out_b6) + fea

        output = self.upsampler(out_lr)

        return output


def build(upscale=4):
    return IMDN(upscale=upscale)


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model instance
    model = build().to(device).eval()

    # Generate random input data
    input_shape = (1, 3, 1024, 1024)  # Batch size of 1
    input_data = torch.randn(input_shape).to(device)

    # Set number of inferences
    num_inferences = 50

    # Warm up
    print('Warming up...\n')
    with torch.inference_mode():
        for _ in range(5):
            _ = model(input_data)

    # Reset CUDA memory stats
    torch.cuda.reset_peak_memory_stats()

    # Initialize CUDA events
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((num_inferences, 1))

    # Measure inference time
    print('Measuring inference time...\n')
    with torch.inference_mode():
        for i in tqdm.tqdm(range(num_inferences)):
            starter.record()
            output = model(input_data)
            ender.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded
            timings[i] = starter.elapsed_time(ender)  # Time in milliseconds

    average_inference_time = np.mean(timings)

    # Calculate memory usage
    memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
    max_memory_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)  # MB

    # Calculate parameters and MACs using thop
    macs, params = profile(model, inputs=(input_data,))

    # Output results
    print(f"Input shape: {input_shape}")
    print(f"Output shape: {output.shape}")
    print(f"Average inference time over {num_inferences} runs: {average_inference_time:.4f} ms")
    print(f"Memory allocated: {memory_allocated:.2f} MB")
    print(f"Max memory allocated: {max_memory_allocated:.2f} MB")
    print(f"Max memory reserved: {max_memory_reserved:.2f} MB")
    print(f"Number of parameters: {params / 1e3:.2f} K")
    print(f"MACs: {macs / 1e9:.2f} G")


if __name__ == '__main__':
    from thop import profile
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()