

class Architecture:

    # must be a power of 2
    # assumes input and output images are same size
    # assumes images are square
    img_size = 256 

    input_channels = 1
    output_channels = 3

    # must be a power of 2
    # larger values add more parameters to the network
    # caps the number of convolution filters in generator layers to:
        # (max_channel_multiplier * num_input_channels) in the encoder
        # (2 * max_channel_multiplier * num_input_channels) in the decoder
    max_channel_multiplier = 8 

    # dropouts for generator decoder
    # length must match the number of layers in generator decoder
    # last value must be 0.0, as the last layer should not have dropout
    dropouts = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
