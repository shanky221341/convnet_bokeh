import numpy as np


class Convolution:
    """

    """

    def __init__(self, channel, stride, filter_size, filter):
        """

        :param input_matrix:
        :param stride:
        :param filter_size:
        """
        self.channel = channel
        self.stride = stride
        self.filter_size = filter_size
        self.filter = filter
        self._index_list = None
        self._vertical_start_pos = None
        self._v_mat = None
        self.convolve_res_ver = None

    def convolute(self, verbose):
        self._calculate_sub_mat_indexs()
        self._find_submatrices()
        self.perform_convolve_operation(verbose=verbose)

    def _calculate_sub_mat_indexs(self):
        index_list = []
        vertical_start_pos = []
        i = 0
        j = 0
        while j < self.channel.shape[1] - 1:
            j = (i + self.filter_size - 1)
            index_list.append((i, j))
            vertical_start_pos.append(i)
            i += self.stride
        self._index_list = index_list
        self._vertical_start_pos = vertical_start_pos

    def _find_submatrices(self):
        v_mat = []
        for v_start in self._vertical_start_pos:
            h_mat = []
            for indexes in self._index_list:
                sub_matrice = []
                for vertical_start in range(v_start, v_start + self.filter_size):
                    sub_matrice.append(self.channel[vertical_start,][indexes[0]:indexes[1] + 1])
                h_mat.append(np.array(sub_matrice))
            v_mat.append(h_mat)
        self._v_mat = v_mat

    def print_sub_mats(self):
        for h_mat_set in self._v_mat:
            aa = h_mat_set
            cc = ''
            for j in range(aa[0].shape[0]):
                for i in range(len(aa)):
                    cc = cc + str(aa[i][j]) + " "
                cc = cc + '\n'
            print(cc)

    def perform_convolve_operation(self, verbose=False):
        convolve_res_ver = []
        mat_index = 0
        for h_mats in self._v_mat:
            mat_index += 1
            if verbose:
                print("Showing Convolution for horizontal stack {}".format(mat_index))
            convolve_res_hor = []
            for h_mat in h_mats:
                tmp = h_mat
                res = tmp * self.filter
                convolve_res_hor.append(np.sum(res))
                c = '    I/P           K           O/P\n'
                mid_index = (len(list(range(tmp.shape[0]))) - 1) / 2
                for i in range(tmp.shape[0]):
                    c = c + str(tmp[i])
                    if i == mid_index:
                        c = c + ' *'
                    c = c + "   " + str(self.filter[i])
                    if i == mid_index:
                        c = c + ' = '
                    c = c + "  " + str(res[i])
                    if i == mid_index:
                        c = c + ' -> sum(O/P)-> ' + str(np.sum(res))
                    c = c + '\n'
                    #     mat_index+=1
                if verbose:
                    print(c)
            convolve_res_ver.append(convolve_res_hor)
        self.convolve_res_ver = np.array(convolve_res_ver)


class ConvolutionLayer():
    def __init__(self, input_matrix,
                 kernel, stride, bias):
        # super().__init__(input_matrix, stride, filter_size, filter)
        self.input_matrix = input_matrix
        self.kernel = kernel
        self.stride = stride
        self.bias = bias
        self.input_matrix_height = None
        self.input_matrix_width = None
        self.input_matrix_depth = None
        self.kernel_height = None
        self.kernel_width = None
        self.kernel_depth = None
        self.number_of_kernels = None
        self.layer_output = None
        self.all_kernel_convolution_store = None

    def forward_pass(self):
        self._check_shape_input()
        self._check_shape_kernel()
        self._check_shape_bias()
        self.input_matrix_height = self.input_matrix.shape[1]
        self.input_matrix_width = self.input_matrix.shape[2]
        self.input_matrix_depth = self.input_matrix.shape[0]
        self.kernel_height = self.kernel.shape[2]
        self.kernel_width = self.kernel.shape[3]
        self.kernel_depth = self.kernel.shape[1]
        self.number_of_kernels = self.kernel.shape[0]
        bias_index = 0
        all_kernel_convolution_store = {}
        layer_output = []
        for kernel in self.kernel:
            kernel_convolution_store = []
            kernel_channels_final_conv = 0
            for channel_index in range(kernel.shape[0]):
                filt = kernel[channel_index]
                channel = self.input_matrix[channel_index]
                convolution_op = Convolution(channel=channel, stride=self.stride, filter_size=self.kernel_height,
                                             filter=filt)
                convolution_op.convolute(verbose=False)
                kernel_convolution_store.append(convolution_op)
                kernel_channels_final_conv += convolution_op.convolve_res_ver  # All channels get added after individual convolution
            kernel_channels_final_conv = kernel_channels_final_conv + self.bias[bias_index] # add kernel bias to the final channel convolution output
            bias_index += 1  # Note that bias index is same as kernel index, when iterating over kernels
            all_kernel_convolution_store[bias_index] = kernel_convolution_store
            layer_output.append(kernel_channels_final_conv)
        self.layer_output = np.array(layer_output)
        self.all_kernel_convolution_store = all_kernel_convolution_store

    def _check_shape_input(self):
        input_shape = self.input_matrix.shape
        if len(input_shape) != 3:
            raise ValueError("""Shape of input matrix is {}. Input matrix should 
                             contain number of channels,height, width""".format(input_shape))
        if input_shape[1] != input_shape[2]:
            raise ValueError(""" Shape of input matrix is {}. ConvLayer can accept only images 
            which are square in shape.
            """.format(input_shape))

    def _check_shape_bias(self):
        bias_shape = self.bias.shape
        if len(bias_shape) != self.kernel.shape[0]:
            raise ValueError(""" Length of bias ->{} should be equal
             to number of kernels -> {}""".format(bias_shape, self.kernel.shape[0]))

    def _check_shape_kernel(self):
        kernel_shape = self.kernel.shape
        input_shape = self.input_matrix.shape
        if len(kernel_shape) != 4:
            raise ValueError("""Shape of kernel matrix is {}. Kerenl matrix should 
                             contain number of kerenels, number of channels,height,width""".format(kernel_shape))
        if kernel_shape[2] != kernel_shape[3]:
            raise ValueError(""" Shape of kernel matrix is {}. ConvLayer can accept only square kernels.
            """.format(kernel_shape))
        if kernel_shape[1] != input_shape[0]:
            raise ValueError(""" Number of channels in input image ->{} is not equal to number of channels in kernel
              ->{}.""".format(input_shape[0], kernel_shape[1]))


input_matrix = np.array([[1, 1, 1, 0, 0],
                         [0, 1, 1, 1, 0],
                         [0, 0, 1, 1, 1],
                         [0, 0, 1, 1, 0],
                         [0, 1, 1, 0, 0]],
                        dtype='float')

stride = 1
filter_size = 3
filt = np.random.choice([-1, 1, 0], size=[filter_size, filter_size])
convolution = Convolution(input_matrix=input_matrix, stride=stride, filter_size=filter_size, filter=filt)
res = convolution.forward_pass(True)
print(res)
# index_list, vertical_start_pos = calculate_sub_mat_indexs(x, 1, 3)
# v_mat = find_submatrices(x, index_list, vertical_start_pos, 3)
# print_sub_mats(v_mat)
