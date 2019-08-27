import numpy as np


class Convolution:
    """

    """

    def __init__(self, input_matrix, stride, filter_size, filter):
        """

        :param input_matrix:
        :param stride:
        :param filter_size:
        """
        self.input_matrix = input_matrix
        self.stride = stride
        self.filter_size = filter_size
        self.filter = filter
        self._index_list = None
        self._vertical_start_pos = None
        self._v_mat = None

    def forward_pass(self, verbose):
        self._calculate_sub_mat_indexs()
        self._find_submatrices()
        convolve_res_ver = self.perform_convolve_operation(verbose=verbose)
        return convolve_res_ver

    def _calculate_sub_mat_indexs(self):
        index_list = []
        vertical_start_pos = []
        i = 0
        j = 0
        while j < self.input_matrix.shape[1] - 1:
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
                    sub_matrice.append(self.input_matrix[vertical_start,][indexes[0]:indexes[1] + 1])
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
        return np.array(convolve_res_ver)


class ConvolutionLayer():
    def __init__(self, input_matrix_height, input_matrix_width, input_matrix_depth):
        # super().__init__(input_matrix, stride, filter_size, filter)
        self.input_matrix_height = input_matrix_height
        self.input_matrix_width = input_matrix_width
        self.input_matrix_depth = input_matrix_depth
        self.kernel_height=kernel_height
        self.kernel_width=kernel_width
        self.kernel_depth=kernel_depth


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
