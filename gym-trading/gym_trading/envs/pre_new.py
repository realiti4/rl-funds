import os
import numpy as np
import pandas as pd
import sys
# sys.path.insert(1, os.getcwd()+'/utils/')

# from tools import conv_time_v2, column_order
from numba import njit
from matplotlib import pyplot as plt

np.set_printoptions(suppress=True)


class preprocessing_v3:
    def __init__(self, df=None, columns='close', custom_target=None, float32=False):
        self.df_array = df[columns].values
        self.features = len(columns)
        self.percentage_change = False
        self.smoothing = 1

        if float32: self.df_array, self.dtype = self.df_array.astype('float32'), 'float32'
        if custom_target is not None: self.custom_target = custom_target
        print('numpy dtype: ', self.df_array.dtype)
        
        if len(columns) >= 2:
            self.multi_variable = True
            # print('heyo, multi features detected!')
        else: self.multi_variable=False

    def temp_handler(self, convert_to_log=1, normalize=True):
        array = self.build_array_v2(convert_to_log, normalize)

        return array

    def make_windows(self, history_size, target_size, array=None, multi_step=True, 
                        center_zero=False, percentage_change=False, smoothing=1, convert_to_log=1, center_to=-1, normalize=True, 
                        evaluate=False):
        self.percentage_change = percentage_change
        self.smoothing = smoothing

        if self.percentage_change: center_zero=False
        if array is None: array = self.build_array_v2(convert_to_log, normalize)
        # if evaluate: return self.build_array_v2(history_size, convert_to_log, normalize=False)

        new_len = len(array)-history_size-target_size        

        array_train = np.zeros((new_len, history_size, self.features))
        array_target = np.zeros((new_len, target_size))   # multiple targets
        if not multi_step:
            array_target = np.zeros((new_len))  # single target

        for i in range(new_len):            
            array_train[i] = array[i:history_size+i]
            if multi_step:
                array_target[i] = array[i+history_size:i+history_size+target_size][:,0]  # multiple targets
            else:
                if self.custom_target is not None:
                    array_target[i] = self.custom_target[i+history_size]
                else:
                    array_target[i] = array[i+history_size+target_size][0]   # single future target

            if center_zero:
                array_target[i] = array_target[i] - array_train[i][center_to][0]    # -1 target 0, 0 başlangıç 0
                # array_train[i][:,0] = array_train[i][:,0] - array_train[i][center_to][0]
                # TODO add and test multi centering
                for ie in range(0, convert_to_log):                
                    array_train[i][:,ie] = array_train[i][:,ie] - array_train[i][center_to][ie]                

        return array_train, array_target

    def build_array_v2(self, convert_to_log, normalize=True):
        self.norm_values = np.zeros((self.features, 2))

        # TODO better nan management
        # self.smoothing = 14
        # df_array_cut = self.df_array[self.smoothing-1:]

        for i in range(0, convert_to_log):
            self.df_array[:, i] = self.log_return(self.df_array[:,i])
        #     df_array_cut[:, i] = self.log_return(self.df_array[:,i])
        # self.df_array = df_array_cut

        if normalize:
            for i in range(self.features):
                self.df_array[:, i], self.norm_values[i] = self.normalize_z_score(self.df_array[:, i])
            
        return self.df_array

    def log_return(self, close_array):      
        log_array =  np.zeros((2, len(close_array)))        
        for i in range(1, len(close_array)):
            log_array[0][i] = (np.log(close_array[i]/close_array[i-1]))
            log_array[1][i] = log_array[0][i]+log_array[1][i-1]
        if self.percentage_change:
            # TODO try rolling log change
            log_change = pd.Series(log_array[0])
            if not self.smoothing == 1:
                log_change = log_change.ewm(span=self.smoothing, adjust=False).mean()                
                # # Holt
                # from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
                # log_change = Holt(log_change).fit(smoothing_level=0.2, smoothing_slope=0.2, optimized=False)
                # log_change = log_change.fittedvalues

            a = log_change.values

            for i in range(self.smoothing-1):
                a[i] = log_array[0][i]
            
            return a
        return log_array[1]

    def normalize_z_score(self, data):
        mean = data.mean()
        std = data.std()
        data = (data-mean)/std
        return data, [mean, std]


# Data Balancers
def data_balancer(train, target):
    # TODO check for errors make it clearer
    # np.random.seed(44)
    ups_downs = [len(target)-int(np.sum(target)), int(np.sum(target))]

    if ups_downs[0] > ups_downs[1]:
        majority = 0
        minority_len = ups_downs[1]
    else:
        majority = 1
        minority_len = ups_downs[0]

    up_down_map = np.zeros(len(target))

    for i in range(len(target)):
        if target[i] == majority:   # TODO check here later
            up_down_map[i] = 1

    while True:
        select = np.random.randint(len(train))
        up_down_map[select] = 0
        if up_down_map.sum() == minority_len:
            break

    mask_delete = np.ones(len(target), dtype=bool)
    for i in range(len(target)):
        if target[i] == 0 and up_down_map[i] == 0:
            mask_delete[i] = False

    train = train[mask_delete]
    target = target[mask_delete]

    up = 0
    down = 0
    for i in range(len(target)):
        if target[i] == 0:
            up += 1
        elif target[i] == 1:
            down += 1

    print(f'balanced.. up: {up}, down: {down}')

    return train, target

def data_balancer_add(train, target):
    ups_downs = [len(target)-int(np.sum(target)), int(np.sum(target))]

    if ups_downs[0] > ups_downs[1]:
        majority, minority = 0, 1
        majority_len, minority_len = ups_downs[0], ups_downs[1]
    else:
        majority = 1
        minority_len = ups_downs[0]

    select_list = []

    while True:
        select = np.random.randint(len(train))

        if target[select] == minority:        
            if select not in select_list:
                select_list.append(select)

        if len(select_list) == (majority_len-minority_len):
            break
    
    add = np.zeros([len(select_list), train.shape[1], train.shape[2]])
    add_target = np.zeros([len(select_list), target.shape[1]])
    for i in range(len(select_list)):
        add[i] = train[select_list[i]]
        add_target[i] = target[select_list[i]]
    
    return np.concatenate((train, add), 0), np.concatenate((target, add_target), 0)


# Embedding
@njit
def find_index(array, item, near_index):
    for i in range(near_index, len(array)):
        if array[i] == item:
            return i
        elif (item - array[i]) < 0:
            return i-1

@njit(parallel=True)
def find_nearest(array, item):
    # TODO optimize
    return (np.abs(array-item)).argmin()

def convert_to_embedded(array, norm_values):
    embed_lookup = np.arange(-.4, .4, 0.001)
    embed_lookup = (embed_lookup - norm_values[0][0]) / norm_values[0][1]

    # embed_one_hot = np.zeros([array.shape[0], array.shape[1], len(embed_lookup)], dtype=np.int8)
    embed_array = np.zeros([array.shape[0], array.shape[1], 1], dtype=np.float32)
    embed_index = np.zeros([array.shape[0], array.shape[1], 1], dtype=np.int_)

    # index = find_nearest(embed_lookup, array[0, 1, 0])

    for i in range(len(array)):
        for ie in range(len(array[0])):
            index = find_nearest(embed_lookup, array[i][ie][0])
            # embed_one_hot[i][ie][index] = 1
            embed_array[i][ie] = embed_lookup[index]
            embed_index[i][ie] = index

    return embed_array, embed_index


# Convert to fft
def fft_convert(array):
    close_fft = np.fft.fft(array)
    fft_df = pd.DataFrame({'fft':close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

    fft_list = np.asarray(fft_df['fft'].tolist())

    output = []
    for num_ in [6, 9, 14, 60]:
        fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
        output.append(np.fft.ifft(fft_list_m10))
        # plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
    return output

def fft_dataset(array):
    array_train = np.zeros([array.shape[0], array.shape[1], 4], dtype=np.complex128)

    for i in range(len(array_train)):
        fft_list = fft_convert(array[i, :, 0])
        for ie in range(4):
            array_train[i][:, ie] = fft_list[ie]

    return array_train.astype(np.float32)
