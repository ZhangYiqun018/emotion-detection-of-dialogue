import numpy as np

if __name__ == '__main__':
    data = np.load('./train_data.npy', allow_pickle=True)
    print(data, data.shape)
