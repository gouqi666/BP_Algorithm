from PIL import Image
import numpy as np
import struct
def open_image(data_path):
    buffer = open(data_path,'rb').read()
    index = 0
    magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buffer, index)
    assert numRows == 28
    index += struct.calcsize('>IIII')
    data = []
    for i in range(numImages):
        im = struct.unpack_from('>784B', buffer, index)
        index += struct.calcsize('>784B')
        im = np.array(im).reshape(numRows * numColumns)
        data.append(im)
        # img = Image.fromarray(im.astype('uint8')).convert('RGB')
        # img.show()
    return data
def open_label(label_path):
    buffer = open(label_path,'rb').read()
    index = 0
    magic, numLabels = struct.unpack_from('>II', buffer, index)
    index += struct.calcsize('>II')
    label = []
    for i in range(numLabels):
        lab = struct.unpack_from('>B', buffer, index)
        index += struct.calcsize('>B') # 这里 lab是一个tuple，取出来
        label.append(lab[0])
        # img = Image.fromarray(im.astype('uint8')).convert('RGB')
        # img.show()
    return label
if __name__ == "__main__":
    train_data_path = "./data/train-images.idx3-ubyte"
    train_label = "./data/train-labels.idx1-ubyte"
    train_data = open_image(train_data_path)
    train_label = open_label(train_label)