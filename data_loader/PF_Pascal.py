import json, os
import tensorflow as tf
from .dataset_class import Dataset

class PF_Pascal_dataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.annotation_dir = os.path.join(root_path, "PF-dataset-PASCAL/Annotations")
        self.image_dir = os.path.join(root_path, "PF-dataset-PASCAL/JPEGImages")
        self.class_names = [name for name in os.listdir(self.annotation_dir) if name != '.DS_Store']
    def process_path(self, path):
        label = get_label(path, self.class_names)
        path = get_imgpath(path, self.image_dir)
        img = decode_img(path)
        return img, label
    def load_classification(self, split=0.2):
        list_ds = tf.data.Dataset.list_files(self.annotation_dir+'/*/*', shuffle=False)
        image_count = len(list_ds); num_val = int(image_count*split);
        list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
        val_ds = list_ds.take(num_val).map(self.process_path)
        train_ds = list_ds.skip(num_val).map(self.process_path)
        return (train_ds, val_ds), self.class_names


def get_label(path, class_names):
    label = tf.strings.split(path, '/')[-2]
    return tf.argmax(label == class_names)
def get_imgpath(path, image_dir):
    filename = tf.strings.split(path, '/')[-1]
    filename = tf.strings.split(filename, '.')[-2]
    image_path = tf.strings.join([image_dir, '/', filename, '.jpg'])
    return image_path
def decode_img(path):
    img = tf.io.read_file(path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    return tf.image.resize(img, [224, 224])
