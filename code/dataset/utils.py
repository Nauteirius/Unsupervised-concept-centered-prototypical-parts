import cv2
import os
import pandas as pd
from PIL import Image
import torchvision.utils


def transform_to_show(data):
    img, label = data
    return img.permute(1, 2, 0)


# Function takes all images in ./data/CUB_200_2011/images then crops them according to
# ./data/CUB_200_2011/bounding_boxes.txt and saves them in ./data/CUB_200_2011/cropped
def crop_all_images(root):
    data_path = os.path.join(root, 'CUB_200_2011')
    img_source = os.path.join(data_path, 'images')
    img_target = os.path.join(data_path, 'cropped')
    boxes = pd.read_csv(data_path + "/bounding_boxes.txt", header=None, sep=" ")
    boxes.columns = ["id", "x", "y", "width", "height"]
    boxes = [x for x in zip(boxes.x.astype('int'), boxes.y.astype('int'),
                            boxes.width.astype('int'), boxes.height.astype('int'))]
    imgs_in_order = pd.read_csv(data_path + "/images.txt", header=None, sep=" ")
    imgs_in_order.columns = ["id", "name"]
    imgs_in_order = [x for x in imgs_in_order.name]
    if not os.path.exists(img_target):
        os.mkdir(img_target)
    for idx, path in enumerate(imgs_in_order):
        image_path = os.path.join(img_source, path)
        directory, filename = os.path.split(path)
        output_dir = os.path.join(img_target, directory)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        cropped_path = os.path.join(output_dir, filename)
        x, y, width, height = boxes[idx]
        image = cv2.imread(image_path)
        cropped_image = image[y:y + height, x:x + width]
        cv2.imwrite(cropped_path, cropped_image)


# Function takes cropped images applies given augmentation and saves them according to given name
def augmentation_on_all_images(root, augmentation, name):
    data_path = os.path.join(root, 'CUB_200_2011')
    img_source = os.path.join(data_path, 'cropped')
    img_target = os.path.join(data_path, name)
    number = 0
    for root, dirs, files in os.walk(img_source):
        output_subdir = root.replace(img_source, img_target)
        if not os.path.exists(output_subdir):
            os.mkdir(output_subdir)
        for filename in files:
            image_path = os.path.join(root, filename)
            image = Image.open(image_path)
            aug = augmentation(image)
            aug_path = os.path.join(output_subdir, filename)
            torchvision.utils.save_image(aug, aug_path)

