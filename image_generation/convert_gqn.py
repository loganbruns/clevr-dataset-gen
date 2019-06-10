
import argparse
import os
import json
import re
import numpy as np
from imageio import imread
from PIL import Image

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--image_height', default=64, type=int)
parser.add_argument('--image_width', default=64, type=int)
parser.add_argument('--input_image_dir', default='../output')
parser.add_argument('--sequence_length', default=9, type=int)
parser.add_argument('--split', default='train')

def convert(scene_dir, image_dir, output_dir, input_path, output_path, sequence_length, image_size):
    print(f'Converting {input_path} to {output_path}')

    frames = []
    cameras = []

    scene_format = input_path.replace('.json', '_{}.json')
    image_format = input_path.replace('.json', '_{}.png')
    for frame in range(sequence_length):
        with open(os.path.join(scene_dir, scene_format.format(frame))) as scene_file:
            frame_scene = json.load(scene_file)
            cameras += frame_scene['camera_location']
            cameras.append(frame_scene['camera_rotation'][2])
            cameras.append(frame_scene['camera_rotation'][1])
        image = imread(os.path.join(image_dir, image_format.format(frame)), pilmode='RGB')
        image = np.array(Image.fromarray(image).resize(image_size, resample=Image.BICUBIC))
        frames.append(tf.image.encode_jpeg(image, quality=100).numpy())
    frames = tf.train.Feature(bytes_list=tf.train.BytesList(value=frames))
    cameras = tf.train.Feature(float_list=tf.train.FloatList(value=cameras))

    feature = {
        'frames': frames,
        'cameras': cameras
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example

        
def main(args):
    scene_dir = os.path.join(args.input_image_dir, 'scenes')
    image_dir = os.path.join(args.input_image_dir, 'images')
    output_dir = os.path.join(args.input_image_dir, args.split)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    image_size = [args.image_width, args.image_height]

    regex = re.compile(f'CLEVR_{args.split}_[^_]+.json')
    input_files = sorted(filter(regex.match, os.listdir(scene_dir)))

    # TODO: chunk by 5000
    num_files = 1
    length = len(str(num_files))
    template = '{:0%d}-of-{:0%d}.tfrecord' % (length, length)
    output_path = template.format(1, 1)
    with tf.io.TFRecordWriter(os.path.join(output_dir, output_path)) as writer:
        for path in input_files:
            example = convert(scene_dir, image_dir, output_dir, path, output_path, args.sequence_length, image_size)
            writer.write(example.SerializeToString())            

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
