
import argparse
import os
import json
import re
from datetime import datetime as dt

parser = argparse.ArgumentParser()
parser.add_argument('--output_scene_dir', default='../output/scenes/',
    help="The directory where output JSON scene structures will be stored. " +
         "It will be created if it does not exist.")
parser.add_argument('--output_scene_file', default='../output/CLEVR_scenes.json',
    help="Path to write a single JSON file containing all scene information")
parser.add_argument('--split', default='new',
    help="Name of the split for which we are rendering. This will be added to " +
         "the names of rendered images, and will also be stored in the JSON " +
         "scene structure for each image.")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
    help="String to store in the \"date\" field of the generated JSON file; " +
         "defaults to today's date")
parser.add_argument('--version', default='1.0',
    help="String to store in the \"version\" field of the generated JSON file")
parser.add_argument('--license',
    default="Creative Commons Attribution (CC-BY 4.0)",
    help="String to store in the \"license\" field of the generated JSON file")

def main(args):
    regex = re.compile(f'CLEVR_{args.split}_[^_]+.json')
    input_files = sorted(filter(regex.match, os.listdir(args.output_scene_dir)))

    all_scenes = []
    for scene_path in input_files:
        with open(os.path.join(args.output_scene_dir, scene_path), 'r') as f:
            all_scenes.append(json.load(f))
    output = {
        'info': {
            'date': args.date,
            'version': args.version,
            'split': args.split,
            'license': args.license,
        },
        'scenes': all_scenes
    }
    with open(args.output_scene_file, 'w') as f:
        json.dump(output, f)

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
