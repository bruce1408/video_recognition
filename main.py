from __future__ import division, print_function
import argparse
from ai import video_demo_v
import sys
sys.path.append('../')
parser = argparse.ArgumentParser(description="save_model.pb detect people")
parser.add_argument("-f", "--fps", type=str, default=5, help="frames")
parser.add_argument('--num-frames', default=8, type=int)
parser.add_argument('--sampling-rate', default=1, type=int)
args = parser.parse_args()


def run(fps):
    video_demo_v.run(fps)


if __name__ == "__main__":
    run(args.num_frames * args.sampling_rate)


