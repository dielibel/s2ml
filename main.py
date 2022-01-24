# main.py
#   run s2ml
# by: Noah Syrkis

# imports
import argparse
import sys

# path
sys.path.append('./taming-transformers')
ImageFile.LOAD_TRUNCATED_IMAGES = True
sys.path.append('./CLIP')
sys.path.append('./guided-diffusion')

# call stack
def main():
    parser = argparse.ArgumentParser(description='generate some art.')
    parser.add_argument('--prompts', default="the first page of a new masterpiece")
    parser.add_argument('--width', default=6000)
    parser.add_argument('--height', default=4000)
    parser.add_argument('--clip_model', default="ViT-B/32")
    parser.add_argument('--vqgan_model', default="wikiart_16384")
    parser.add_argument('--initial_image', default=None)
    parser.add_argument('--target_images', default=None)
    parser.add_argument('--max_iterations', default=1000)
    args = parser.parse_args()
    vqgan_clip(args)


if __name__ == "__main__":
    main()
