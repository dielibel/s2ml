# main.py
#   run s2ml
# by: Noah Syrkis

# imports
import argparse

def parse():
    parser = argparse.ArgumentParser(description="Visual GAN")
    parser.add_argument('--project')
    parser.add_argument('--prompt', default="")
    parser.add_argument('--clip_model', default="")
    parser.add_argument('--vqgan_model', default="")
    parser.add_argument('--diff', default=False)
    parser.add_argument('--width', default=600)
    parser.add_argument('--height', default=400)
    parser.add_argument('--video', default=False)
    parser.add_argument('--upscale', default=False)
    return parser.parse_args()


# call stack
def main():
    args = parse()

if __name__ == "__main__":
    main()
