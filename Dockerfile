FROM python:3.7

MAINTAINER dielibel@syrkis.com

WORKDIR /usr/src/app

COPY . .

RUN \
    git clone https://github.com/openai/CLIP && \
    git clone https://github.com/dielibel/guided-diffusion && \
    python -m pip install --upgrade pip && \
    pip install -e ./CLIP && \
    pip install -e ./guided-diffusion && \
    git clone https://github.com/CompVis/taming-transformers && \
    pip install ftfy regex tqdm omegaconf pytorch-lightning && \
    pip install kornia && \
    pip install einops && \
    pip install transformers && \
    pip install stegano && \
    apt update && apt install -y exempi && \
    pip install python-xmp-toolkit && \
    pip install imgtag && \
    pip install pillow==7.1.2 && \
    pip install taming-transformers && \
    git clone https://github.com/xinntao/ESRGAN && \
    pip install imageio-ffmpeg

ENTRYPOINT ["python", "main.py"]

CMD ["no_config_specified"]
