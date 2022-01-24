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

RUN \
    mkdir models && \
    curl -L -o models/wikiart_16384.ckpt -C - 'http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.ckpt' && \
    curl -L -o models/wikiart_16384.yaml -C - 'http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.yaml'

ENTRYPOINT ["python", "main.py"]

CMD ls models
