FROM python:3.7

WORKDIR /usr/src/app

COPY models.sh

RUN apt update && apt install -y exempi && \
    mkdir models vqgan-steps diffusion-steps && \
    bash models.sh 

COPY requirements.txt ESRGAN taming-transformers ./

RUN pip install -r requirements.txt

COPY script.py ./

CMD python script.py
