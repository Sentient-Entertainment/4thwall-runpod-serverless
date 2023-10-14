FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN mkdir data
WORKDIR /data

# Install Python dependencies (Worker Template)
COPY requirements.txt /data/requirements.txt
RUN pip install --upgrade pip && \
    pip install safetensors==0.3.1 sentencepiece huggingface_hub accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.34.0 trl==0.4.7 scipy spacy==3.2.0 openai zss  \
    git+https://github.com/winglian/runpod-python.git@fix-generator-check ninja==1.11.1 \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.2.0/en_core_web_sm-3.2.0.tar.gz
# RUN git clone https://github.com/turboderp/exllama
# RUN pip install -r exllama/requirements.txt

COPY handler.py /data/handler.py
COPY __init.py__ /data/__init__.py

# ENV PYTHONPATH=/data/exllama
ENV MODEL_REPO=""
ENV PROMPT_PREFIX=""
ENV PROMPT_SUFFIX=""
ENV HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub"
ENV TRANSFORMERS_CACHE="/runpod-volume/huggingface-cache/hub"

CMD [ "python", "-m", "handler" ]