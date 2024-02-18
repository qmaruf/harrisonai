FROM ubuntu:22.04

RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y python3 python3-pip ffmpeg libsm6 libxext6 wget

WORKDIR /app

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . /app

RUN mkdir -p weights && \
	wget https://huggingface.co/qmaruf/petnet.pth/resolve/main/model.pth -O weights/model.pth && \
	mkdir -p server/uploaded_files && \
	chmod a+rwx -R server/uploaded_files




