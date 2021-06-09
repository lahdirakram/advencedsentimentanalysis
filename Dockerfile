FROM pytorch/pytorch
LABEL creator.name="Mohamed Akram LAHDIR"
LABEL creator.email="mohamedakram.lahdir@gmail.com"
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential
COPY /web-app /app
WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
ENTRYPOINT ["waitress-serve"]
CMD ["--port=80","app:app"]