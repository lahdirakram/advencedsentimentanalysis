From ubuntu:latest
LABEL creator.name="Mohamed Akram LAHDIR" 
LABEL creator.email="mohamedakram.lahdir@gmail.com" 
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
COPY /web-app/* /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]