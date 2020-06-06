FROM python:3.7

RUN apt-get update -y
RUN apt-get install -y python3-pip

COPY . /opt

WORKDIR /opt

EXPOSE 5000

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT ["python3"]

CMD ["app.py"]