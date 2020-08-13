# Make sure that you have tensorflow/tensorflow:latest-gpu locally: ~3.84GB
FROM tensorflow/tensorflow:latest-gpu

MAINTAINER Ofentse Jabari

ENV PYTHONUNBUFFERED 1

RUN mkdir /app

WORKDIR /app

COPY requirements.txt /app/

RUN pip install -r requirements.txt

COPY . /app/
COPY .data/.technovice /technovice/
# -- All keras data is stored in ~/.keras/ --
COPY .data/.keras /root/.keras/

## Create a user that is going to be used to run processes only called user
#RUN adduser -D user
#RUN chown -R user:user /app/
#RUN chmod -R 755 /app/
#USER user