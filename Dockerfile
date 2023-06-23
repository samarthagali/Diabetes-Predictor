FROM python:3.8

COPY . /home
WORKDIR /home

RUN apt update
RUN pip install -r requirements.txt

CMD python app.py