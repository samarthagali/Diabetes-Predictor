FROM python:3.8

COPY . /home
WORKDIR /home

RUN pip install -r requirements.txt
EXPOSE 5000

CMD python app.py