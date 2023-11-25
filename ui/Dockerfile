FROM python:3.8


WORKDIR /ui


COPY ./requirements.txt /ui/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /ui/requirements.txt


COPY ./app /ui/app


CMD ["python", "main.py"]