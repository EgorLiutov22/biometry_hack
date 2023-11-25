FROM python:3.8


WORKDIR /api


COPY ./requirements.txt /code/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /api/requirements.txt


COPY ./app /api/app


CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80"]