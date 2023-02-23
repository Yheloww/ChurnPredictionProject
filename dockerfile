FROM python:3.10

RUN mkdir /app
WORKDIR /app
COPY . /app
RUN pip install --trusted-host pypi.python.org -r requirements.txt
EXPOSE 5000
ENV FLASK_APP app.py
CMD ["flask", "run", "--host=0.0.0.0"]