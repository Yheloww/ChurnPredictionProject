FROM python:3.10

RUN mkdir /app
WORKDIR /app
COPY . /app
RUN pip install --trusted-host pypi.python.org -r requirements.txt
EXPOSE 5000
ENV FLASK_APP app.py
CMD ["flask", "run", "--host=0.0.0.0"]

#docker build -t churn_prediction_app .
#docker run -p 5000:5000 churn_prediction_app
