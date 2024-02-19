FROM python:3.10.13-slim-bookworm

WORKDIR /app

COPY api/ ./api

RUN pip install -U pip
RUN pip install -r api/requirements.txt

COPY model/model.pkl ./model/model.pkl

COPY initializer.sh .

RUN chmod +x initializer.sh

EXPOSE 8000

ENTRYPOINT [ "./initializer.sh" ]l