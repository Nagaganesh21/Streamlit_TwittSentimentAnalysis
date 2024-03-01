FROM python:3.10.4

RUN pip install --upgrade pip

WORKDIR /app
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

RUN python -m nltk.downloader punkt

# Expose port
ENV PORT 8089

ENTRYPOINT ["python"]

# Run the application:
CMD ["gunicorn", "app:app", "--config=config.py"]