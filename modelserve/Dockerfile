FROM python:3.12-slim-bullseye

# Set work directory
WORKDIR /app

# Copy only requirements to cache them in docker layer
COPY requirements.txt config.yaml /app/

# Install dependencies
RUN pip install "-r requirements.txt"
# Copy the current directory contents into the container at /app
COPY ./ /app/

CMD [ "serve","run","config.yaml" ]
