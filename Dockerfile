FROM python:3.8

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY extract_faces.py .
COPY demoupload.py .
COPY haarcascade_frontalface_default.xml .
# RUN pip install --no-cache-dir deepface opencv-python numpy
RUN pip install deepface opencv-python numpy flask werkzeug
# RUN mkdir output
EXPOSE 5000

CMD ["python", "./demoupload.py"]


# FROM python:3.8

# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# COPY main.py .
# COPY haarcascade_frontalface_default.xml .
# COPY input/ ./input/
# COPY output/
# RUN pip install --no-cache-dir deepface opencv-python numpy
# EXPOSE 5000
# # RUN mkdir output

# CMD ["python", "./main.py"]
