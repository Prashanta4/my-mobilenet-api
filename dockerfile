FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Copy and install requirements first for caching
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application files
COPY ./app/GB_stu_mob.pth app/GB_stu_mob.pth
COPY ./app/model.py app/model.py
COPY ./app/main.py app/main.py
COPY ./app/__init__.py app/__init__.py

# Expose port
EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]