FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Copy and install requirements first for caching
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application files
COPY ./GB_stu_mob.pth GB_stu_mob.pth
COPY ./model.py model.py
COPY ./app.py app.py

# Expose Render's default port
EXPOSE 10000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]