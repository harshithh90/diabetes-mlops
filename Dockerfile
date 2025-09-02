# use python Base image
FROM python:3.8-slim

# set the working directory
WORKDIR /app

# Copy all necessary files
COPY requirements.txt .
COPY app/ ./app/
COPY diabetes.csv .
COPY train.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Train the model (ensure diabetes_model.pkl is saved in /app)
RUN python train.py

# expose the port
EXPOSE 8000

# run the application
CMD ["python", "app/app.py"]

