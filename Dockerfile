# Start with a lightweight Python 3.11 image
FROM python:3.11.10

# Set the working directory in the container to /app
WORKDIR /app

# Copy only the necessary files to leverage Docker caching for dependencies
COPY pyproject.toml poetry.lock /app/

# Install Poetry
RUN pip install poetry && \
    poetry config virtualenvs.in-project true && \
    poetry install --no-root && \
    poetry shell

# Copy the entire project into the container
COPY . /app

# Expose the port your FastAPI app will run on
EXPOSE 8510

# Command to run FastAPI app with Uvicorn
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8510"]
