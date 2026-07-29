# Use an official Python runtime as a parent image
FROM python:3.11-slim-bullseye

# Set the working directory in the container
WORKDIR /influencer-automation-2.0

# 设置/influencer-automation-2.0目录权限为777
RUN chmod 777 /influencer-automation-2.0

ENV PYTHONPATH="/influencer-automation-2.0"

# Install system dependencies with retry logic (official Debian mirrors)
RUN ( \
        for i in 1 2 3; do \
            echo "Attempt $i: installing system dependencies"; \
            apt-get update && apt-get install -y --no-install-recommends \
                git \
                ffmpeg && break || \
            echo "Attempt $i failed, retrying..."; \
            sleep 5; \
        done \
    ) && rm -rf /var/lib/apt/lists/*

# Copy only the requirements.txt first to leverage Docker cache
COPY requirements.txt ./

# Install Python dependencies from the official PyPI.
RUN pip install --no-cache-dir --retries 3 --timeout 60 -r requirements.txt

# Now copy the rest of the codebase into the image
COPY . .

# Expose the port the app runs on
EXPOSE 8501

# 容器内部必须监听 0.0.0.0，宿主机仍通过 docker 端口映射限制为 127.0.0.1。
# browser.serverAddress 只决定浏览器展示的访问地址，不能替代 server.address。
CMD ["streamlit", "run", "./webui/Main.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.serverAddress=127.0.0.1", "--server.enableCORS=True", "--browser.gatherUsageStats=False", "--client.toolbarMode=minimal", "--logger.hideWelcomeMessage=True", "--server.showEmailPrompt=False"]

# 1. Build the Docker image using the following command
# docker build -t influencer-automation-2.0 .

# 2. Run the Docker container using the following command
## For Linux or MacOS:
# docker run -v $(pwd)/config.toml:/influencer-automation-2.0/config.toml -v $(pwd)/storage:/influencer-automation-2.0/storage -p 127.0.0.1:8501:8501 influencer-automation-2.0
## For Windows:
# docker run -v ${PWD}/config.toml:/influencer-automation-2.0/config.toml -v ${PWD}/storage:/influencer-automation-2.0/storage -p 127.0.0.1:8501:8501 influencer-automation-2.0
