FROM jenkins/jenkins:lts-jdk17

USER root

# Install curl and wget
RUN apt-get update && apt-get install -y curl wget

# Download and install kubectl
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    chmod +x kubectl && \
    mv kubectl /usr/local/bin/

USER jenkins

# Install Jenkins plugins