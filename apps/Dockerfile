FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY apps/package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy application
COPY apps/backend.js .

# Create directories
RUN mkdir -p /app/data /app/logs && chmod 777 /app/data /app/logs

# Health check
HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
  CMD wget --quiet --tries=1 --spider http://localhost:5000/health || exit 1

# Expose port
EXPOSE 5000

# Run application
CMD ["node", "backend.js"]
