FROM node:18-alpine

WORKDIR /app

# Install dependencies required for development
RUN apk add --no-cache libc6-compat

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci

# Copy the rest of the application
COPY . .

# Expose port 3000
EXPOSE 3000

# Set environment variables
ENV PORT 3000
ENV HOSTNAME "0.0.0.0"
ENV NODE_ENV development

# Start Next.js in development mode
CMD ["npx", "next", "dev"]