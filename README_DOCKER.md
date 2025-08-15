# Docker Deployment Guide

This guide explains how to run the Flight Delay Analysis application using Docker.

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Build and start the main application
docker-compose up --build

# Or run in background
docker-compose up -d --build

# Start with Streamlit web interface
docker-compose up streamlit-app --build

# For development with Jupyter notebooks
docker-compose --profile dev up --build
```

### Option 2: Docker Commands

```bash
# Build the image
docker build -t flight-delay-analysis .

# Run data analysis
docker run --rm -v $(pwd)/data:/app/data flight-delay-analysis

# Run Streamlit app
docker run -p 8501:8501 -v $(pwd)/data:/app/data flight-delay-analysis \
  streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0

# Run interactive container
docker run -it --rm -v $(pwd)/data:/app/data flight-delay-analysis bash
```

## Services Available

| Service | Port | Description |
|---------|------|-------------|
| Main App | - | Data download and analysis |
| Streamlit | 8502 | Web interface for analysis |
| Jupyter | 8889 | Notebook development (dev profile) |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| PYTHONPATH | /app | Python module search path |
| DATA_DIR | /app/data | Data storage directory |
| LOG_LEVEL | INFO | Logging level |

## Volume Mounts

- `./data:/app/data` - Persistent data storage
- `./logs:/app/logs` - Application logs
- `./src:/app/src` - Source code (development)
- `./notebooks:/app/notebooks` - Jupyter notebooks

## Commands

### Development Workflow

```bash
# Start all services for development
docker-compose --profile dev up --build

# Run analysis only
docker-compose run flight-delay-app python explore_data.py

# Access container shell
docker-compose exec flight-delay-app bash

# View logs
docker-compose logs -f flight-delay-app
```

### Production Deployment

```bash
# Production build
docker-compose -f docker-compose.yml up -d --build

# Scale services (if needed)
docker-compose up --scale streamlit-app=2

# Update and restart
docker-compose pull && docker-compose up -d
```

## Data Persistence

Data is automatically persisted in the `./data` directory:
- Raw data downloads are saved to `data/raw/`
- Processed datasets are saved to `data/processed/`
- Analysis reports are saved to the project root

## Troubleshooting

### Common Issues

1. **Port conflicts**: Change ports in docker-compose.yml if 8501/8502 are in use
2. **Memory issues**: Ensure Docker has sufficient memory (4GB+ recommended)
3. **Permission errors**: Check file permissions in mounted volumes

### Debug Commands

```bash
# Check container status
docker-compose ps

# View container logs
docker-compose logs [service-name]

# Execute commands in running container
docker-compose exec flight-delay-app python --version

# Rebuild without cache
docker-compose build --no-cache
```

### Clean Up

```bash
# Stop all services
docker-compose down

# Remove containers and networks
docker-compose down --volumes

# Remove images
docker-compose down --rmi all

# Full cleanup
docker system prune -a
```

## Image Information

- **Base Image**: python:3.12-slim
- **Size**: ~500MB (compressed)
- **Architecture**: Multi-platform (amd64, arm64)
- **Security**: Runs as non-root user
- **Health Check**: Validates Python dependencies

## Security Considerations

- Container runs as non-root user `appuser`
- No sensitive data stored in image
- Network isolated by default
- Health checks monitor container status

## Performance Tips

1. Use volume mounts for data persistence
2. Enable BuildKit for faster builds: `DOCKER_BUILDKIT=1`
3. Use `.dockerignore` to exclude unnecessary files
4. Consider multi-stage builds for production optimization