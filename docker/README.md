# Docker Deployment Guide

This directory contains Docker configuration and deployment scripts for the Menu Translation Backend.

## Quick Start

### Development Environment

```bash
# Start development environment with hot reload
docker-compose up -d

# View logs
docker-compose logs -f menu-translation-backend

# Stop services
docker-compose down
```

### Production Environment

```bash
# Deploy to production
./docker/scripts/deploy.sh --env production

# Check status
docker-compose -f docker-compose.yml -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.yml -f docker-compose.production.yml logs -f
```

## Environment Configurations

### Development
- **Target**: `development` stage in Dockerfile
- **Features**: Hot reload, debug mode, Redis UI
- **Port**: 8000
- **Redis UI**: http://localhost:8081

### Staging
- **Target**: `runtime` stage in Dockerfile
- **Features**: Production-like setup with debugging tools
- **Port**: 8000
- **Redis UI**: http://localhost:8081

### Production
- **Target**: `runtime` stage in Dockerfile
- **Features**: Optimized for performance, Nginx reverse proxy
- **Port**: 80 (via Nginx)
- **Replicas**: 2 (load balanced)

## Services

### menu-translation-backend
Main application service running the FastAPI backend.

**Ports:**
- 8000: Application port

**Volumes:**
- `app_logs`: Application logs
- `app_temp`: Temporary file storage
- `app_models`: AI model storage

**Health Check:**
- Endpoint: `/health`
- Interval: 30s
- Timeout: 10s

### redis
Redis cache service for storing processed results and session data.

**Ports:**
- 6379: Redis port

**Volumes:**
- `redis_data`: Persistent Redis data

**Configuration:**
- Memory limit: 256MB (dev), 512MB (prod)
- Persistence: AOF + RDB
- Eviction policy: allkeys-lru

### redis-commander (Development/Staging only)
Web UI for Redis management and debugging.

**Ports:**
- 8081: Web interface

**Access:**
- URL: http://localhost:8081
- Available in: development, staging profiles

### nginx (Production only)
Reverse proxy and load balancer for production deployments.

**Ports:**
- 80: HTTP
- 443: HTTPS (when SSL configured)

**Features:**
- Rate limiting
- Gzip compression
- Load balancing
- Security headers

## Build Scripts

### Build Image

```bash
# Build development image
./docker/scripts/build.sh --env development

# Build production image with custom tag
./docker/scripts/build.sh --env production --tag v1.0.0

# Build and push to registry
./docker/scripts/build.sh --env production --tag v1.0.0 --push --registry your-registry.com
```

### Deploy Services

```bash
# Deploy development environment
./docker/scripts/deploy.sh --env development

# Deploy production environment
./docker/scripts/deploy.sh --env production

# Stop services
./docker/scripts/deploy.sh --env production --action down

# View logs
./docker/scripts/deploy.sh --env production --action logs

# Check service status
./docker/scripts/deploy.sh --env production --action ps
```

## Environment Variables

### Application Configuration
- `ENVIRONMENT`: Target environment (development, staging, production)
- `DEBUG`: Enable debug mode
- `LOG_LEVEL`: Logging level
- `HOST_PORT`: Host port mapping (default: 8000)

### Redis Configuration
- `REDIS_HOST_PORT`: Redis host port mapping (default: 6379)
- `REDIS_PASSWORD`: Redis password (optional)
- `REDIS_DB`: Redis database number

### Nginx Configuration (Production)
- `NGINX_HTTP_PORT`: HTTP port (default: 80)
- `NGINX_HTTPS_PORT`: HTTPS port (default: 443)

### UI Configuration (Development)
- `REDIS_UI_PORT`: Redis Commander port (default: 8081)

## Volume Management

### Persistent Volumes
- `redis_data`: Redis database files
- `app_logs`: Application log files
- `app_models`: AI model files
- `nginx_logs`: Nginx access and error logs

### Temporary Volumes
- `app_temp`: Temporary processing files (auto-cleanup)

### Backup Volumes

```bash
# Backup Redis data
docker run --rm -v menu-translation_redis_data:/data -v $(pwd):/backup alpine tar czf /backup/redis-backup.tar.gz -C /data .

# Restore Redis data
docker run --rm -v menu-translation_redis_data:/data -v $(pwd):/backup alpine tar xzf /backup/redis-backup.tar.gz -C /data
```

## Monitoring and Debugging

### Health Checks

```bash
# Check application health
curl http://localhost:8000/health

# Check Redis health
docker-compose exec redis redis-cli ping
```

### Logs

```bash
# Application logs
docker-compose logs -f menu-translation-backend

# Redis logs
docker-compose logs -f redis

# All services logs
docker-compose logs -f
```

### Resource Usage

```bash
# Container stats
docker stats

# Specific service stats
docker stats menu-translation-backend menu-translation-redis
```

## Troubleshooting

### Common Issues

1. **Port conflicts**
   ```bash
   # Change ports in .env file or use environment variables
   HOST_PORT=8001 docker-compose up -d
   ```

2. **Permission issues**
   ```bash
   # Fix volume permissions
   sudo chown -R 1000:1000 ./logs ./temp
   ```

3. **Memory issues**
   ```bash
   # Check memory usage
   docker stats
   
   # Increase memory limits in compose files
   ```

4. **Redis connection issues**
   ```bash
   # Check Redis connectivity
   docker-compose exec menu-translation-backend ping redis
   ```

### Debug Mode

```bash
# Run with debug output
DEBUG=true docker-compose up

# Access container shell
docker-compose exec menu-translation-backend bash

# Check configuration
docker-compose exec menu-translation-backend python -c "from config import get_settings; print(get_settings().dict())"
```

## Security Considerations

### Production Security
- Use strong Redis passwords
- Configure SSL/TLS for Nginx
- Set up proper firewall rules
- Use secrets management for API keys
- Regular security updates

### Network Security
- Custom bridge network isolation
- Internal service communication
- Rate limiting via Nginx
- CORS configuration

### Data Security
- Encrypted volumes (if required)
- Secure backup procedures
- Access logging
- Regular security audits

## Performance Tuning

### Application Performance
- Adjust worker count based on CPU cores
- Configure memory limits appropriately
- Use Redis for caching
- Enable gzip compression

### Redis Performance
- Tune memory settings
- Configure appropriate eviction policy
- Monitor slow queries
- Use connection pooling

### Nginx Performance
- Adjust worker processes
- Configure buffer sizes
- Enable caching for static content
- Use upstream load balancing