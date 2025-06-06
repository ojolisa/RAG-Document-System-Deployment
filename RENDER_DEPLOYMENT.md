# Render Deployment Guide

This guide will help you deploy your RAG Document System on [Render](https://render.com/).

## Prerequisites

1. **GitHub Account**: Your code needs to be in a GitHub repository
2. **Render Account**: Create a free account at [render.com](https://render.com)
3. **Gemini API Key**: Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## Deployment Steps

### 1. Prepare Your Repository

Make sure your repository includes these files (already created):
- `render.yaml` - Render service configuration
- `start.sh` - Startup script
- `requirements.txt` - Python dependencies (updated with gunicorn)
- Updated `api/api.py` - With environment port support

### 2. Push to GitHub

```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### 3. Deploy on Render

1. **Go to Render Dashboard**: Visit [dashboard.render.com](https://dashboard.render.com)

2. **Create a New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select your RAG Document System repository

3. **Configure the Service**:
   - **Name**: `rag-document-system` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `chmod +x start.sh && ./start.sh`

4. **Set Environment Variables**:
   - Click "Environment" tab
   - Add: `GEMINI_API_KEY` = `your_actual_api_key_here`
   - Add: `PYTHON_VERSION` = `3.11.4`

5. **Deploy**: Click "Create Web Service"

### 4. Configuration Options

#### Using render.yaml (Recommended)
Your repository already includes a `render.yaml` file that will automatically configure:
- Service type and environment
- Build and start commands
- Environment variables
- Persistent disk for PDF storage

#### Manual Configuration
If you prefer manual setup, use these settings:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `chmod +x start.sh && ./start.sh`
- **Python Version**: `3.11.4`

### 5. Domain and SSL

Render automatically provides:
- A free `.onrender.com` subdomain
- Free SSL certificate
- Your app will be available at: `https://your-service-name.onrender.com`

## Important Notes

### Free Tier Limitations
- **Sleep Mode**: Free services sleep after 15 minutes of inactivity
- **Cold Starts**: First request after sleep takes ~30 seconds
- **Build Time**: Limited to 500 build minutes per month
- **Bandwidth**: 100GB/month

### Database Persistence
- SQLite database will persist across deployments
- PDF files are stored on a persistent disk (1GB free)

### Environment Variables
Make sure to set your `GEMINI_API_KEY`:
1. Go to your service dashboard
2. Click "Environment"
3. Add `GEMINI_API_KEY` with your actual API key

## Testing Your Deployment

1. **Health Check**: Visit `https://your-app.onrender.com/health`
2. **API Documentation**: Visit `https://your-app.onrender.com/docs`
3. **Upload Test**: Try uploading a PDF document
4. **Query Test**: Ask questions about your uploaded documents

## Troubleshooting

### Common Issues

1. **Service Won't Start**:
   - Check logs in Render dashboard
   - Verify `GEMINI_API_KEY` is set correctly
   - Ensure `start.sh` has executable permissions

2. **Import Errors**:
   - Check all dependencies are in `requirements.txt`
   - Verify file paths in your application

3. **Database Issues**:
   - SQLite file permissions
   - Check disk mount path configuration

### Monitoring

- **Logs**: Available in Render dashboard under "Logs" tab
- **Metrics**: CPU, memory, and request metrics in dashboard
- **Health Check**: Built-in endpoint at `/health`

## Scaling Options

### Upgrade Plans
For production use, consider upgrading to:
- **Starter Plan**: $7/month, no sleep, faster builds
- **Standard Plan**: $25/month, more resources, better performance

### Performance Optimization
- Use PostgreSQL instead of SQLite for better performance
- Implement caching for frequently accessed data
- Consider using Redis for session storage

## Custom Domain

To use your own domain:
1. Go to service settings
2. Add custom domain
3. Update DNS records as instructed
4. SSL certificate auto-provisioned

## Support

- **Render Docs**: [render.com/docs](https://render.com/docs)
- **Community**: [community.render.com](https://community.render.com)
- **Status**: [status.render.com](https://status.render.com)

## Security Considerations

1. **Environment Variables**: Never commit API keys to your repository
2. **HTTPS**: All traffic is encrypted with free SSL
3. **CORS**: Currently set to allow all origins - restrict for production
4. **Input Validation**: PDF upload validation is implemented

---

Your RAG Document System should now be successfully deployed on Render! 🚀
