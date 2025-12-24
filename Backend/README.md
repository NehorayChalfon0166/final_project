# FastAPI Backend Template

## Project Structure

```
Backend/
├── main.py              # Main application entry point
├── requirements.txt     # Python dependencies
├── routes/
│   ├── __init__.py      # Routes package
│   ├── health.py        # Health check routes
│   └── items.py         # Items CRUD routes
```

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### 3. Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health Check
- `GET /api/v1/health` - Check if API is healthy
- `GET /api/v1/ping` - Ping the API

### Items (CRUD Operations)
- `GET /api/v1/items` - List all items
- `GET /api/v1/items/{item_id}` - Get a specific item
- `POST /api/v1/items` - Create a new item
- `PUT /api/v1/items/{item_id}` - Update an item
- `DELETE /api/v1/items/{item_id}` - Delete an item

## Features

- ✅ FastAPI framework
- ✅ Async/await support
- ✅ CORS middleware enabled
- ✅ Pydantic models for data validation
- ✅ Organized routing structure
- ✅ Lifespan event handlers
- ✅ Automatic interactive API documentation

## Next Steps

1. Customize models in `routes/items.py` to match your needs
2. Add more routes by creating new files in the `routes/` directory
3. Add database integration (SQLAlchemy, MongoDB, etc.)
4. Implement authentication and authorization
5. Add environment configuration (.env file)
