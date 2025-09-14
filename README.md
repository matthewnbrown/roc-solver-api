# Captcha Solve API

A FastAPI-based web service that provides captcha solving capabilities using a PyTorch neural network model.

## Features

- **Captcha Solving**: Upload a captcha image and get a predicted answer with confidence score
- **Feedback System**: Report whether predictions were correct to improve model performance tracking
- **Statistics**: View API usage statistics and accuracy metrics
- **Thread-Safe Database**: SQLite with WAL mode and connection pooling for concurrent access
- **RESTful API**: Clean, documented API endpoints
- **Concurrency Support**: Handles multiple simultaneous requests safely

## Project Structure

```
captcha_solve_api/
├── app/                      # Main application package
│   ├── __init__.py
│   ├── main.py              # FastAPI application and startup
│   ├── api/                 # API endpoints
│   │   ├── __init__.py
│   │   └── endpoints.py     # API route definitions
│   ├── core/                # Core application components
│   │   ├── __init__.py
│   │   ├── config.py        # Configuration management
│   │   └── database.py      # Thread-safe database manager
│   └── models/              # ML models and prediction logic
│       ├── __init__.py
│       ├── model.py         # PyTorch model definitions
│       └── predict.py       # Prediction and preprocessing
├── examples/                # Example client code
│   └── example_client.py    # Python client library
├── scripts/                 # Utility scripts
│   └── start_server.py     # Server startup script
├── tests/                   # Test files
│   ├── test_api.py         # Basic API tests
│   └── test_concurrency.py # Concurrency testing
├── model/                   # Trained model files
│   └── 2025_09_12best_model.pth
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── Dockerfile              # Docker configuration
└── docker-compose.yml     # Docker Compose setup
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your trained model is in the correct location:
```
model/2025_09_12best_model.pth
```

## Usage

### Starting the Server

**Option 1: Production startup (no auto-reload):**
```bash
python run.py
```

**Option 2: Development startup (with auto-reload):**
```bash
python run_dev.py
```

**Option 3: Using the startup script:**
```bash
python scripts/start_server.py
```

**Option 4: Direct uvicorn command:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### API Endpoints

#### 1. Solve Captcha
**POST** `/api/v1/solve`

Upload a captcha image and get a prediction.

**Parameters:**
- `captcha_hash` (form data): Unique identifier for the captcha
- `image` (file): The captcha image file

**Response:**
```json
{
  "request_id": "uuid-string",
  "captcha_hash": "your-hash",
  "predicted_answer": "5",
  "confidence": 0.9234,
  "timestamp": "2024-01-01T12:00:00"
}
```

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/api/v1/solve" \
  -F "captcha_hash=abc123" \
  -F "image=@captcha.png"
```

#### 2. Submit Feedback
**POST** `/api/v1/feedback`

Report whether a prediction was correct.

**Parameters:**
- `request_id` (form data): The request ID from the solve endpoint
- `is_correct` (form data): Boolean indicating if prediction was correct
- `actual_answer` (form data, optional): The actual answer

**Response:**
```json
{
  "message": "Feedback submitted successfully",
  "request_id": "uuid-string",
  "is_correct": true,
  "actual_answer": "5",
  "timestamp": "2024-01-01T12:00:00"
}
```

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/api/v1/feedback" \
  -F "request_id=uuid-string" \
  -F "is_correct=true" \
  -F "actual_answer=5"
```

#### 3. Get Statistics
**GET** `/api/v1/stats`

Get API usage statistics and accuracy metrics.

**Response:**
```json
{
  "total_requests": 150,
  "feedback_received": 100,
  "correct_predictions": 85,
  "accuracy_percentage": 85.0,
  "average_confidence": 0.8765,
  "recent_requests_24h": 25,
  "timestamp": "2024-01-01T12:00:00"
}
```

#### 4. Health Check
**GET** `/health`

Check if the API is running and the model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "predictor_loaded": true,
  "timestamp": "2024-01-01T12:00:00"
}
```

## Database Schema

The API uses SQLite to store data in two tables:

### captcha_requests
- `id`: Unique request identifier
- `captcha_hash`: User-provided hash identifier
- `image_data`: Binary image data
- `predicted_answer`: Model's prediction
- `confidence`: Confidence score (0-1)
- `created_at`: Timestamp of request
- `feedback_received`: Boolean indicating if feedback was provided
- `is_correct`: Boolean indicating if prediction was correct

### feedback
- `id`: Auto-incrementing primary key
- `request_id`: Foreign key to captcha_requests
- `is_correct`: Boolean indicating correctness
- `actual_answer`: The actual answer (optional)
- `feedback_at`: Timestamp of feedback

## Model Requirements

The API expects a PyTorch model file with the following structure:
- Saved with `torch.save()` containing:
  - `model_state_dict`: The model's state dictionary
  - `class_names`: List of class names (e.g., ['0', '1', '2', ..., '9'])
  - `model_type`: Type of model used ('lightweight', 'basic', or 'improved')

## Thread Safety & Concurrency

The API is designed to handle multiple concurrent requests safely:

- **WAL Mode**: SQLite database uses Write-Ahead Logging for better concurrency
- **Connection Pooling**: Thread-safe database connection management
- **Retry Logic**: Automatic retry on database lock conflicts
- **Transaction Safety**: Proper transaction handling with rollback on errors
- **Connection Timeout**: Configurable timeout to prevent hanging connections

### Testing Concurrency

Run the concurrency test suite to verify thread safety:
```bash
python tests/test_concurrency.py
```

Or run the basic API tests:
```bash
python tests/test_api.py
```

This will simulate multiple simultaneous requests to ensure the database handling is robust.

## Error Handling

The API includes comprehensive error handling:
- Invalid image formats return 400 Bad Request
- Missing model file returns 503 Service Unavailable
- Database errors return 500 Internal Server Error
- Invalid request IDs return 404 Not Found
- Database connection issues are automatically retried

## Development

To run in development mode with auto-reload:
```bash
python run_dev.py
```

The server will automatically reload when code changes are detected.

## Production Deployment

For production deployment, consider:
- Using a production ASGI server like Gunicorn with Uvicorn workers
- Setting up proper logging
- Implementing rate limiting
- Adding authentication if needed
- Using a more robust database like PostgreSQL
- Setting up monitoring and health checks
