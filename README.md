# Lung Cancer Model Deployment (Docker + API)

This project now includes a production-ready FastAPI endpoint for your existing model file:
- `lung_cancer_model_final2.h5`

## What is included

- `app/main.py` FastAPI inference API
- `requirements.txt` Python dependencies
- `Dockerfile` container build file
- `docker-compose.yml` one-command local or VPS run
- `class_names.example.json` example class labels file

## API Endpoints

- `GET /` simple status message
- `GET /health` health and model metadata
- `POST /predict` image upload endpoint (`multipart/form-data`)
- `POST /predict/base64` JSON base64 image endpoint

## 1. Set class labels (important)

Your model has 4 output classes. To get meaningful class names, set one of these:

### Option A: Environment variable

Set `CLASS_NAMES` to 4 comma-separated labels in the exact training order.

Example:

```bash
CLASS_NAMES=adenocarcinoma,large_cell_carcinoma,normal,squamous_cell_carcinoma
```

### Option B: JSON file

Create `class_names.json` in project root (same folder as `Dockerfile`) with 4 labels:

```json
[
  "adenocarcinoma",
  "large_cell_carcinoma",
  "normal",
  "squamous_cell_carcinoma"
]
```

If none is provided, fallback labels are used: `class_0`, `class_1`, `class_2`, `class_3`.

## 2. Run with Docker Compose

```bash
docker compose up --build -d
```

API will be available at:

```text
http://<your-server-ip>:8000
```

Check health:

```bash
curl http://localhost:8000/health
```

## 3. Run with plain Docker

Build image:

```bash
docker build -t lung-cancer-api .
```

Run container:

```bash
docker run -d --name lung-cancer-api -p 8000:8000 \
  -e MODEL_PATH=lung_cancer_model_final2.h5 \
  -e IMAGE_SIZE=224 \
  -e CLASS_NAMES='adenocarcinoma,large_cell_carcinoma,normal,squamous_cell_carcinoma' \
  lung-cancer-api
```

## 4. Example requests

### Predict with image file

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/absolute/path/to/test-image.png"
```

### Predict with base64 JSON

```bash
curl -X POST "http://localhost:8000/predict/base64" \
  -H "Content-Type: application/json" \
  -d '{"image_base64":"<BASE64_STRING>"}'
```

## 5. VPS deployment notes

- Open firewall/security group for port `8000` (or reverse proxy through Nginx).
- For HTTPS, put Nginx/Caddy in front and proxy to `localhost:8000`.
- For production reliability, keep `restart: unless-stopped` in Compose.

## Response format

Prediction endpoints return:

- `predicted_index`
- `predicted_label`
- `confidence`
- `probabilities` (per-class probabilities)

## Important

Ensure the label order exactly matches the class index order used during training. If order is wrong, predicted label names will be wrong even when the model prediction is correct.
# modle-dl
