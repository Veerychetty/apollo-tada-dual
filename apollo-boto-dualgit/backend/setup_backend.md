# Backend Setup Instructions

## Virtual Environment Setup

The virtual environment has been created in the `backend/venv` directory.

### Activating the Virtual Environment

**On Windows (PowerShell):**
```powershell
cd backend
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
cd backend
venv\Scripts\activate.bat
```

**On Linux/Mac:**
```bash
cd backend
source venv/bin/activate
```

### Installing Dependencies

**Important:** The requirements.txt has been updated with Python 3.13-compatible versions. If you encounter any issues, make sure pip and setuptools are up to date:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Then install all required packages:

```bash
pip install -r requirements.txt
```

### Running the Backend

After activating the venv and installing dependencies, run:

```bash
python botodual.py
```

The server will start on `http://0.0.0.0:5000`

## Environment Variables

Make sure you have a `file.env` file in the backend directory with your AWS credentials:

```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=ap-south-1
FRONTEND_ORIGINS=http://localhost:3000,http://localhost:5173
```

## Dependencies

All dependencies are listed in `requirements.txt`:
- Flask (web framework)
- Flask-CORS (CORS support)
- boto3 (AWS SDK)
- opencv-python (image processing)
- numpy (numerical operations)
- Pillow (image handling)
- matplotlib (plotting)
- numba (JIT compilation)
- nltk (natural language processing)
- python-dotenv (environment variables)

