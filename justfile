set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# Determine the correct Python commands based on OS
sys_python := if os() == "windows" { "python" } else { "python3" }
venv_python := if os() == "windows" { ".venv\\Scripts\\python.exe" } else { ".venv/bin/python" }

# List available recipes by default
default:
    @just --list

# One-shot setup for the project
setup: check-ffmpeg create-venv install-deps test
    @echo ""
    @echo "==> Setup complete. Activate with:"
    @echo "    macOS/Linux:  source .venv/bin/activate"
    @echo "    Windows:      .venv\\Scripts\\activate"
    @echo "==> Then run:"
    @echo "    python scripts/make_test_video.py"
    @echo "    python scripts/ablation_bgfg.py --quick --saliency yolo+spectral"

# Check if ffmpeg is reachable in the system path
check-ffmpeg:
    @{{sys_python}} -c "import shutil, sys; print('==> ffmpeg found') if shutil.which('ffmpeg') else sys.exit('==> ffmpeg not found! Please install it (e.g., brew install ffmpeg, sudo apt install ffmpeg, or winget install ffmpeg)')"

# Create a Python virtual environment
create-venv:
    @echo "==> Creating/verifying .venv"
    @{{sys_python}} -m venv .venv

# Install required Python dependencies
install-deps:
    @echo "==> Upgrading pip..."
    @{{venv_python}} -m pip install --upgrade pip
    @echo "==> Installing requirements..."
    @{{venv_python}} -m pip install -r requirements.txt

# Run a quick import smoke test
test:
    @echo "==> Running smoke test..."
    @{{venv_python}} -c "import cv2, torch, numpy, skimage; print('numpy', numpy.__version__, '| cv2', cv2.__version__, '| torch', torch.__version__)"
