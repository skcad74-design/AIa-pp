import os
import time
import uuid
import base64
import tempfile
import shutil
import threading
import queue
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from gradio_client import Client, handle_file


# =========================================================
# APP CONFIG
# =========================================================

app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=False
)

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

# Current Hugging Face Space
HF_SPACE = "tencent/Hunyuan3D-2"

# Folder for generated/downloaded models
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "generated_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Temporary uploaded images
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Queue
request_queue = queue.Queue()

# Task information
task_status = {}

# Thread safety
task_lock = threading.Lock()

# Maximum number of queued jobs
MAX_QUEUE_SIZE = 10


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def update_task(task_id, **values):
    """Safely update task information."""

    with task_lock:
        if task_id in task_status:
            task_status[task_id].update(values)


def get_task(task_id):
    """Safely get task information."""

    with task_lock:
        data = task_status.get(task_id)

        if data is None:
            return None

        return dict(data)


def save_base64_image(base64_string, task_id):
    """
    Convert base64 image to a local temporary PNG/JPG file.
    """

    try:

        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]

        image_bytes = base64.b64decode(base64_string)

        if len(image_bytes) > 15 * 1024 * 1024:
            raise ValueError("Image is too large. Maximum size is 15 MB.")

        file_path = UPLOAD_DIR / f"{task_id}.png"

        with open(file_path, "wb") as f:
            f.write(image_bytes)

        return str(file_path)

    except Exception as e:
        print("IMAGE SAVE ERROR:", repr(e))
        raise


def find_file_in_result(result):
    """
    Find a local generated file from Gradio's result.
    Supports:
      - string path
      - dict FileData
      - tuple/list
      - nested structures
    """

    if result is None:
        return None

    # String / Path
    if isinstance(result, (str, Path)):

        path = Path(result)

        if path.exists() and path.is_file():
            return str(path)

        return None

    # Dictionary / FileData
    if isinstance(result, dict):

        possible_path = result.get("path")

        if possible_path:
            path = Path(possible_path)

            if path.exists() and path.is_file():
                return str(path)

        # Sometimes file information can be nested
        for value in result.values():

            found = find_file_in_result(value)

            if found:
                return found

        return None

    # List / Tuple
    if isinstance(result, (list, tuple)):

        for item in result:

            found = find_file_in_result(item)

            if found:
                return found

        return None

    return None


def copy_generated_file(source_file, task_id):
    """
    Copy generated model into our own public model directory.
    """

    source = Path(source_file)

    if not source.exists():
        raise FileNotFoundError(
            f"Generated file does not exist: {source}"
        )

    extension = source.suffix.lower()

    if extension not in [".glb", ".obj", ".ply", ".stl", ".fbx"]:
        extension = ".glb"

    filename = f"{task_id}{extension}"

    destination = MODEL_DIR / filename

    shutil.copy2(source, destination)

    return filename


# =========================================================
# HUGGING FACE GENERATION
# =========================================================

def process_3d_conversion(image_file, task_id):

    client = None

    try:

        print("=" * 60)
        print("STARTING 3D GENERATION")
        print("Task:", task_id)
        print("Image:", image_file)
        print("=" * 60)

        update_task(
            task_id,
            status="processing",
            message="Connecting to Hunyuan3D..."
        )

        # -------------------------------------------------
        # Connect to current Hugging Face Space
        # -------------------------------------------------

        client_kwargs = {
            "download_files": str(MODEL_DIR),
            "verbose": True
        }

        if HF_TOKEN:
            client_kwargs["token"] = HF_TOKEN

        client = Client(
            HF_SPACE,
            **client_kwargs
        )

        print("Connected to Hugging Face:", HF_SPACE)

        update_task(
            task_id,
            status="processing",
            message="Sending image to Hunyuan3D..."
        )

        # -------------------------------------------------
        # Prepare image
        # -------------------------------------------------

        image_input = handle_file(image_file)

        # -------------------------------------------------
        # Current Hunyuan3D shape_generation API
        # -------------------------------------------------

        print("Calling /shape_generation ...")

        update_task(
            task_id,
            status="processing",
            message="3D model generation started..."
        )

        result = client.predict(

            None,                  # caption

            image_input,           # image

            None,                  # mv_image_front
            None,                  # mv_image_back
            None,                  # mv_image_left
            None,                  # mv_image_right

            30,                    # steps

            5.0,                   # guidance_scale

            1234,                  # seed

            256,                   # octree_resolution

            True,                  # check_box_rembg

            8000,                  # num_chunks

            True,                  # randomize_seed

            api_name="/shape_generation"
        )

        print("Hugging Face result:")
        print(repr(result))

        # -------------------------------------------------
        # Find generated file
        # -------------------------------------------------

        generated_file = find_file_in_result(result)

        if not generated_file:

            raise RuntimeError(
                "Hunyuan3D completed but no model file was returned. "
                f"Raw result: {repr(result)}"
            )

        print("Generated file:", generated_file)

        # -------------------------------------------------
        # Copy model to our own server folder
        # -------------------------------------------------

        filename = copy_generated_file(
            generated_file,
            task_id
        )

        download_url = f"/download/{filename}"

        print("Public download URL:", download_url)

        update_task(
            task_id,
            status="completed",
            message="3D model successfully generated!",
            result=download_url,
            filename=filename
        )

        print("TASK COMPLETED:", task_id)

    except Exception as e:

        error_message = str(e)

        print("=" * 60)
        print("3D GENERATION ERROR")
        print(error_message)
        print("=" * 60)

        update_task(
            task_id,
            status="failed",
            message="3D model generation failed.",
            error=error_message
        )

    finally:

        # Remove uploaded image
        try:

            if image_file:

                image_path = Path(image_file)

                if image_path.exists():
                    image_path.unlink()

        except Exception as cleanup_error:

            print(
                "Upload cleanup error:",
                cleanup_error
            )


# =========================================================
# WORKER THREAD
# =========================================================

def worker_thread():

    print("3D worker thread started.")

    while True:

        task_id, image_file = request_queue.get()

        try:

            process_3d_conversion(
                image_file,
                task_id
            )

        except Exception as e:

            print(
                "WORKER UNEXPECTED ERROR:",
                repr(e)
            )

            update_task(
                task_id,
                status="failed",
                error=str(e)
            )

        finally:

            request_queue.task_done()


# Start exactly one worker.
# Important because task_status is stored in memory.

worker = threading.Thread(
    target=worker_thread,
    daemon=True
)

worker.start()


# =========================================================
# ROUTES
# =========================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "Server is running successfully!",
        "service": "INSA3D 3D Generator",
        "engine": "Tencent Hunyuan3D-2",
        "queue": request_queue.qsize()
    })


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok"
    })


# ---------------------------------------------------------
# Generate 3D
# ---------------------------------------------------------

@app.route("/generate-3d", methods=["POST"])
def generate_3d():

    try:

        # Prevent unlimited queue
        if request_queue.qsize() >= MAX_QUEUE_SIZE:

            return jsonify({
                "error": "Server is busy. Please try again later."
            }), 429

        data = request.get_json(
            silent=True
        ) or {}

        image_data = data.get("image")

        if not image_data:

            return jsonify({
                "error": "No image provided."
            }), 400

        if not isinstance(image_data, str):

            return jsonify({
                "error": "Invalid image data."
            }), 400

        # Create unique task ID
        task_id = (
            "task_"
            + str(int(time.time() * 1000))
            + "_"
            + uuid.uuid4().hex[:8]
        )

        # Save image
        image_file = save_base64_image(
            image_data,
            task_id
        )

        queue_position = request_queue.qsize() + 1

        estimated_wait = queue_position * 180

        # Save task
        with task_lock:

            task_status[task_id] = {

                "status": "pending",

                "message": (
                    "Your image has been added "
                    "to the generation queue."
                ),

                "position": queue_position,

                "wait_time": estimated_wait,

                "created_at": time.time()
            }

        # Add to queue
        request_queue.put(
            (
                task_id,
                image_file
            )
        )

        print(
            "NEW TASK:",
            task_id,
            "Position:",
            queue_position
        )

        return jsonify({

            "success": True,

            "task_id": task_id,

            "status": "pending",

            "message": (
                "Your request has been added "
                "to the generation queue."
            ),

            "position": queue_position,

            "wait_time": estimated_wait
        })

    except Exception as e:

        print(
            "GENERATE ERROR:",
            repr(e)
        )

        return jsonify({
            "error": str(e)
        }), 500


# ---------------------------------------------------------
# Task status
# ---------------------------------------------------------

@app.route(
    "/status/<task_id>",
    methods=["GET"]
)
def check_status(task_id):

    status_info = get_task(task_id)

    if not status_info:

        return jsonify({

            "status": "failed",

            "error": (
                "Task not found. "
                "The Render server may have restarted."
            )

        }), 200

    return jsonify(status_info)


# ---------------------------------------------------------
# Download generated model
# ---------------------------------------------------------

@app.route(
    "/download/<filename>",
    methods=["GET"]
)
def download_model(filename):

    # Security
    if (
        "/" in filename
        or "\\" in filename
        or ".." in filename
    ):

        return jsonify({
            "error": "Invalid filename."
        }), 400

    file_path = MODEL_DIR / filename

    if not file_path.exists():

        return jsonify({
            "error": "Model file not found."
        }), 404

    return send_from_directory(
        MODEL_DIR,
        filename,
        as_attachment=True
    )


# =========================================================
# CLEANUP OLD FILES
# =========================================================

def cleanup_old_models():

    while True:

        try:

            now = time.time()

            for file in MODEL_DIR.iterdir():

                if not file.is_file():
                    continue

                age = now - file.stat().st_mtime

                # Delete files older than 2 hours
                if age > 7200:

                    try:
                        file.unlink()

                        print(
                            "Deleted old model:",
                            file.name
                        )

                    except Exception:
                        pass

        except Exception as e:

            print(
                "Cleanup error:",
                e
            )

        # Check every 30 minutes
        time.sleep(1800)


cleanup_thread = threading.Thread(
    target=cleanup_old_models,
    daemon=True
)

cleanup_thread.start()


# =========================================================
# LOCAL DEVELOPMENT
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
