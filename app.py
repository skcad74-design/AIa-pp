import os
import time
import uuid
import base64
import shutil
import threading
import queue
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from gradio_client import Client, handle_file


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/*": {
            "origins": "*"
        }
    },
    supports_credentials=False
)


# =========================================================
# CONFIGURATION
# =========================================================

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

HF_SPACE = "tencent/Hunyuan3D-2"

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "generated_models"
UPLOAD_DIR = BASE_DIR / "uploads"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# QUEUE
# =========================================================

request_queue = queue.Queue()

MAX_QUEUE_SIZE = 10


# =========================================================
# TASK STORAGE
# =========================================================

task_status = {}

task_lock = threading.Lock()


# =========================================================
# WORKER STATE
# =========================================================

worker_thread_instance = None

worker_lock = threading.Lock()


# =========================================================
# TASK UPDATE
# =========================================================

def update_task(task_id, **values):
    """Safely update task information."""

    with task_lock:
        if task_id in task_status:
            task_status[task_id].update(values)


# =========================================================
# GET TASK
# =========================================================

def get_task(task_id):
    """Safely get task information."""

    with task_lock:
        data = task_status.get(task_id)

        if data is None:
            return None

        return dict(data)


# =========================================================
# SAVE BASE64 IMAGE
# =========================================================

def save_base64_image(base64_string, task_id):
    """Save base64 image to temporary file."""

    if not base64_string:
        raise ValueError("No image data received.")

    try:

        # Remove data:image/...;base64, prefix
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]

        image_bytes = base64.b64decode(
            base64_string,
            validate=True
        )

        # Maximum 15 MB
        if len(image_bytes) > 15 * 1024 * 1024:
            raise ValueError(
                "Image is too large. Maximum size is 15 MB."
            )

        file_path = UPLOAD_DIR / f"{task_id}.png"

        with open(file_path, "wb") as file:
            file.write(image_bytes)

        print(
            f"IMAGE SAVED: {file_path}",
            flush=True
        )

        return str(file_path)

    except Exception as error:

        print(
            f"IMAGE SAVE ERROR: {error}",
            flush=True
        )

        raise


# =========================================================
# FIND FILE IN HUGGING FACE RESULT
# =========================================================

def find_file_in_result(result):
    """
    Search recursively for a generated file.

    Supports:
    - string
    - pathlib.Path
    - dict
    - list
    - tuple
    """

    if result is None:
        return None

    # -----------------------------------------
    # STRING / PATH
    # -----------------------------------------

    if isinstance(result, (str, Path)):

        path = Path(result)

        if path.exists() and path.is_file():
            return str(path)

        return None

    # -----------------------------------------
    # DICTIONARY
    # -----------------------------------------

    if isinstance(result, dict):

        possible_path = result.get("path")

        if possible_path:

            path = Path(possible_path)

            if path.exists() and path.is_file():
                return str(path)

        for value in result.values():

            found = find_file_in_result(value)

            if found:
                return found

        return None

    # -----------------------------------------
    # LIST / TUPLE
    # -----------------------------------------

    if isinstance(result, (list, tuple)):

        for item in result:

            found = find_file_in_result(item)

            if found:
                return found

        return None

    return None


# =========================================================
# COPY GENERATED MODEL
# =========================================================

def copy_generated_file(source_file, task_id):
    """Copy generated model to public model directory."""

    source = Path(source_file)

    if not source.exists():
        raise FileNotFoundError(
            f"Generated model does not exist: {source}"
        )

    extension = source.suffix.lower()

    allowed_extensions = [
        ".glb",
        ".obj",
        ".ply",
        ".stl",
        ".fbx"
    ]

    if extension not in allowed_extensions:
        extension = ".glb"

    filename = f"{task_id}{extension}"

    destination = MODEL_DIR / filename

    shutil.copy2(
        source,
        destination
    )

    print(
        f"MODEL COPIED: {destination}",
        flush=True
    )

    return filename


# =========================================================
# HUNYUAN 3D PROCESS
# =========================================================

def process_3d_conversion(image_file, task_id):

    client = None

    try:

        print(
            "\n" + "=" * 60,
            flush=True
        )

        print(
            "STARTING 3D GENERATION",
            flush=True
        )

        print(
            f"TASK: {task_id}",
            flush=True
        )

        print(
            f"IMAGE: {image_file}",
            flush=True
        )

        print(
            "=" * 60,
            flush=True
        )

        # -----------------------------------------
        # PROCESSING
        # -----------------------------------------

        update_task(
            task_id,
            status="processing",
            message="Connecting to Hunyuan3D..."
        )

        # -----------------------------------------
        # HUGGING FACE CLIENT
        # -----------------------------------------

        print(
            f"CONNECTING TO: {HF_SPACE}",
            flush=True
        )

        client_kwargs = {}

        if HF_TOKEN:
            client_kwargs["token"] = HF_TOKEN

        client = Client(
            HF_SPACE,
            **client_kwargs
        )

        print(
            "HUGGING FACE CONNECTED",
            flush=True
        )

        update_task(
            task_id,
            status="processing",
            message="Connected to Hunyuan3D. Preparing image..."
        )

        # -----------------------------------------
        # PREPARE IMAGE
        # -----------------------------------------

        image_input = handle_file(
            image_file
        )

        print(
            "IMAGE PREPARED",
            flush=True
        )

        # -----------------------------------------
        # GENERATION
        # -----------------------------------------

        update_task(
            task_id,
            status="processing",
            message="3D model generation started..."
        )

        print(
            "CALLING /shape_generation",
            flush=True
        )

        generation_started = time.time()

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

        generation_time = (
            time.time() - generation_started
        )

        print(
            f"GENERATION FINISHED: "
            f"{round(generation_time, 2)} seconds",
            flush=True
        )

        print(
            "RAW HUGGING FACE RESULT:",
            repr(result),
            flush=True
        )

        # -----------------------------------------
        # FIND GENERATED FILE
        # -----------------------------------------

        generated_file = find_file_in_result(
            result
        )

        if not generated_file:

            raise RuntimeError(
                "Hunyuan3D completed, but no "
                "generated model file was returned. "
                f"Raw result: {repr(result)}"
            )

        print(
            f"GENERATED FILE: {generated_file}",
            flush=True
        )

        # -----------------------------------------
        # COPY MODEL
        # -----------------------------------------

        filename = copy_generated_file(
            generated_file,
            task_id
        )

        download_url = f"/download/{filename}"

        # -----------------------------------------
        # COMPLETE
        # -----------------------------------------

        update_task(
            task_id,

            status="completed",

            message="3D model successfully generated!",

            result=download_url,

            filename=filename,

            generation_time=round(
                generation_time,
                2
            )
        )

        print(
            f"TASK COMPLETED: {task_id}",
            flush=True
        )

    except Exception as error:

        error_message = str(error)

        print(
            "\n" + "=" * 60,
            flush=True
        )

        print(
            "3D GENERATION ERROR",
            flush=True
        )

        print(
            error_message,
            flush=True
        )

        print(
            "=" * 60,
            flush=True
        )

        update_task(
            task_id,

            status="failed",

            message="3D model generation failed.",

            error=error_message
        )

    finally:

        # -----------------------------------------
        # DELETE TEMPORARY IMAGE
        # -----------------------------------------

        try:

            if image_file:

                image_path = Path(
                    image_file
                )

                if image_path.exists():

                    image_path.unlink()

                    print(
                        f"TEMP IMAGE DELETED: {image_path}",
                        flush=True
                    )

        except Exception as cleanup_error:

            print(
                f"UPLOAD CLEANUP ERROR: "
                f"{cleanup_error}",
                flush=True
            )


# =========================================================
# WORKER
# =========================================================

def worker_function():

    print(
        "\n" + "=" * 60,
        flush=True
    )

    print(
        "INSA3D 3D WORKER STARTED",
        flush=True
    )

    print(
        "=" * 60,
        flush=True
    )

    while True:

        task = None

        try:

            # -------------------------------------
            # WAIT FOR TASK
            # -------------------------------------

            task = request_queue.get(
                block=True
            )

            print(
                f"WORKER RECEIVED TASK: {task}",
                flush=True
            )

            task_id, image_file = task

            # -------------------------------------
            # PROCESSING
            # -------------------------------------

            update_task(
                task_id,

                status="processing",

                message="Worker started processing..."
            )

            print(
                f"PROCESSING TASK: {task_id}",
                flush=True
            )

            # -------------------------------------
            # GENERATE
            # -------------------------------------

            process_3d_conversion(
                image_file,
                task_id
            )

        except Exception as error:

            print(
                f"WORKER UNEXPECTED ERROR: {error}",
                flush=True
            )

            if task is not None:

                try:

                    task_id = task[0]

                    update_task(
                        task_id,

                        status="failed",

                        message="Worker failed.",

                        error=str(error)
                    )

                except Exception:
                    pass

        finally:

            if task is not None:

                try:
                    request_queue.task_done()
                except Exception:
                    pass

            print(
                "WORKER LOOP READY",
                flush=True
            )


# =========================================================
# START WORKER
# =========================================================

def start_worker():

    global worker_thread_instance

    with worker_lock:

        # Don't start twice
        if (
            worker_thread_instance is not None
            and
            worker_thread_instance.is_alive()
        ):
            return

        worker_thread_instance = threading.Thread(
            target=worker_function,
            daemon=True,
            name="INSA3D-3D-WORKER"
        )

        worker_thread_instance.start()

        print(
            "WORKER THREAD CREATED",
            flush=True
        )


# =========================================================
# START WORKER NOW
# =========================================================

start_worker()


# =========================================================
# HOME
# =========================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    worker_alive = (
        worker_thread_instance is not None
        and
        worker_thread_instance.is_alive()
    )

    return jsonify({

        "status": "Server is running successfully!",

        "service": "INSA3D 3D Generator",

        "engine": "Tencent Hunyuan3D-2",

        "queue": request_queue.qsize(),

        "worker_alive": worker_alive

    })


# =========================================================
# HEALTH
# =========================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    worker_alive = (
        worker_thread_instance is not None
        and
        worker_thread_instance.is_alive()
    )

    return jsonify({

        "status": "ok",

        "worker_alive": worker_alive,

        "queue_size": request_queue.qsize(),

        "tasks": len(task_status)

    })


# =========================================================
# GENERATE 3D
# =========================================================

@app.route(
    "/generate-3d",
    methods=["POST"]
)
def generate_3d():

    try:

        # -----------------------------------------
        # QUEUE LIMIT
        # -----------------------------------------

        if request_queue.qsize() >= MAX_QUEUE_SIZE:

            return jsonify({

                "success": False,

                "error":
                    "Server is busy. Please try again later."

            }), 429

        # -----------------------------------------
        # REQUEST JSON
        # -----------------------------------------

        data = request.get_json(
            silent=True
        )

        if not data:

            return jsonify({

                "success": False,

                "error":
                    "Invalid JSON request."

            }), 400

        # -----------------------------------------
        # IMAGE
        # -----------------------------------------

        image_data = data.get(
            "image"
        )

        if not image_data:

            return jsonify({

                "success": False,

                "error":
                    "No image provided."

            }), 400

        if not isinstance(
            image_data,
            str
        ):

            return jsonify({

                "success": False,

                "error":
                    "Invalid image data."

            }), 400

        # -----------------------------------------
        # TASK ID
        # -----------------------------------------

        task_id = (
            "task_"
            + str(
                int(
                    time.time() * 1000
                )
            )
            + "_"
            + uuid.uuid4().hex[:8]
        )

        # -----------------------------------------
        # SAVE IMAGE
        # -----------------------------------------

        image_file = save_base64_image(
            image_data,
            task_id
        )

        # -----------------------------------------
        # QUEUE POSITION
        # -----------------------------------------

        queue_position = (
            request_queue.qsize() + 1
        )

        estimated_wait = (
            queue_position * 180
        )

        # -----------------------------------------
        # SAVE TASK
        # -----------------------------------------

        with task_lock:

            task_status[task_id] = {

                "status": "pending",

                "message":
                    "Your image has been added "
                    "to the generation queue.",

                "position":
                    queue_position,

                "wait_time":
                    estimated_wait,

                "created_at":
                    time.time()

            }

        # -----------------------------------------
        # ADD TO QUEUE
        # -----------------------------------------

        request_queue.put(
            (
                task_id,
                image_file
            )
        )

        print(
            "\n" + "=" * 60,
            flush=True
        )

        print(
            f"NEW TASK: {task_id}",
            flush=True
        )

        print(
            f"QUEUE POSITION: {queue_position}",
            flush=True
        )

        print(
            "=" * 60,
            flush=True
        )

        # -----------------------------------------
        # RESPONSE
        # -----------------------------------------

        return jsonify({

            "success": True,

            "task_id": task_id,

            "status": "pending",

            "message":
                "Your request has been "
                "added to the generation queue.",

            "position":
                queue_position,

            "wait_time":
                estimated_wait

        })

    except Exception as error:

        print(
            f"GENERATE ERROR: {error}",
            flush=True
        )

        return jsonify({

            "success": False,

            "error":
                str(error)

        }), 500


# =========================================================
# TASK STATUS
# =========================================================

@app.route(
    "/status/<task_id>",
    methods=["GET"]
)
def check_status(task_id):

    status_info = get_task(
        task_id
    )

    # -----------------------------------------
    # TASK NOT FOUND
    # -----------------------------------------

    if status_info is None:

        return jsonify({

            "status": "failed",

            "error":
                "Task not found. "
                "The Render server may have restarted."

        }), 200

    # -----------------------------------------
    # PENDING QUEUE POSITION
    # -----------------------------------------

    if (
        status_info.get("status")
        == "pending"
    ):

        pending_tasks = []

        with task_lock:

            for tid, info in task_status.items():

                if (
                    info.get("status")
                    == "pending"
                ):

                    pending_tasks.append(
                        (
                            info.get(
                                "created_at",
                                0
                            ),
                            tid
                        )
                    )

        pending_tasks.sort(
            key=lambda item: item[0]
        )

        position = 1

        for index, item in enumerate(
            pending_tasks,
            start=1
        ):

            if item[1] == task_id:

                position = index

                break

        status_info["position"] = position

    # -----------------------------------------
    # RESPONSE
    # -----------------------------------------

    return jsonify(
        status_info
    )


# =========================================================
# DOWNLOAD MODEL
# =========================================================

@app.route(
    "/download/<filename>",
    methods=["GET"]
)
def download_model(filename):

    # -----------------------------------------
    # SECURITY
    # -----------------------------------------

    if (
        "/" in filename
        or
        "\\" in filename
        or
        ".." in filename
    ):

        return jsonify({

            "error":
                "Invalid filename."

        }), 400

    # -----------------------------------------
    # FILE
    # -----------------------------------------

    file_path = (
        MODEL_DIR / filename
    )

    if not file_path.exists():

        return jsonify({

            "error":
                "Model file not found."

        }), 404

    # -----------------------------------------
    # DOWNLOAD
    # -----------------------------------------

    return send_from_directory(

        MODEL_DIR,

        filename,

        as_attachment=True

    )


# =========================================================
# CLEANUP OLD FILES
# =========================================================

def cleanup_old_files():

    while True:

        try:

            now = time.time()

            # -------------------------------------
            # GENERATED MODELS
            # -------------------------------------

            for file in MODEL_DIR.iterdir():

                if not file.is_file():
                    continue

                age = (
                    now -
                    file.stat().st_mtime
                )

                # Delete after 2 hours
                if age > 7200:

                    try:

                        file.unlink()

                        print(
                            f"OLD MODEL DELETED: "
                            f"{file.name}",
                            flush=True
                        )

                    except Exception:
                        pass

            # -------------------------------------
            # UPLOADS
            # -------------------------------------

            for file in UPLOAD_DIR.iterdir():

                if not file.is_file():
                    continue

                age = (
                    now -
                    file.stat().st_mtime
                )

                # Delete after 1 hour
                if age > 3600:

                    try:

                        file.unlink()

                        print(
                            f"OLD UPLOAD DELETED: "
                            f"{file.name}",
                            flush=True
                        )

                    except Exception:
                        pass

        except Exception as error:

            print(
                f"CLEANUP ERROR: {error}",
                flush=True
            )

        # Run every 30 minutes
        time.sleep(1800)


# =========================================================
# CLEANUP THREAD
# =========================================================

cleanup_thread = threading.Thread(

    target=cleanup_old_files,

    daemon=True,

    name="INSA3D-CLEANUP"

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
