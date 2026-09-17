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
# APP
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
# CONFIG
# =========================================================

HF_TOKEN = os.getenv(
    "HF_TOKEN",
    ""
).strip()


HF_SPACE = (
    "tencent/Hunyuan3D-2"
)


BASE_DIR = Path(
    __file__
).resolve().parent


MODEL_DIR = (
    BASE_DIR /
    "generated_models"
)


UPLOAD_DIR = (
    BASE_DIR /
    "uploads"
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# QUEUE
# =========================================================

request_queue = queue.Queue()


MAX_QUEUE_SIZE = 10


# =========================================================
# TASK DATABASE
# =========================================================

task_status = {}


task_lock = threading.Lock()


# =========================================================
# WORKER STATE
# =========================================================

worker_started = False


worker_lock = threading.Lock()


# =========================================================
# TASK UPDATE
# =========================================================

def update_task(
    task_id,
    **values
):

    with task_lock:

        if task_id in task_status:

            task_status[
                task_id
            ].update(
                values
            )


# =========================================================
# GET TASK
# =========================================================

def get_task(
    task_id
):

    with task_lock:

        data =
            task_status.get(
                task_id
            )

        if data is None:

            return None

        return dict(
            data
        )


# =========================================================
# SAVE IMAGE
# =========================================================

def save_base64_image(
    base64_string,
    task_id
):

    if not base64_string:

        raise ValueError(
            "No image data received."
        )


    try:

        if "," in base64_string:

            base64_string = \
                base64_string.split(
                    ",",
                    1
                )[1]


        image_bytes = \
            base64.b64decode(
                base64_string,
                validate=True
            )


        if len(image_bytes) > (
            15 * 1024 * 1024
        ):

            raise ValueError(
                "Image is too large. Maximum size is 15 MB."
            )


        file_path = (
            UPLOAD_DIR /
            f"{task_id}.png"
        )


        with open(
            file_path,
            "wb"
        ) as f:

            f.write(
                image_bytes
            )


        print(
            "IMAGE SAVED:",
            file_path,
            flush=True
        )


        return str(
            file_path
        )


    except Exception as e:

        print(
            "IMAGE SAVE ERROR:",
            repr(e),
            flush=True
        )

        raise


# =========================================================
# FIND GENERATED FILE
# =========================================================

def find_file_in_result(
    result
):

    if result is None:

        return None


    # String / Path
    if isinstance(
        result,
        (str, Path)
    ):

        path = Path(
            result
        )


        if (
            path.exists()
            and
            path.is_file()
        ):

            return str(
                path
            )


        return None


    # Dictionary
    if isinstance(
        result,
        dict
    ):

        possible_path = \
            result.get(
                "path"
            )


        if possible_path:

            path = Path(
                possible_path
            )


            if (
                path.exists()
                and
                path.is_file()
            ):

                return str(
                    path
                )


        for value in result.values():

            found = \
                find_file_in_result(
                    value
                )


            if found:

                return found


        return None


    # List / tuple
    if isinstance(
        result,
        (list, tuple)
    ):

        for item in result:

            found = \
                find_file_in_result(
                    item
                )


            if found:

                return found


        return None


    return None


# =========================================================
# COPY MODEL
# =========================================================

def copy_generated_file(
    source_file,
    task_id
):

    source = Path(
        source_file
    )


    if not source.exists():

        raise FileNotFoundError(
            "Generated model file does not exist: "
            + str(source)
        )


    extension = \
        source.suffix.lower()


    allowed = [
        ".glb",
        ".obj",
        ".ply",
        ".stl",
        ".fbx"
    ]


    if extension not in allowed:

        extension = ".glb"


    filename = (
        task_id +
        extension
    )


    destination = (
        MODEL_DIR /
        filename
    )


    shutil.copy2(
        source,
        destination
    )


    print(
        "MODEL COPIED:",
        destination,
        flush=True
    )


    return filename


# =========================================================
# HUNYUAN 3D
# =========================================================

def process_3d_conversion(
    image_file,
    task_id
):

    client = None


    try:

        print(
            "\n"
            + "=" * 60,
            flush=True
        )

        print(
            "STARTING 3D GENERATION",
            flush=True
        )

        print(
            "TASK:",
            task_id,
            flush=True
        )

        print(
            "IMAGE:",
            image_file,
            flush=True
        )

        print(
            "=" * 60,
            flush=True
        )


        # ---------------------------------------------
        # PROCESSING
        # ---------------------------------------------

        update_task(
            task_id,
            status="processing",
            message="Connecting to Hunyuan3D..."
        )


        # ---------------------------------------------
        # CONNECT
        # ---------------------------------------------

        print(
            "CONNECTING TO:",
            HF_SPACE,
            flush=True
        )


        client_kwargs = {}


        if HF_TOKEN:

            client_kwargs[
                "token"
            ] = HF_TOKEN


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
            message="Connected to Hunyuan3D. Uploading image..."
        )


        # ---------------------------------------------
        # IMAGE
        # ---------------------------------------------

        image_input = handle_file(
            image_file
        )


        # ---------------------------------------------
        # GENERATION
        # ---------------------------------------------

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

            None,

            image_input,

            None,

            None,

            None,

            None,

            30,

            5.0,

            1234,

            256,

            True,

            8000,

            True,

            api_name="/shape_generation"

        )


        generation_time = (
            time.time()
            -
            generation_started
        )


        print(
            "GENERATION FINISHED:",
            round(
                generation_time,
                2
            ),
            "seconds",
            flush=True
        )


        print(
            "RAW HUGGING FACE RESULT:",
            repr(result),
            flush=True
        )


        # ---------------------------------------------
        # FIND MODEL
        # ---------------------------------------------

        generated_file = \
            find_file_in_result(
                result
            )


        if not generated_file:

            raise RuntimeError(
                "Hunyuan3D completed, "
                "but no generated model file was returned.\n"
                +
                repr(result)
            )


        print(
            "GENERATED FILE:",
            generated_file,
            flush=True
        )


        # ---------------------------------------------
        # COPY
        # ---------------------------------------------

        filename = \
            copy_generated_file(
                generated_file,
                task_id
            )


        download_url = (
            "/download/" +
            filename
        )


        # ---------------------------------------------
        # COMPLETE
        # ---------------------------------------------

        update_task(
            task_id,

            status="completed",

            message=(
                "3D model successfully generated!"
            ),

            result=download_url,

            filename=filename,

            generation_time=round(
                generation_time,
                2
            )
        )


        print(
            "TASK COMPLETED:",
            task_id,
            flush=True
        )


    except Exception as e:

        error_message = str(e)


        print(
            "\n"
            + "=" * 60,
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

            message=(
                "3D model generation failed."
            ),

            error=error_message
        )


    finally:

        # ---------------------------------------------
        # DELETE UPLOAD
        # ---------------------------------------------

        try:

            if image_file:

                image_path = Path(
                    image_file
                )


                if image_path.exists():

                    image_path.unlink()


                    print(
                        "TEMP IMAGE DELETED:",
                        image_path,
                        flush=True
                    )


        except Exception as cleanup_error:

            print(
                "UPLOAD CLEANUP ERROR:",
                repr(
                    cleanup_error
                ),
                flush=True
            )


# =========================================================
# WORKER
# =========================================================

def worker_thread():

    print(
        "\n"
        + "=" * 60,
        flush=True
    )

    print(
        "3D WORKER THREAD STARTED",
        flush=True
    )

    print(
        "=" * 60,
        flush=True
    )


    while True:

        task = None


        try:

            # -----------------------------------------
            # WAIT FOR TASK
            # -----------------------------------------

            task = request_queue.get(
                block=True
            )


            print(
                "\nWORKER RECEIVED TASK:",
                task,
                flush=True
            )


            task_id, image_file = task


            # -----------------------------------------
            # UPDATE
            # -----------------------------------------

            update_task(
                task_id,

                status="processing",

                message=(
                    "Worker started processing..."
                )
            )


            print(
                "PROCESSING TASK:",
                task_id,
                flush=True
            )


            # -----------------------------------------
            # GENERATE
            # -----------------------------------------

            process_3d_conversion(
                image_file,
                task_id
            )


        except Exception as e:

            print(
                "WORKER UNEXPECTED ERROR:",
                repr(e),
                flush=True
            )


            if task:

                try:

                    task_id = task[0]


                    update_task(
                        task_id,

                        status="failed",

                        error=str(e)
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

    global worker_started


    with worker_lock:

        if worker_started:

            return


        worker_started = True


        worker = threading.Thread(

            target=worker_thread,

            daemon=True,

            name="INSA3D-3D-WORKER"

        )


        worker.start()


        print(
            "WORKER THREAD CREATED",
            flush=True
        )


# Start worker
start_worker()


# =========================================================
# HOME
# =========================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return jsonify({

        "status":
            "Server is running successfully!",

        "service":
            "INSA3D 3D Generator",

        "engine":
            "Tencent Hunyuan3D-2",

        "queue":
            request_queue.qsize(),

        "worker_alive":
            worker_started

    })


# =========================================================
# HEALTH
# =========================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "ok",

        "worker_alive":
            worker_started,

        "queue_size":
            request_queue.qsize(),

        "tasks":
            len(task_status)

    })


# =========================================================
# GENERATE
# =========================================================

@app.route(
    "/generate-3d",
    methods=["POST"]
)
def generate_3d():

    try:

        # ---------------------------------------------
        # QUEUE LIMIT
        # ---------------------------------------------

        if (
            request_queue.qsize()
            >=
            MAX_QUEUE_SIZE
        ):

            return jsonify({

                "error":
                    "Server is busy. Please try again later."

            }), 429


        # ---------------------------------------------
        # JSON
        # ---------------------------------------------

        data = request.get_json(
            silent=True
        )


        if not data:

            return jsonify({

                "error":
                    "Invalid JSON request."

            }), 400


        image_data = \
            data.get(
                "image"
            )


        if not image_data:

            return jsonify({

                "error":
                    "No image provided."

            }), 400


        if not isinstance(
            image_data,
            str
        ):

            return jsonify({

                "error":
                    "Invalid image data."

            }), 400


        # ---------------------------------------------
        # TASK ID
        # ---------------------------------------------

        task_id = (

            "task_"

            +
            str(
                int(
                    time.time()
                    *
                    1000
                )
            )

            +
            "_"

            +
            uuid.uuid4().hex[:8]

        )


        # ---------------------------------------------
        # SAVE IMAGE
        # ---------------------------------------------

        image_file = \
            save_base64_image(
                image_data,
                task_id
            )


        # ---------------------------------------------
        # QUEUE POSITION
        # ---------------------------------------------

        queue_position = (
            request_queue.qsize()
            +
            1
        )


        estimated_wait = (
            queue_position
            *
            180
        )


        # ---------------------------------------------
        # SAVE TASK
        # ---------------------------------------------

        with task_lock:

            task_status[
                task_id
            ] = {

                "status":
                    "pending",

                "message":
                    (
                        "Your image has been "
                        "added to the generation queue."
                    ),

                "position":
                    queue_position,

                "wait_time":
                    estimated_wait,

                "created_at":
                    time.time()

            }


        # ---------------------------------------------
        # QUEUE
        # ---------------------------------------------

        request_queue.put(

            (
                task_id,
                image_file
            )

        )


        print(
            "\nNEW TASK:",
            task_id,
            "POSITION:",
            queue_position,
            flush=True
        )


        return jsonify({

            "success":
                True,

            "task_id":
                task_id,

            "status":
                "pending",

            "message":
                (
                    "Your request has been "
                    "added to the generation queue."
                ),

            "position":
                queue_position,

            "wait_time":
                estimated_wait

        })


    except Exception as e:

        print(
            "GENERATE ERROR:",
            repr(e),
            flush=True
        )


        return jsonify({

            "error":
                str(e)

        }), 500


# =========================================================
# STATUS
# =========================================================

@app.route(
    "/status/<task_id>",
    methods=["GET"]
)
def check_status(
    task_id
):

    status_info = \
        get_task(
            task_id
        )


    if not status_info:

        return jsonify({

            "status":
                "failed",

            "error":
                (
                    "Task not found. "
                    "The Render server may have restarted."
                )

        })


    # ---------------------------------------------
    # UPDATE QUEUE POSITION
    # ---------------------------------------------

    if (
        status_info.get(
            "status"
        )
        ==
        "pending"
    ):

        position = 1


        with task_lock:

            pending_tasks = []


            for tid, info in task_status.items():

                if (
                    info.get(
                        "status"
                    )
                    ==
                    "pending"
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


        pending_tasks.sort()


        for index, item in enumerate(
            pending_tasks,
            start=1
        ):

            if item[1] == task_id:

                position = index

                break


        status_info[
            "position"
        ] = position


    return jsonify(
        status_info
    )


# =========================================================
# DOWNLOAD
# =========================================================

@app.route(
    "/download/<filename>",
    methods=["GET"]
)
def download_model(
    filename
):

    # Security
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


    file_path = (
        MODEL_DIR /
        filename
    )


    if not file_path.exists():

        return jsonify({

            "error":
                "Model file not found."

        }), 404


    return send_from_directory(

        MODEL_DIR,

        filename,

        as_attachment=True

    )


# =========================================================
# CLEANUP
# =========================================================

def cleanup_old_models():

    while True:

        try:

            now = time.time()


            # -----------------------------------------
            # MODELS
            # -----------------------------------------

            for file in MODEL_DIR.iterdir():

                if not file.is_file():

                    continue


                age = (
                    now -
                    file.stat().st_mtime
                )


                if age > 7200:

                    try:

                        file.unlink()


                        print(
                            "OLD MODEL DELETED:",
                            file.name,
                            flush=True
                        )

                    except Exception:

                        pass


            # -----------------------------------------
            # UPLOADS
            # -----------------------------------------

            for file in UPLOAD_DIR.iterdir():

                if not file.is_file():

                    continue


                age = (
                    now -
                    file.stat().st_mtime
                )


                if age > 3600:

                    try:

                        file.unlink()


                        print(
                            "OLD UPLOAD DELETED:",
                            file.name,
                            flush=True
                        )

                    except Exception:

                        pass


        except Exception as e:

            print(
                "CLEANUP ERROR:",
                repr(e),
                flush=True
            )


        time.sleep(
            1800
        )


# =========================================================
# CLEANUP THREAD
# =========================================================

cleanup_thread = threading.Thread(

    target=cleanup_old_models,

    daemon=True,

    name="INSA3D-CLEANUP"

)


cleanup_thread.start()


# =========================================================
# LOCAL
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
