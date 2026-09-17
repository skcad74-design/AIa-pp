import os
import queue
import threading
import time
from flask import Flask, jsonify, request
from flask_cors import CORS
from gradio_client import Client, handle_file

app = Flask(__name__)
CORS(app)

# -------------------------------------------------------------
# ১. Gradio API Token Configuration
# -------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN", "")

# -------------------------------------------------------------
# ২. Request Queue এবং Task Status Management
# -------------------------------------------------------------
request_queue = queue.Queue()
task_status = {}


def process_3d_conversion(image_data, task_id):
    """Gradio Client দিয়ে Hunyuan3D-2 মডেল জেনারেট করার প্রসেস"""
    try:
        space_id = "tencent/Hunyuan3D-2"

        client = Client(space_id, hf_token=HF_TOKEN if HF_TOKEN else None)

        # ইমেজ ইনপুট হ্যান্ডলিং
        image_input = (
            handle_file(image_data)
            if isinstance(image_data, str) and os.path.exists(image_data)
            else image_data
        )

        # /shape_generation API কল
        result = client.predict(
            caption=None,
            image=image_input,
            mv_image_front=None,
            mv_image_back=None,
            mv_image_left=None,
            mv_image_right=None,
            steps=30,
            guidance_scale=5,
            seed=1234,
            octree_resolution=256,
            check_box_rembg=True,
            num_chunks=8000,
            randomize_seed=True,
            api_name="/shape_generation",
        )

        return {"success": True, "model_url": result}

    except Exception as e:
        print(f"Error during Gradio API call: {e}")
        return {"success": False, "error": str(e)}


def worker_thread():
    """লাইনের টাস্কগুলো ব্যাকগ্রাউন্ডে একটি একটি করে প্রসেস করবে"""
    while True:
        task_id, image_data = request_queue.get()
        task_status[task_id]["status"] = "processing"

        result = process_3d_conversion(image_data, task_id)

        if result["success"]:
            task_status[task_id]["status"] = "completed"
            task_status[task_id]["result"] = result["model_url"]
        else:
            task_status[task_id]["status"] = "failed"
            task_status[task_id]["error"] = result.get("error")

        request_queue.task_done()


threading.Thread(target=worker_thread, daemon=True).start()


# -------------------------------------------------------------
# ৩. API Endpoints
# -------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Server is running successfully!"})


@app.route("/generate-3d", methods=["POST"])
def generate_3d():
    data = request.json or {}
    image_data = data.get("image")

    if not image_data:
        return jsonify({"error": "No image provided"}), 400

    task_id = f"task_{int(time.time() * 1000)}"
    q_size = request_queue.qsize()
    estimated_wait_time = (q_size + 1) * 120

    task_status[task_id] = {
        "status": "pending",
        "wait_time": estimated_wait_time,
        "position": q_size + 1,
    }

    request_queue.put((task_id, image_data))

    return jsonify(
        {
            "task_id": task_id,
            "message": f"আপনি লাইনে আছেন ({q_size + 1} নম্বর)। অনুগ্রহ করে অপেক্ষা করুন।",
            "wait_time": estimated_wait_time,
        }
    )


@app.route("/status/<task_id>", methods=["GET"])
def check_status(task_id):
    status_info = task_status.get(task_id)
    if not status_info:
        return jsonify({"error": "Invalid Task ID"}), 404

    return jsonify(status_info)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
