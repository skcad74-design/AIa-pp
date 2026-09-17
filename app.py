import base64
import os
import queue
import tempfile
import threading
import time
from flask import Flask, jsonify, request
from flask_cors import CORS
from gradio_client import Client, handle_file

app = Flask(__name__)

# CORS সমস্যা সমাধানের জন্য সম্পূর্ণ ওপেন কনফিগারেশন
CORS(app, resources={r"/*": {"origins": "*"}})

HF_TOKEN = os.getenv("HF_TOKEN", "")
request_queue = queue.Queue()
task_status = {}


def save_base64_to_temp_file(base64_string):
    """Base64 ইমেজ থেকে টেম্পোরারি ফাইল তৈরি করে"""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        image_bytes = base64.b64decode(base64_string)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_file.write(image_bytes)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None


def process_3d_conversion(image_data, task_id):
    """Gradio Client দিয়ে 3D মডেল জেনারেট করে"""
    temp_file_path = None
    try:
        space_id = "tencent/Hunyuan3D-2"
        client = Client(space_id, hf_token=HF_TOKEN if HF_TOKEN else None)

        if isinstance(image_data, str) and (
            image_data.startswith("data:image") or len(image_data) > 500
        ):
            temp_file_path = save_base64_to_temp_file(image_data)
            image_input = handle_file(temp_file_path)
        elif isinstance(image_data, str) and os.path.exists(image_data):
            image_input = handle_file(image_data)
        else:
            image_input = image_data

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

        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        return {"success": True, "model_url": result}

    except Exception as e:
        print(f"Error during Gradio API call: {e}")
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return {"success": False, "error": str(e)}


def worker_thread():
    """ব্যাকগ্রাউন্ডে টাস্ক ফিল্টারিং ও প্রসেসিং"""
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
    estimated_wait_time = (q_size + 1) * 180

    task_status[task_id] = {
        "status": "pending",
        "wait_time": estimated_wait_time,
        "position": q_size + 1,
    }

    request_queue.put((task_id, image_data))

    return jsonify(
        {
            "task_id": task_id,
            "message": "আপনার রিকোয়েস্ট প্রক্রিয়াধীন রয়েছে।",
            "wait_time": estimated_wait_time,
        }
    )


@app.route("/status/<task_id>", methods=["GET"])
def check_status(task_id):
    status_info = task_status.get(task_id)

    # ৪০৪ এরর এড়াতে টাস্ক না পাওয়া গেলেও ফ্রন্টএন্ডকে উপযুক্ত তথ্য জানানো
    if not status_info:
        return (
            jsonify(
                {
                    "status": "failed",
                    "error": "Task not found or server restarted.",
                }
            ),
            200,
        )

    return jsonify(status_info)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
