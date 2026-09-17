import time
import queue
import threading
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # ওয়েবসাইট থেকে এপিআই কলের অনুমতি দেয়

# -------------------------------------------------------------
# ১. Tencent Cloud / Gemini API Keys (আপনার ফ্রি কী-গুলো এখানে দিন)
# -------------------------------------------------------------
API_KEYS = [
    "YOUR_API_KEY_1",
    "YOUR_API_KEY_2",
    "YOUR_API_KEY_3"
]

current_key_index = 0
key_lock = threading.Lock()

# -------------------------------------------------------------
# ২. Request Queue এবং Task Status Management
# -------------------------------------------------------------
request_queue = queue.Queue()
task_status = {}  # task_id অনুযায়ী বর্তমান অবস্থা স্টোর থাকবে

def get_next_api_key():
    """একাধিক API Key রোটেশন করার লজিক"""
    global current_key_index
    with key_lock:
        key = API_KEYS[current_key_index]
        current_key_index = (current_key_index + 1) % len(API_KEYS)
        return key

def process_3d_conversion(image_data, task_id):
    """এপিআই দিয়ে ৩ডি মডেল জেনারেট করার প্রসেস"""
    attempts = 0
    max_attempts = len(API_KEYS)
    
    while attempts < max_attempts:
        api_key = get_next_api_key()
        try:
            # ---------------------------------------------------------
            # Tencent/Gemini API Call
            # ---------------------------------------------------------
            # উদাহরণস্বরূপ প্রসেসিং টাইম ধরে নেওয়া হলো (Real API Call Here)
            time.sleep(5) 
            
            # সফল হলে ৩ডি ফাইলের URL রিটার্ন করবে
            return {
                "success": True, 
                "model_url": f"https://your-domain.com/models/{task_id}.glb"
            }

        except Exception as e:
            attempts += 1
            print(f"Key error occurred. Switching to next key... ({attempts}/{max_attempts})")
            
    return {"success": False, "error": "All API Keys reached limit or server error."}

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

# ব্যাকগ্রাউন্ড ওয়ার্কার রান করা
threading.Thread(target=worker_thread, daemon=True).start()

# -------------------------------------------------------------
# ৩. API Endpoints
# -------------------------------------------------------------

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "Server is running successfully!"})

@app.route('/generate-3d', methods=['POST'])
def generate_3d():
    """ইউজার ছবি জমা দেওয়ার জন্য এনডপয়েন্ট"""
    data = request.json or {}
    image_data = data.get("image")
    
    if not image_data:
        return jsonify({"error": "No image provided"}), 400

    task_id = f"task_{int(time.time() * 1000)}"
    q_size = request_queue.qsize()
    estimated_wait_time = (q_size + 1) * 120  # প্রতি মডেলে ২ মিনিট ওয়েটিং টাইম ধরে
    
    task_status[task_id] = {
        "status": "pending",
        "wait_time": estimated_wait_time,
        "position": q_size + 1
    }
    
    request_queue.put((task_id, image_data))
    
    return jsonify({
        "task_id": task_id,
        "message": f"আপনি লাইনে আছেন ({q_size + 1} নম্বর)। অনুগ্রহ করে অপেক্ষা করুন।",
        "wait_time": estimated_wait_time
    })

@app.route('/status/<task_id>', methods=['GET'])
def check_status(task_id):
    """কাউন্টডাউন এবং স্ট্যাটাস চেক করার এনডপয়েন্ট"""
    status_info = task_status.get(task_id)
    if not status_info:
        return jsonify({"error": "Invalid Task ID"}), 404
        
    return jsonify(status_info)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
