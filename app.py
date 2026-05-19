from flask import Flask, render_template, request, jsonify
import requests
import json
import re
import traceback
import os
from supabase import create_client, Client

app = Flask(__name__)

# ==========================================
# KONFIGURASI SUPABASE (DATABASE)
# ==========================================
SUPABASE_URL = os.environ.get("SUPAURL")
SUPABASE_KEY = os.environ.get("SUPAKEY")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("[INFO] Berhasil terhubung ke Supabase!")
except Exception as e:
    print(f"[ERROR] Gagal inisialisasi Supabase: {e}")
    supabase = None

# ==========================================
# KONFIGURASI OLLAMA (AI CLINICAL SYSTEM)
# ==========================================
NGROK_URL = "https://bagginess-scruffy-starship.ngrok-free.dev"
OLLAMA_URL = f"{NGROK_URL}/api/generate"
MODEL_NAME = "clinical-ai"

HEADERS = {
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true"
}

CHAT_PROMPT = """You are an empathetic clinical AI assistant designed to conduct diagnostic interviews based on DSM-5 standards.
Tasks:
1. Manage dialogue flow (Exploration -> Crosscut -> Measurement -> Reasoning -> Closure).
2. Generate internal reasoning trace (Evidence Map) before speaking."""

REPORT_PROMPT = """You are a Clinical Expert System. 
Task: Analyze the full interview transcript and generate a comprehensive 'Diagnostic Report' in JSON format.
The JSON must strictly follow the schema provided in the training data."""

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get_history', methods=['POST'])
def get_history():
    """Mengambil riwayat berdasarkan Session ID"""
    try:
        # Ambil session_id dari frontend
        session_id = request.json.get('session_id', 'default_session')
        
        if not supabase:
            return jsonify({"history": []})
        
        # Tarik data HANYA yang miliknya session_id ini
        response = supabase.table('messages').select('*').eq('session_id', session_id).order('created_at', desc=False).limit(50).execute()
        
        formatted_history = []
        for msg in response.data:
            role = "user" if msg['sender'] == "Patient" else "assistant"
            formatted_history.append({
                "role": role,
                "content": msg['content'],
                "trace": msg.get('trace_log', '') if msg.get('trace_log') else ""
            })
            
        return jsonify({"history": formatted_history})
        
    except Exception as e:
        print("\n[ERROR GET HISTORY]:")
        traceback.print_exc()
        return jsonify({"error": f"Gagal memuat riwayat: {str(e)}"}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Menerima chat dan menyimpannya beserta Session ID"""
    try:
        data = request.json
        history = data.get('history', [])
        session_id = data.get('session_id', 'default_session')
        
        context_turns = []
        for msg in history[-10:]:
            role = "Patient" if msg['role'] == 'user' else "Clinician"
            context_turns.append(f"{role}: {msg['content']}")
        
        context = "\n".join(context_turns)
        prompt = f"<|im_start|>system\n{CHAT_PROMPT}<|im_end|>\n<|im_start|>user\n{context}<|im_end|>\n<|im_start|>assistant\n"
        
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "raw": True
        }
        
        print(f"\n[INFO] Request Chat (Sesi: {session_id}) ke Ollama...")
        response = requests.post(OLLAMA_URL, json=payload, headers=HEADERS)
        response.raise_for_status()
        
        response_text = response.json().get("response", "").replace("<|im_end|>", "").strip()
        
        trace_match = re.search(r'<trace>(.*?)</trace>', response_text, re.DOTALL)
        if trace_match:
            trace_content = trace_match.group(1).strip()
            final_response = response_text.replace(trace_match.group(0), '').strip()
        else:
            trace_content = "Trace tidak terdeteksi."
            final_response = response_text.strip()

        # Sinkronisasi ke Supabase DENGAN session_id
        if supabase:
            try:
                last_user_msg = history[-1]['content'] if history and history[-1]['role'] == 'user' else ""
                if last_user_msg:
                    supabase.table('messages').insert({
                        "session_id": session_id,
                        "sender": "Patient", 
                        "content": last_user_msg
                    }).execute()
                
                supabase.table('messages').insert({
                    "session_id": session_id,
                    "sender": "Clinician", 
                    "content": final_response,
                    "trace_log": trace_content
                }).execute()
            except Exception as db_error:
                print(f"[WARNING] Gagal backup ke Supabase: {db_error}")

        return jsonify({
            "trace": trace_content, 
            "response": final_response
        })
        
    except Exception as e:
        print("\n[ERROR CHAT]:")
        traceback.print_exc()
        return jsonify({"error": f"Backend Error: {str(e)}"}), 500


@app.route('/reset', methods=['POST'])
def reset_chat():
    """Menghapus pesan khusus untuk Session ID saat ini"""
    try:
        session_id = request.json.get('session_id')
        if supabase and session_id:
            # Hapus data HANYA yang milik session ini
            supabase.table('messages').delete().eq('session_id', session_id).execute()
            print(f"[INFO] Database di-reset untuk sesi: {session_id}.")
            
        return jsonify({"status": "success"})
    except Exception as e:
        print("\n[ERROR RESET]:")
        traceback.print_exc()
        return jsonify({"error": f"Gagal mereset database: {str(e)}"}), 500


@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        data = request.json
        history = data.get('history', [])
        session_id = data.get('session_id', 'default_session')
        
        transcript = "\n".join([f"{'Clinician' if m['role']=='assistant' else 'Patient'}: {m['content']}" for m in history])
        
        prompt = f"<|im_start|>system\n{REPORT_PROMPT}<|im_end|>\n<|im_start|>user\nHere is the full transcript of the patient interview:\n{transcript}\n\nGenerate the Final Diagnostic Report.<|im_end|>\n<|im_start|>assistant\n"
        
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "raw": True
        }
        
        response = requests.post(OLLAMA_URL, json=payload, headers=HEADERS)
        response.raise_for_status()
        report_raw = response.json().get("response", "")
        
        parsed_json = json.loads(report_raw)

        if supabase:
            try:
                # Simpan report dilengkapi dengan session_id
                supabase.table('reports').insert({
                    "session_id": session_id,
                    "report_data": parsed_json
                }).execute()
            except Exception as db_error:
                print(f"[WARNING] Gagal menyimpan laporan ke Supabase: {db_error}")

        return jsonify({"report": parsed_json})
        
    except Exception as e:
        print("\n[ERROR REPORT]:")
        traceback.print_exc()
        return jsonify({"error": f"Backend Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
