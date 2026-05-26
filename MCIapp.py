from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import requests
import json
import re
import traceback
import os
from difflib import SequenceMatcher
from supabase import create_client, Client

app = Flask(__name__)
CORS(app)

# KONFIGURASI SUPABASE (DATABASE)
SUPABASE_URL = "https://guuohxjkoylcaxegrvsy.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imd1dW9oeGprb3lsY2F4ZWdydnN5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzkwODIwNTgsImV4cCI6MjA5NDY1ODA1OH0.jX_gVrHgo2DM1yHyBeTapwUEe59MokZivYVizr-6jpQ"

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("[INFO] Berhasil terhubung ke Supabase!")
except Exception as e:
    print(f"[ERROR] Gagal inisialisasi Supabase: {e}")
    supabase = None

# KONFIGURASI OLLAMA (AI CLINICAL SYSTEM)
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "clinical-ai"

HEADERS = {
    "Content-Type": "application/json"
}

CHAT_PROMPT = """You are an empathetic clinical AI assistant designed to conduct diagnostic interviews based on DSM-5 standards.
Tasks:
1. Manage dialogue flow (Exploration -> Crosscut -> Measurement -> Reasoning -> Closure).
2. Generate internal reasoning trace (Evidence Map) before speaking."""

REPORT_PROMPT = """You are a Clinical Expert System. 
Task: Analyze the full interview transcript and generate a comprehensive 'Diagnostic Report' in JSON format.

CRITICAL RULES:
1. STRICT JSON ONLY. Do NOT use parentheses `)` to close JSON objects. ONLY use curly braces `}`.
2. `criteria_map` MUST be a single flat object containing key-value pairs (e.g., {"A1": "...", "B1": "..."}), NOT multiple separated objects like {"A1": "..."} , {"B1": "..."}.
3. CRITICAL INSTRUCTION: Analyze each diagnostic instrument (e.g., ASRS, GAD-7) ONLY ONCE. Do not output duplicates. Ensure all strings inside the JSON are properly escaped.
4. Follow this skeleton structure exactly:
{
  "crosscut_summary": {},
  "instrument_summary": {},
  "diagnostic_competition": {
    "ranked": [ { "disorder": "", "confidence": 0.0, "criteria_map": { "A1": "", "B1": "" } } ],
    "excluded": []
  },
  "final_dx": "",
  "icd_code": { "code": "", "rationale": "" },
  "summary": "",
  "quality": { "realism_score": 0.0, "consistency_score": 0.0 }
}"""

@app.route('/')
def home():
    return jsonify({"status": "Backend Clinical AI Aktif"})

@app.route('/get_history', methods=['POST'])
def get_history():
    try:
        session_id = request.json.get('session_id', 'default_session')
        
        if not supabase:
            return jsonify({"history": []})
        
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
    try:
        data = request.json
        history = data.get('history', [])
        session_id = data.get('session_id', 'default_session')
        
        context_turns = []
        for msg in history[-12:]:
            role = "Patient" if msg['role'] == 'user' else "Clinician"
            context_turns.append(f"{role}: {msg['content']}")
        
        context = "\n".join(context_turns)
        is_complete = False
        user_message_count = sum(1 for msg in history if msg['role'] == 'user')
        
        prompt = f"<|im_start|>system\n{CHAT_PROMPT}<|im_end|>\n<|im_start|>user\n{context}<|im_end|>\n<|im_start|>assistant\n"
        
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "raw": True,
            "options": {
                "temperature": 0.0,
                "top_p": 0.1,
                "top_k": 1
            }
        }
        
        print(f"\n[INFO] Request Chat (Sesi: {session_id}) ke Ollama Lokal...")
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

        is_looping = False
        past_clinician_msgs = [msg['content'] for msg in history if msg['role'] == 'assistant']
        
        # Cek pengulangan pertanyaan AI (Threshold: 75%)
        for past_msg in past_clinician_msgs[-5:]:
            kemiripan = SequenceMatcher(None, final_response.lower(), past_msg.lower()).ratio()
            if kemiripan > 0.75:
                is_looping = True
                break
        
        # Batas pengaman agar memori tidak meledak
        if user_message_count >= 15:
            is_looping = True
            print("[INFO] Batas memori token tercapai (15 chat).")

        # Jika AI mengulang atau batas tercapai, suntikkan instruksi untuk langsung mendiagnosis
        if is_looping:
            print(f"\n[WARNING] Deteksi pengulangan/looping aktif! Memaksa AI memberikan kesimpulan...")
            is_complete = True
            
            forced_context = context + "\nPatient: [SYSTEM COMMAND TO CLINICIAN: I have provided enough information. You MUST STOP asking questions now. Summarize all my symptoms and provide your final diagnostic impression directly to me based on the DSM-5.]"
            override_prompt = f"<|im_start|>system\n{CHAT_PROMPT}<|im_end|>\n<|im_start|>user\n{forced_context}<|im_end|>\n<|im_start|>assistant\n"
            
            payload["prompt"] = override_prompt
            
            response_override = requests.post(OLLAMA_URL, json=payload, headers=HEADERS)
            response_override.raise_for_status()
            
            response_text = response_override.json().get("response", "").replace("<|im_end|>", "").strip()
            
            trace_match = re.search(r'<trace>(.*?)</trace>', response_text, re.DOTALL)
            if trace_match:
                trace_content = trace_match.group(1).strip()
                final_response = response_text.replace(trace_match.group(0), '').strip()
            else:
                trace_content = "Trace Override Aktif."
                final_response = response_text.strip()

        # Simpan ke Supabase
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
            "response": final_response,
            "is_complete": is_complete
        })
        
    except Exception as e:
        print("\n[ERROR CHAT]:")
        traceback.print_exc()
        return jsonify({"error": f"Backend Error: {str(e)}"}), 500

@app.route('/reset', methods=['POST'])
def reset_chat():
    try:
        session_id = request.json.get('session_id')
        if supabase and session_id:
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
        
        prompt = f"<|im_start|>system\n{REPORT_PROMPT}\n<|im_end|>\n<|im_start|>user\nHere is the full transcript of the patient interview:\n{transcript}\n\nGenerate the Final Diagnostic Report.<|im_end|>\n<|im_start|>assistant\n"
        
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "raw": True,
            "options": {
                "temperature": 0.1,
                "num_predict": 4096
            }
        }
        
        response = requests.post(OLLAMA_URL, json=payload, headers=HEADERS)
        response.raise_for_status()
        report_raw = response.json().get("response", "").strip()
        
        # =======================================================
        # SISTEM KEKEBALAN JSON (AUTO-FIX SINTAKS HALUSINASI AI)
        # =======================================================
        # 1. Obati kurung ')' nyasar untuk menutup objek besar
        report_raw = report_raw.replace("]}),", "]}},").replace("]})", "]}}")
        report_raw = report_raw.replace("}),", "},").replace("})", "}")
        
        # 2. Obati criteria_map yang terpecah menjadi banyak kurung kurawal
        report_raw = re.sub(r'(\"criteria_map\":\s*{[^{}]*)\},\s*\{([^{}]*})', r'\1, \2', report_raw)
        report_raw = re.sub(r'(\"criteria_map\":\s*{[^{}]*)\},\s*\{([^{}]*})', r'\1, \2', report_raw)
        # =======================================================
        
        parsed_json = None
        try:
            parsed_json = json.loads(report_raw)
            if supabase:
                try:
                    supabase.table('reports').insert({
                        "session_id": session_id,
                        "report_data": parsed_json
                    }).execute()
                except Exception as db_error:
                    print(f"[WARNING] Gagal menyimpan laporan ke DB: {db_error}")
        except json.JSONDecodeError:
            print("[INFO] AI gagal menyusun JSON sempurna, tapi draf kasar tetap dikirim ke UI.")

        return jsonify({
            "raw_text": report_raw,
            "report_json": parsed_json
        })
        
    except Exception as e:
        print("\n[ERROR REPORT]:")
        traceback.print_exc()
        return jsonify({"error": f"Backend Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
