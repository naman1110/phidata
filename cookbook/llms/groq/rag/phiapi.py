from collections import defaultdict
from flask import Flask, request, jsonify 
import os
from flask_cors import CORS

from phi.assistant import Assistant
from phi.document import Document
from phi.document.reader.pdf import PDFReader
from phi.document.reader.website import WebsiteReader
from phi.utils.log import logger
from typing import List
import logging
from logging.handlers import RotatingFileHandler



from assistant import get_groq_assistant  
from cryptography.fernet import Fernet
import base64
import binascii
import urllib.parse
import shutil

app = Flask(__name__)
upload_folder = 'uploads'
os.makedirs(upload_folder, exist_ok=True)
ds=defaultdict(list)  
p_llm_model = "llama3-70b-8192"
p_embeddings_model = "text-embedding-3-large"
custom_key = 42 #"NetComLearning@PhiRagChatBot"
CORS(app) 

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s:%(message)s'))

app.logger.addHandler(handler)

@app.route('/receive-file', methods=['POST'])
def receive_file():
    # Get Knowledge Base (KB) name from request parameter
    folder_name_param = request.form.get('kb_name')
    folder_name = folder_name_param
    if folder_name_param is None:
        folder_name = "General-Domain"
    
    # Create folder if it doesn't exist
    folder_path = os.path.join(upload_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Handle file uploads
    uploaded_files = request.files.getlist('file')
    rag_assistant = get_groq_assistant(llm_model=p_llm_model, embeddings_model=p_embeddings_model,user_id=folder_name) 
    for uploaded_file in uploaded_files:
        if uploaded_file.filename != '':
            file_path = os.path.join(folder_path, uploaded_file.filename)
            uploaded_file.save(file_path)
            process_file(file_path,rag_assistant,folder_name_param,uploaded_file.filename)  # Process the file directly for the knowledge base

        else: logging.error(f"Error in processing file ")
    
    return jsonify({'message': 'Files uploaded successfully', 'kb_name': folder_name_param, 'kb_path':folder_path}),200


def process_file(filepath,rag_assistant,user_id,name):
    print("Processing and integrating file into knowledge base:", filepath)
    if rag_assistant.knowledge_base:
        with open(filepath, 'rb') as file:
                # pdf_file = io.BytesIO(file.read())
                reader = PDFReader(chunk_size=1750)
                # rag_name = name
                rag_documents: List[Document] = reader.read(filepath)
                if rag_documents:
                    rag_assistant.knowledge_base.load_documents(rag_documents, upsert=True,skip_existing=True)
                    logging.debug("PDF processed and loaded into the knowledge base")
                    ds[user_id].append(name)
                else:
                    logging.error(f"Could not read PDF {filepath}")

@app.route('/listKB', methods=['GET'])
def list_kb():
    if request.is_json:
      data = request.get_json()
      id=data.get('kb_name')
      directory_path = upload_folder+"/"+id
      files = list_files_in_folder(directory_path)
      if files:
          return jsonify({"kb_list": files, "kb_name":id}),200
      else:
          return jsonify({"kb_list": files, "kb_name":id,'message': 'The Knowledge Base does not exists.'}),200
    else:
         return jsonify({"error": "Missing parameters in request"}), 400
         
  
 
def list_files_in_folder(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            print(file)
            file_list.append(file)
    return file_list

@app.route('/chat', methods=['POST'])
def rag_chat():  
    data = None
    user_prompt = None
    id=None

    if request.is_json:
          data = request.get_json()
          user_prompt = data.get('user_prompt')
          id=data.get('kb_name')

    rag_assistant = get_groq_assistant(llm_model=p_llm_model, embeddings_model=p_embeddings_model,user_id=id)
    rag_assistant_run_ids: List[str] = rag_assistant.storage.get_all_run_ids(user_id=id) 
    if not rag_assistant_run_ids:    
     run_id=rag_assistant.create_run()
    else: run_id=rag_assistant_run_ids[0]
    rag_assistant=get_groq_assistant(llm_model=p_llm_model, embeddings_model=p_embeddings_model,run_id=run_id,user_id=id)
     
    response=''
    
    for delta in rag_assistant.run(user_prompt):
                response += delta 
    
    
    logging.info(f"run ids: {rag_assistant_run_ids} for user id:{rag_assistant.user_id}")
    return jsonify({"content": response,"kb_name":id}),200
    #return response

@app.route('/clear', methods=['POST'])
def clear_db():
    try:
         
        # Extract the 'user_prompt' from the JSON data
        data = request.get_json()
        id=data.get('kb_name')
        directory_path = upload_folder+"/"+id
        rag_assistant = get_groq_assistant(llm_model=p_llm_model, embeddings_model=p_embeddings_model,user_id=id)
        logging.info("Clearing KB : "+id)
        clear_status = rag_assistant.knowledge_base.vector_db.clear()
        if clear_status:
            try:
                shutil.rmtree(directory_path)
                logger.info(f"Directory '{directory_path}' deleted successfully.")
            except OSError as e:
                logger.info(f"Error deleting directory '{directory_path}': {e}")
        
        return jsonify({'message': 'Knowledge Base Cleared successfully.', 'kb_name': id,"kb_path":directory_path}),200
    except:
         return jsonify({'message': 'The Knowledge Base does not exists.', 'kb_name': id,"kb_path":directory_path}),404
         
    #return "Knowledge base cleared"
     






if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9000)
