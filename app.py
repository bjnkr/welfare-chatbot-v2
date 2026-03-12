import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
from google import genai
from google.genai import types
import io
from pypdf import PdfReader, PdfWriter

API_KEY = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else "직접입력"
st.set_page_config(page_title="bjn.kr 복지 챗봇", page_icon="🤖")

@st.cache_resource
def init_db():
  db_path = "./welfare_backup"
  gemini_ef = GoogleGenerativeAiEmbeddingFunction(api_key=API_KEY)
  db_client = chromadb.PersistentClient(path=db_path)
  return db_client.get_collection(name="welfare_docs", embedding_function=gemini_ef)

try:
collection = init_db()
genai_client = genai.Client(api_key=API_KEY)
st.success("통합 복지 지식 창고가 연결되었습니다!")
except Exception as e:
st.error(f"DB 로드 실패: {e}")

st.title("🤖 bjn.kr AI 복지 상담사")
st.info("국민기초생활보장 및 다양한 복지 안내서를 통합 학습했습니다.")

if "messages" not in st.session_state:
st.session_state.messages = []

for msg in st.session_state.messages:
st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("궁금한 복지 혜택을 물어보세요!"):
st.session_state.messages.append({"role": "user", "content": prompt})
st.chat_message("user").write(prompt)
