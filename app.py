import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
from google import genai
from google.genai import types
import os

# --- 1. 설정 및 보안 ---
# Streamlit Cloud의 Secrets 기능을 사용해 API 키를 숨기는 것이 안전합니다.
# 직접 입력할 경우: API_KEY = "AIza..."
API_KEY = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else "직접입력"

st.set_page_config(page_title="bjn.kr 복지 챗봇", page_icon="🤖")

# --- 2. 로컬에 저장된 DB 불러오기 ---
@st.cache_resource
def init_db():
    # GitHub에 welfare_backup 폴더를 함께 올렸다고 가정합니다.
    db_path = "./welfare_backup" 
    gemini_ef = GoogleGenerativeAiEmbeddingFunction(api_key=API_KEY)
    db_client = chromadb.PersistentClient(path=db_path)
    return db_client.get_collection(name="welfare_docs", embedding_function=gemini_ef)

# 초기화
try:
    collection = init_db()
    genai_client = genai.Client(api_key=API_KEY)
    st.success("복지 지식 창고가 연결되었습니다!")
except Exception as e:
    st.error(f"DB 로드 실패: {e}")

# --- 3. 채팅 UI 구현 ---
st.title("🤖 bjn.kr AI 복지 상담사")
st.info("366페이지 분량의 2026년 복지 안내서를 학습했습니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("지침서를 확인하고 있습니다..."):
            # 1. 벡터 DB에서 관련 문구 10개 검색
            results = collection.query(query_texts=[prompt], n_results=10)
            
            # 2. 검색된 페이지 번호 추출 (메타데이터 활용)
            pages = [str(m['page_num']) for m in results['metadatas'][0]]
            
            # 3. Gemini에게 답변 생성 요청 (이전 테스트 로직 적용)
            # 여기서는 편의상 간단히 구현하지만, 이전에 성공했던 
            # PDF 바이트 전송 로직을 결합하면 더 강력해집니다.
            response = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[f"지침서 내용 검색 결과: {results['documents'][0]}\n질문: {prompt}"]
            )
            
            answer = response.text
            st.write(answer)
            st.caption(f"📍 참고 페이지: {', '.join(set(pages))}페이지")
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
