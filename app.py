import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
from google import genai
from google.genai import types
import io
from pypdf import PdfReader, PdfWriter

# --- 1. 설정 및 보안 ---
# Streamlit Cloud의 Secrets에서 API 키를 가져옵니다.
API_KEY = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else "직접입력"

st.set_page_config(page_title="bjn.kr 복지 챗봇", page_icon="🤖")

# --- 2. 로컬에 저장된 DB 불러오기 ---
@st.cache_resource
def init_db():
    db_path = "./welfare_backup" 
    gemini_ef = GoogleGenerativeAiEmbeddingFunction(api_key=API_KEY)
    db_client = chromadb.PersistentClient(path=db_path)
    return db_client.get_collection(name="welfare_docs", embedding_function=gemini_ef)

try:
    collection = init_db()
    genai_client = genai.Client(api_key=API_KEY)
    st.success("복지 지식 창고가 연결되었습니다!")
except Exception as e:
    st.error(f"DB 로드 실패: {e}")

# --- 3. 채팅 UI 구현 ---
st.title("🤖 bjn.kr AI 복지 상담사")
st.info("366페이지 분량의 2026년 복지 안내서를 학습했습니다.")

# 대화 기록 저장용 세션 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 내용 화면에 출력
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- 4. 질문 입력 및 답변 생성 ---
if prompt := st.chat_input("궁금한 복지 혜택을 물어보세요! (예: 3인가구 주거급여 얼마야?)"):
    # 사용자가 입력한 질문 화면에 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 챗봇 답변 영역
    with st.chat_message("assistant"):
        with st.spinner("지침서 원본을 꼼꼼히 읽고 있습니다... (약 3~5초 소요)"):
            try:
                # 1) 벡터 DB에서 관련 페이지 10장 검색
                results = collection.query(query_texts=[prompt], n_results=10)
                pages = [m['page_num'] for m in results['metadatas'][0]]
                
                # 2) 원본 PDF에서 해당 페이지만 쏙 뽑아내기
                # ★ 중요: 깃허브에 올린 원본 PDF 파일 이름과 똑같이 적어주세요! ★
                PDF_FILENAME = "국민기초생활보장.pdf" 
                
                reader = PdfReader(PDF_FILENAME)
                writer = PdfWriter()
                
                # 검색된 페이지 번호에 해당하는 페이지만 추출
                for page_num in pages:
                    idx = int(page_num) - 1 # 파이썬은 0페이지부터 셈
                    if 0 <= idx < len(reader.pages):
                        writer.add_page(reader.pages[idx])
                
                # 추출한 페이지들을 메모리에 임시 저장 (바이트 변환)
                pdf_bytes_io = io.BytesIO()
                writer.write(pdf_bytes_io)
                pdf_bytes = pdf_bytes_io.getvalue()
                
                # Gemini에게 보낼 수 있는 파일 형태로 묶기
                pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
                
                # 3) Gemini에게 최종 답변 요청
                response = genai_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        pdf_part,
                        f"당신은 대한민국 복지 안내 전문가입니다. 제공된 PDF 문서의 표와 텍스트를 아주 꼼꼼히 분석하여 다음 질문에 정확하게 대답해 주세요. \n\n질문: {prompt}"
                    ]
                )
                
                # 4) 화면에 결과 출력
                answer = response.text
                st.write(answer)
                
                # 참고한 페이지 번호 보기 좋게 정렬해서 하단에 표시
                unique_pages = sorted(list(set([int(p) for p in pages])))
                st.caption(f"📍 참고 페이지: {', '.join(map(str, unique_pages))}페이지")
                
                # 대화 기록에 챗봇 답변 저장
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except FileNotFoundError:
                st.error(f"🚨 오류: 깃허브 저장소에 '{PDF_FILENAME}' 파일이 업로드되어 있지 않습니다! 파일을 꼭 함께 올려주세요.")
            except Exception as e:
                st.error(f"🚨 실행 중 오류가 발생했습니다: {e}")           
