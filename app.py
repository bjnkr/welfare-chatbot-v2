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

    with st.chat_message("assistant"):
        with st.spinner("여러 지침서를 꼼꼼히 뒤적이고 있습니다..."):
            try:
                results = collection.query(query_texts=[prompt], n_results=10)
                metadatas = results['metadatas'][0]
                
                files_to_pages = {}
                for m in metadatas:
                    source_file = m['source']
                    page_num = m['page_num']
                    if source_file not in files_to_pages:
                        files_to_pages[source_file] = []
                    files_to_pages[source_file].append(page_num)
                
                writer = PdfWriter()
                
                for source_file, pages in files_to_pages.items():
                    reader = PdfReader(source_file)
                    for page_num in pages:
                        idx = int(page_num) - 1
                        if 0 <= idx < len(reader.pages):
                            writer.add_page(reader.pages[idx])
                
                pdf_bytes_io = io.BytesIO()
                writer.write(pdf_bytes_io)
                pdf_bytes = pdf_bytes_io.getvalue()
                
                pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
                
                response = genai_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        pdf_part,
                        f"당신은 bjn.kr의 스마트 복지 챗봇입니다. 제공된 PDF 문서를 바탕으로 아래 지침을 엄격히 지켜 답변하세요.\n1. 길이 자동 조절:\n- 단순 확인 질문: 서론 없이 핵심 정답만 2~3문장 이내로 아주 짧고 명확하게 답변하세요.\n- 복잡한 질문: 상세히 설명하되, 반드시 글머리 기호(-, *)를 사용하여 요약하세요.\n2. 금지 사항: 인사말과 기계적인 서론/결론은 절대 쓰지 마세요.\n3. 명확성: 조건에 따라 결과가 달라지는 경우 명확한 가이드라인만 제시하세요.\n질문: {prompt}"
                    ]
                )
                
                answer = response.text
                st.write(answer)
                
                ref_text = []
                for source_file, pages in files_to_pages.items():
                    unique_pages = sorted(list(set([int(p) for p in pages])))
                    ref_text.append(f"{source_file} ({', '.join(map(str, unique_pages))}p)")
                
                st.caption(f"📍 참고 문서: {' / '.join(ref_text)}")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except FileNotFoundError as fnfe:
                st.error(f"🚨 깃허브에 파일이 없습니다: {fnfe}")
            except Exception as e:
                st.error(f"🚨 오류가 발생했습니다: {e}")
