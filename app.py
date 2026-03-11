with st.spinner("지침서를 꼼꼼히 읽고 있습니다..."):
            # 1. DB 검색
            results = collection.query(query_texts=[prompt], n_results=10)
            pages = [m['page_num'] for m in results['metadatas'][0]]
            
            # --- ✨ 핵심 추가 로직: 실제 PDF 추출해서 넘겨주기 ---
            import io
            from pypdf import PdfReader, PdfWriter
            
            # 본인이 깃허브에 올린 원본 PDF 파일명으로 변경하세요!
            PDF_FILENAME = "국민기초생활보장.pdf" 
            
            reader = PdfReader(PDF_FILENAME)
            writer = PdfWriter()
            
            # 검색된 페이지 번호에 해당하는 페이지만 쏙쏙 뽑아내기
            for page_num in pages:
                idx = int(page_num) - 1 # 파이썬은 0번부터 세기 때문
                if 0 <= idx < len(reader.pages):
                    writer.add_page(reader.pages[idx])
            
            # 뽑아낸 페이지들을 바이트(메모리) 형태로 변환
            pdf_bytes_io = io.BytesIO()
            writer.write(pdf_bytes_io)
            pdf_bytes = pdf_bytes_io.getvalue()
            
            # Gemini에게 넘겨줄 '문서 뭉치' 완성
            pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
            # ------------------------------------------------
            
            # 3. Gemini에게 찐 PDF 데이터와 함께 질문하기!
            response = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    pdf_part,
                    f"당신은 대한민국 복지 안내 전문가입니다. 제공된 PDF 페이지의 표와 텍스트를 꼼꼼히 분석하여 다음 질문에 대답해 주세요. \n\n질문: {prompt}"
                ]
            )
            
            answer = response.text
            st.write(answer)
            st.caption(f"📍 참고 페이지: {', '.join(set([str(p) for p in pages]))}페이지")
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
