import os
import streamlit as st
import pandas as pd
from core_video import extract_audio, transcribe_audio, spell_check_segments, extract_and_filter_frames, spell_check_frames

st.set_page_config(page_title="영상 맞춤법 검사기", page_icon="🎥", layout="wide")

st.title("🎥 영상 대본 및 화면 글자 자동 교정기")
st.markdown("MP4 영상을 업로드하면 AI가 **음성 대본(STT)** 및 **화면 속 텍스트(OCR)**를 분석하여 맞춤법/띄어쓰기 오류를 모두 잡아줍니다.")

# -----------------
# 비밀번호 및 API 키 보안 처리 (클라우드 배포 필수 사항)
# -----------------
st.sidebar.header("⚙️ 설정")

# 🚨 치명적 보안 위협: 소스 코드에 키를 남겨둔 채로 깃허브(인터넷)에 올리면 해커들이 1분 안에 키를 훔쳐갑니다!
# 따라서 사용자님의 실제 키를 파일에서 삭제하고, 안전한 클라우드 보안 금고(Secrets)에서 불러오도록 구조를 변경했습니다.
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = ""

if not api_key:
    st.sidebar.warning("💻 클라우드 보안 환경(Secrets)이 없으므로 일회성으로 키를 직접 입력해야 합니다.")
    api_key = st.sidebar.text_input("OpenAI API Key 입력", type="password", help="sk-... 로 시작하는 키를 잠시만 입력해주세요.")
    
    if not api_key:
        st.stop()

st.sidebar.markdown("""
**멀티모달 검사 원리**
1. **음성**: 오디오 추출 후 Whisper API 로컬 대사 변환.
2. **화면**: 1초 간격 OpenCV 화면 분석 후 안 바뀐 유사 화면 스킵, 이후 남은 주요 정보 화면만 OpenAI GPT-4o Vision API로 판독.
3. 두 결과를 시간대에 맞춰 하나의 통합 표로 제공합니다.
""")

st.sidebar.divider()
st.sidebar.header("🎯 검사 대상 선택")
check_audio = st.sidebar.checkbox("음성 대본 검사", value=True)
check_screen = st.sidebar.checkbox("화면 텍스트 검사", value=True)

if not api_key or api_key == "sk-여기에_실제_API_키를_붙여넣으세요":
    st.error("app_video.py 파일을 메모장이나 에디터로 열고, 15번째 줄의 api_key 변수에 실제 키를 붙여넣어 주세요!")
    st.stop()

if not check_audio and not check_screen:
    st.warning("👈 좌측에서 검사 대상을 최소 하나 이상 체크해주세요.")
    st.stop()

# -----------------
# 메인 영역
# -----------------
uploaded_file = st.file_uploader("검사할 동영상 파일(MP4)을 업로드하세요.", type=["mp4"])

if uploaded_file is not None:
    video_path = "temp_video.mp4"
    audio_path = "temp_audio.mp3"
    
    st.video(uploaded_file)
    
    if st.button("🚀 멀티모달 맞춤법 검사 시작", type="primary", use_container_width=True):
        
        with st.spinner("비디오 파일을 분석을 위해 서버 임시 변환 중..."):
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        all_results = []
        
        # 1. 화면 텍스트 검사 진행 (선택 시)
        if check_screen:
            st.divider()
            with st.spinner("🖼️ 영상에서 핵심 프레임(화면) 추출 및 중복 제거 중 (OpenCV)..."):
                unique_frames = extract_and_filter_frames(video_path, sample_rate=1.5, diff_threshold=25.0)
                if unique_frames:
                    st.success(f"영상 최적화 완료: 비슷한 화면을 건너뛰고 총 {len(unique_frames)}장의 핵심 화면을 분석합니다.")
                else:
                    st.warning("분석할 만한 화면이 감지되지 않았습니다.")
                
            if unique_frames:
                with st.spinner("👁️ OpenAI Vision AI가 화면 속 텍스트를 읽고 시각 검사 중 (시간이 다소 소요됩니다)..."):
                    screen_results = spell_check_frames(unique_frames, api_key)
                    all_results.extend(screen_results)
                    st.success("✅ 화면 텍스트 검사 완료!")

        # 2. 음성 대본 검사 진행 (선택 시)
        if check_audio:
            st.divider()
            with st.spinner("🎧 동영상에서 오디오 분리 중..."):
                success = extract_audio(video_path, audio_path)
            
            if success:
                st.success("오디오 분리 완료.")
                with st.spinner("🗣️ 음성을 텍스트로 변환 중 (Whisper API)..."):
                    try:
                        segments = transcribe_audio(audio_path, api_key)
                    except Exception as e:
                        st.error(f"오디오 변환 오류: {e}")
                        segments = []
                        
                if segments:
                    with st.spinner("🔍 대본의 전문 맞춤법을 검사하는 중 (GPT-4o-mini)..."):
                        audio_results = spell_check_segments(segments, api_key)
                        
                        for r in audio_results:
                            # 구분 컬럼 추가 후 딕셔너리 순서 재정렬
                            r["구분"] = "음성 대본"
                            sorted_r = {"구분": "음성 대본", "시간": r["시간"], "수정 전": r["수정 전"], "수정 후": r["수정 후"]}
                            all_results.append(sorted_r)
                            
                    st.success("✅ 음성 대본 검사 완료!")
                else:
                    st.warning("유의미한 음성이 잡히지 않았습니다.")
            else:
                st.error("오디오 추출에 실패하여 음성 검사를 건너뜁니다.")

        # -----------------
        # 검사 결과 통합 출력
        # -----------------
        st.divider()
        st.subheader("📊 통합 문장 맞춤법 교정 결과 & 하이라이트")
        if len(all_results) > 0:
            df = pd.DataFrame(all_results)
            df = df[["구분", "시간", "수정 전", "수정 후"]]
            
            df.sort_values(by="시간", inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            # HTML 렌더링을 위해 st.markdown과 HTML 스타일 테이블 사용
            # st.dataframe에서는 기본적으로 HTML 태그가 동작하지 않음
            
            html_table = df.to_html(escape=False, index=False)
            
            # CSS를 약간 가미하여 표를 예쁘게 만듦
            styled_html = f"""
            <style>
            .result-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-family: 'Malgun Gothic', sans-serif; }}
            .result-table th {{ background-color: #2b313e; color: #fff; padding: 12px; text-align: left; border: 1px solid #444; }}
            .result-table td {{ padding: 12px; border: 1px solid #ddd; background-color: #ffffff; color: #000; font-size: 15px; vertical-align: top; }}
            .result-table tr:nth-child(even) td {{ background-color: #f7f9fb; }}
            </style>
            <div style="overflow-x: auto;">
            {html_table.replace('<table border="1" class="dataframe">', '<table class="result-table">')}
            </div>
            """
            st.markdown(styled_html, unsafe_allow_html=True)
            
            st.info(f"💡 총 {len(all_results)}건의 교정 제안이 반영되었습니다. (오류 항목 붉은색 표시)")
            
            # 다운로드 버튼 (다운로드 데이터는 HTML 태그를 제거한 순수 텍스트로 전환)
            import re
            df_export = df.copy()
            df_export["수정 전"] = df_export["수정 전"].apply(lambda x: re.sub(r'<[^>]+>', '', x))
            df_export["수정 후"] = df_export["수정 후"].apply(lambda x: re.sub(r'<[^>]+>', '', x))
            
            csv = df_export.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 통합 수정 내역 엑셀/CSV로 다운로드",
                data=csv,
                file_name='멀티모달_교정_결과.csv',
                mime='text/csv',
            )
        else:
            st.success("🎉 분석 결과, 교정을 요하는 오류가 발견되지 않았습니다. 완벽합니다!")
            
        # 임시 파일 정리
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
