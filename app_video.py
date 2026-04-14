import os
import re
import streamlit as st
import pandas as pd
from core_video import (
    extract_audio,
    transcribe_audio,
    spell_check_segments,
    extract_and_filter_frames,
    spell_check_frames,
)

# ─────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(page_title="영상 맞춤법 검사기", page_icon="🎥", layout="wide")

st.title("🎥 영상 대본 및 화면 글자 자동 교정기")
st.markdown(
    "MP4 영상을 업로드하면 AI가 **음성 대본(STT)** 및 **화면 속 텍스트(OCR)**를 분석하여 "
    "맞춤법·오타·띄어쓰기 오류를 잡아줍니다."
)

# ─────────────────────────────────────────────
# 공통 CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* 결과 카드 */
.card {
    background: #fff;
    border: 1px solid #e0e4ec;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 14px;
    font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
    font-size: 13px;
    color: #666;
    border-bottom: 1px solid #f0f0f0;
    padding-bottom: 8px;
}
.badge-audio {
    background: #4f8ef7;
    color: white;
    border-radius: 5px;
    padding: 2px 9px;
    font-size: 12px;
    font-weight: bold;
}
.badge-screen {
    background: #22b07d;
    color: white;
    border-radius: 5px;
    padding: 2px 9px;
    font-size: 12px;
    font-weight: bold;
}
.timestamp {
    background: #f3f4f6;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 12px;
    color: #555;
    font-family: monospace;
}
.label-before {
    font-size: 12px;
    font-weight: bold;
    color: #999;
    margin-bottom: 4px;
    letter-spacing: 0.5px;
}
.label-after {
    font-size: 12px;
    font-weight: bold;
    color: #4f8ef7;
    margin-bottom: 4px;
    letter-spacing: 0.5px;
}
.text-before {
    background: #fff8f8;
    border-left: 3px solid #f87171;
    border-radius: 4px;
    padding: 8px 12px;
    font-size: 15px;
    color: #333;
    line-height: 1.7;
    margin-bottom: 8px;
}
.text-after {
    background: #f0fdf4;
    border-left: 3px solid #34d399;
    border-radius: 4px;
    padding: 8px 12px;
    font-size: 15px;
    color: #333;
    line-height: 1.7;
    margin-bottom: 6px;
}
.reason-box {
    font-size: 12px;
    color: #888;
    margin-top: 4px;
}
.no-result {
    text-align: center;
    padding: 40px;
    color: #aaa;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 결과 카드 렌더링 함수
# ─────────────────────────────────────────────
def render_result_cards(results, badge_type="audio"):
    if not results:
        st.markdown('<div class="no-result">✅ 교정이 필요한 항목이 없습니다.</div>', unsafe_allow_html=True)
        return

    badge_class = "badge-audio" if badge_type == "audio" else "badge-screen"
    badge_label = "음성 대본" if badge_type == "audio" else "화면 텍스트"

    for i, r in enumerate(results, 1):
        time_str = r.get("시간", "")
        before   = r.get("수정 전", "")
        after    = r.get("수정 후", "")
        reason   = r.get("교정 사유", "")

        reason_html = f'<div class="reason-box">📌 {reason}</div>' if reason else ""

        st.markdown(f"""
        <div class="card">
            <div class="card-header">
                <span class="{badge_class}">{badge_label}</span>
                <span class="timestamp">{time_str}</span>
                <span style="margin-left:auto; color:#bbb; font-size:12px;">#{i}</span>
            </div>
            <div class="label-before">수정 전</div>
            <div class="text-before">{before}</div>
            <div class="label-after">수정 후</div>
            <div class="text-after">{after}</div>
            {reason_html}
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CSV 다운로드 헬퍼
# ─────────────────────────────────────────────
def make_csv(results):
    df = pd.DataFrame(results)
    cols = ["구분", "시간", "수정 전", "수정 후", "교정 사유"]
    df = df[[c for c in cols if c in df.columns]]
    for col in ["수정 전", "수정 후"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: re.sub(r'<[^>]+>', '', x))
    return df.to_csv(index=False).encode("utf-8-sig")

# ─────────────────────────────────────────────
# 사이드바 — API 키
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ 설정")

try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = ""

if not api_key:
    st.sidebar.warning("아래에 OpenAI API 키를 직접 입력하세요.")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="sk-... 형태의 키")
    if not api_key:
        st.info("👈 사이드바에서 OpenAI API 키를 입력하면 검사가 시작됩니다.")
        st.stop()

# ─────────────────────────────────────────────
# 사이드바 — 검사 대상
# ─────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.header("🎯 검사 대상")
check_audio  = st.sidebar.checkbox("🎧 음성 대본 검사", value=True)
check_screen = st.sidebar.checkbox("🖼️ 화면 텍스트 검사", value=True)

if not check_audio and not check_screen:
    st.warning("👈 검사 대상을 최소 하나 이상 체크해주세요.")
    st.stop()

# ─────────────────────────────────────────────
# 사이드바 — 고급 설정
# ─────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.header("🔧 고급 설정")

with st.sidebar.expander("화면 프레임 설정", expanded=False):
    sample_rate     = st.slider("프레임 추출 간격 (초)", 0.5, 5.0, 1.5, 0.5)
    diff_threshold  = st.slider("화면 변화 감지 임계값", 5.0, 50.0, 25.0, 5.0)
    batch_size      = st.selectbox("Vision API 배치 크기", [2, 3, 4, 5], index=1)

with st.sidebar.expander("음성 검사 설정", expanded=False):
    context_window  = st.slider("문맥 참고 세그먼트 수", 1, 5, 3, 1)

# ─────────────────────────────────────────────
# 메인 — 파일 업로드
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader("검사할 동영상 파일(MP4)을 업로드하세요.", type=["mp4"])

if uploaded_file is not None:
    video_path = "temp_video.mp4"
    audio_path = "temp_audio.mp3"

    st.video(uploaded_file)

    if st.button("🚀 맞춤법 검사 시작", type="primary", use_container_width=True):

        with st.spinner("파일 준비 중..."):
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        audio_results  = []
        screen_results = []

        # ── 화면 텍스트 검사 ─────────────────────────
        if check_screen:
            st.divider()
            prog = st.progress(0, text="🖼️ 핵심 프레임 추출 중...")
            with st.spinner("OpenCV 프레임 추출 및 전처리 중..."):
                frames = extract_and_filter_frames(video_path, sample_rate, diff_threshold)

            if frames:
                st.caption(f"총 {len(frames)}장 핵심 화면 감지 → Vision AI 분석 시작")
                prog.progress(40, text="👁️ GPT-4o Vision 판독 중...")
                with st.spinner("화면 속 텍스트 OCR 및 교정 중..."):
                    screen_results = spell_check_frames(frames, api_key, batch_size=batch_size)
                prog.progress(100, text="✅ 화면 검사 완료!")
            else:
                st.warning("분석할 화면이 감지되지 않았습니다.")
                prog.progress(100)

        # ── 음성 대본 검사 ────────────────────────────
        if check_audio:
            st.divider()
            prog2 = st.progress(0, text="🎧 오디오 분리 중...")
            with st.spinner("오디오 추출 중..."):
                ok = extract_audio(video_path, audio_path)

            if ok:
                prog2.progress(25, text="🗣️ Whisper STT 변환 중...")
                with st.spinner("음성 → 텍스트 변환 중 (한국어 강제)..."):
                    try:
                        segments = transcribe_audio(audio_path, api_key)
                    except Exception as e:
                        st.error(f"STT 오류: {e}")
                        segments = []

                if segments:
                    prog2.progress(60, text="🔍 GPT-4o 교정 중...")
                    with st.spinner("대본 맞춤법 교정 중..."):
                        audio_results = spell_check_segments(segments, api_key, context_window=context_window)
                    prog2.progress(100, text="✅ 음성 검사 완료!")
                else:
                    st.warning("유의미한 음성이 감지되지 않았습니다.")
                    prog2.progress(100)
            else:
                st.error("오디오 추출 실패")
                prog2.progress(100)

        # ─────────────────────────────────────────────
        # 결과 출력 — 탭으로 분리
        # ─────────────────────────────────────────────
        st.divider()
        st.subheader("📋 교정 결과")

        # 요약 지표
        c1, c2, c3 = st.columns(3)
        c1.metric("전체 교정 건수", f"{len(audio_results) + len(screen_results)}건")
        c2.metric("🎧 음성 대본", f"{len(audio_results)}건")
        c3.metric("🖼️ 화면 텍스트", f"{len(screen_results)}건")

        st.markdown("")

        # 탭 구성
        tab_labels = []
        if check_audio:
            tab_labels.append(f"🎧 음성 대본 ({len(audio_results)}건)")
        if check_screen:
            tab_labels.append(f"🖼️ 화면 텍스트 ({len(screen_results)}건)")

        tabs = st.tabs(tab_labels)
        tab_idx = 0

        if check_audio:
            with tabs[tab_idx]:
                st.markdown("##### 음성에서 발견된 맞춤법·오타 오류")
                render_result_cards(audio_results, badge_type="audio")
                if audio_results:
                    st.download_button(
                        "📥 음성 교정 결과 CSV 다운로드",
                        data=make_csv(audio_results),
                        file_name="음성_교정_결과.csv",
                        mime="text/csv"
                    )
            tab_idx += 1

        if check_screen:
            with tabs[tab_idx]:
                st.markdown("##### 화면 텍스트에서 발견된 맞춤법·오타 오류")
                render_result_cards(screen_results, badge_type="screen")
                if screen_results:
                    st.download_button(
                        "📥 화면 교정 결과 CSV 다운로드",
                        data=make_csv(screen_results),
                        file_name="화면_교정_결과.csv",
                        mime="text/csv"
                    )

        # 임시 파일 정리
        for path in [video_path, audio_path]:
            if os.path.exists(path):
                os.remove(path)
