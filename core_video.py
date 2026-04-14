import os
import json
import time
import base64
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from openai import OpenAI

# ─────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────

def extract_audio(video_path, audio_path):
    """MP4 영상에서 오디오(.mp3)를 추출하여 저장합니다."""
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, logger=None)
        video.close()
        return True
    except Exception as e:
        print(f"오디오 추출 오류: {e}")
        return False


def format_timestamp(seconds):
    """초(float) 단위의 시간을 [HH:MM:SS] 또는 [MM:SS] 형태로 변환합니다."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"
    return f"[{minutes:02d}:{secs:02d}]"


def call_with_retry(fn, retries=3, delay=5):
    """API 호출 실패 시 최대 retries 횟수까지 재시도합니다."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            print(f"API 호출 오류 (시도 {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    raise RuntimeError(f"API 호출이 {retries}회 모두 실패했습니다.")


# ─────────────────────────────────────────────
# 음성(STT) 처리
# ─────────────────────────────────────────────

def transcribe_audio(audio_path, api_key):
    """
    Whisper API로 오디오를 변환합니다.
    - language="ko" 강제 지정으로 한국어 인식 정확도 향상
    - segment + word 단위 타임스탬프를 모두 요청
    """
    client = OpenAI(api_key=api_key)

    def _call():
        with open(audio_path, "rb") as audio_file:
            return client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="ko",                          # ★ 한국어 강제 지정
                response_format="verbose_json",
                timestamp_granularities=["segment", "word"]  # ★ word 단위 추가
            )

    transcription = call_with_retry(_call)
    return transcription.segments


# ─────────────────────────────────────────────
# 음성 세그먼트 맞춤법 검사
# ─────────────────────────────────────────────

# 한국어 주요 맞춤법 규칙 — 프롬프트에 삽입하여 모델 집중도 향상
_KO_SPELL_RULES = """
주요 검사 항목 (특히 집중):
1. 사이시옷: 숫자(숫자), 셋방(셋방) 등 합성어 표기
2. 된소리·거센소리 혼동: '깨끗이' vs '깨끗히', '않다' vs '안다'
3. 어미 혼동: '-이에요'/'-이어요', '-데'/'-대', '-든지'/'-던지'
4. 조사·의존명사 띄어쓰기: '것', '수', '때', '만큼', '뿐' 등
5. 피동·사동 표현 오용: '되어지다', '만들어지다' 등 이중피동
6. 외래어 표기: '컨텐츠(→콘텐츠)', '메세지(→메시지)' 등
7. 공식 명칭·고유어 오기: 틀린 한자어·혼용 표기
8. 불필요한 중복 표현: '미리 예방', '과반수 이상' 등
9. 단위·숫자 표기: '1개월' 단위 붙여쓰기 여부
10. 어순 및 비문(문장 구조 오류)
"""

_SPELL_SYSTEM_PROMPT = f"""당신은 대한민국 방송 표준어 규정과 한글 맞춤법을 완벽히 숙지한 수석 교열 전문가입니다.
아래 텍스트는 동영상 음성을 STT로 변환한 결과입니다. (각 줄에 ID, 시간, 텍스트 포함)

{_KO_SPELL_RULES}

【작업 지시】
- 전체 문맥을 반드시 파악한 뒤 교정하세요. (앞뒤 문장 참고)
- 오류가 확실할 때만 교정하세요. 애매한 경우 원문 유지.
- 수정된 단어·어절만 <red>단어</red>로 감싸세요.
- 원문 문장 전체를 반환하되 수정 부분만 태그 처리.
- 오류 없는 세그먼트는 결과에서 제외(빈 배열 반환 불필요).

응답 형식 — 백틱 없이 순수 JSON 배열만 출력:
[
  {{
    "id": 0,
    "original": "전체 원문 <red>오류단어</red> 나머지",
    "corrected": "전체 교정문 <red>올바른단어</red> 나머지",
    "reason": "교정 이유 간략 설명"
  }}
]
오류가 없으면: []
"""


def spell_check_segments(segments, api_key, context_window=3):
    """
    세그먼트 배치를 GPT-4o로 맞춤법 검사합니다.
    context_window: 앞뒤로 포함할 세그먼트 수 (문맥 제공용, 교정 대상 아님)
    """
    client = OpenAI(api_key=api_key)
    results = []

    # 전체를 하나의 배치로 전송하되, context 정보도 포함
    batch_text = ""
    for idx, seg in enumerate(segments):
        start_time = format_timestamp(seg.start)
        batch_text += f"{start_time} (ID:{idx}): {seg.text.strip()}\n"

    if not batch_text:
        return []

    def _call():
        return client.chat.completions.create(
            model="gpt-4o",             # ★ mini → 4o 업그레이드
            messages=[
                {"role": "system", "content": _SPELL_SYSTEM_PROMPT},
                {"role": "user", "content": batch_text}
            ],
            temperature=0.1,            # ★ 낮춰서 일관성 극대화
            response_format={"type": "json_object"} if False else {"type": "text"}
        )

    try:
        response = call_with_retry(_call)
        content = response.choices[0].message.content.strip()
        content = _strip_json_fences(content)
        corrections = json.loads(content)

        # 결과가 dict 형태({...})로 감싸인 경우 내부 배열 추출
        if isinstance(corrections, dict):
            corrections = next(iter(corrections.values()), [])

        for item in corrections:
            idx = item.get("id")
            if idx is not None and 0 <= idx < len(segments):
                seg = segments[idx]
                start_time = format_timestamp(seg.start)
                orig = item.get("original", "").strip()
                corr = item.get("corrected", "").strip()
                reason = item.get("reason", "")

                orig_html = _red_to_html(orig)
                corr_html = _red_to_html(corr)

                if orig and corr and orig != corr:
                    results.append({
                        "구분": "음성 대본",
                        "시간": start_time,
                        "수정 전": orig_html,
                        "수정 후": corr_html,
                        "교정 사유": reason
                    })

    except Exception as e:
        print(f"음성 맞춤법 검사 오류: {e}")

    return results


# ─────────────────────────────────────────────
# 화면 프레임 처리
# ─────────────────────────────────────────────

def _preprocess_for_ocr(image):
    """
    OCR 전 이미지 전처리:
    1) FHD(1920px) 이하로 리사이즈
    2) CLAHE 대비 강화 (글자 경계 선명화)
    3) 언샤프 마스크 샤프닝
    """
    h, w = image.shape[:2]
    if max(w, h) > 1920:
        scale = 1920 / max(w, h)
        image = cv2.resize(image, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_LANCZOS4)

    # CLAHE 적용 (LAB 색공간에서 L 채널만)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 언샤프 마스크 샤프닝
    gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
    image = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

    return image


def encode_image(image):
    """전처리된 이미지를 JPEG Base64로 인코딩합니다."""
    image = _preprocess_for_ocr(image)
    _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return base64.b64encode(buffer).decode('utf-8')


def extract_and_filter_frames(video_path, sample_rate=1.0, diff_threshold=15.0):
    """
    영상에서 sample_rate(초) 간격으로 프레임을 추출하고
    이전 캡처본과 차이가 작은(유사) 프레임은 제거합니다.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(fps * sample_rate))

    unique_frames = []
    prev_gray = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            small = cv2.resize(frame, (320, 180))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            is_unique = True
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                if np.mean(diff) < diff_threshold:
                    is_unique = False

            if is_unique:
                current_sec = frame_count / fps
                unique_frames.append({
                    "time": current_sec,
                    "time_str": format_timestamp(current_sec),
                    "base64": encode_image(frame)   # ★ 전처리 적용
                })
                prev_gray = gray

        frame_count += 1

    cap.release()
    return unique_frames


# ─────────────────────────────────────────────
# 화면 텍스트 맞춤법 검사
# ─────────────────────────────────────────────

_OCR_SPELL_PROMPT = f"""당신은 초정밀 한국어 OCR 및 맞춤법 교정 전문가입니다.
제공된 비디오 캡처 이미지들을 면밀히 분석하세요.

【작업 순서】
1. 각 이미지에서 보이는 **모든 한국어 텍스트**를 빠짐없이 `transcription`에 기록하세요.
   - 자막, 제목, 자료 화면, UI 라벨, 워터마크 등 위치 무관 전부 포함
   - 글자가 작거나 흐릿해도 최선을 다해 판독하세요.
2. 추출한 텍스트에서 맞춤법·오타 오류를 찾아 `corrections`에 담으세요.

{_KO_SPELL_RULES}

【교정 규칙】
- 문장 전체 를 반환하되, 오류 부분만 <red>단어</red>로 감싸세요.
- 확실한 오류만 교정 (불확실 → 원문 유지).
- 이미지에 텍스트가 없거나 오류 없으면 `corrections: []`.

응답 형식 — 백틱 없이 순수 JSON 배열만 출력:
[
  {{
    "id": 0,
    "transcription": "화면에서 읽은 모든 텍스트(줄바꿈 포함)",
    "corrections": [
      {{
        "original": "전체 원문 <red>오류단어</red> 나머지",
        "corrected": "전체 교정문 <red>올바른단어</red> 나머지",
        "reason": "교정 이유"
      }}
    ]
  }}
]
"""


def spell_check_frames(frames, api_key, batch_size=3):
    """
    추출된 프레임 이미지를 GPT-4o Vision으로 OCR + 맞춤법 검사합니다.
    batch_size: 한 번에 전송할 이미지 수 (토큰 한도 고려)
    """
    if not frames:
        return []

    client = OpenAI(api_key=api_key)
    results = []

    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]

        content_items = [{"type": "text", "text": _OCR_SPELL_PROMPT}]

        for idx, f in enumerate(batch):
            content_items.append({"type": "text", "text": f"[{idx}번 이미지 — {f['time_str']}]"})
            content_items.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{f['base64']}",
                    "detail": "high"    # ★ 고해상도 OCR 유지
                }
            })

        def _call(items=content_items):
            return client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": items}],
                temperature=0.1         # ★ 낮춰서 일관성 극대화
            )

        try:
            response = call_with_retry(_call)
            content = _strip_json_fences(response.choices[0].message.content.strip())
            corrections = json.loads(content)

            if isinstance(corrections, dict):
                corrections = next(iter(corrections.values()), [])

            for item in corrections:
                f_idx = item.get("id")
                if f_idx is None or not (0 <= f_idx < len(batch)):
                    continue
                for corr_item in item.get("corrections", []):
                    orig = corr_item.get("original", "").strip()
                    corr = corr_item.get("corrected", "").strip()
                    reason = corr_item.get("reason", "")

                    orig_html = _red_to_html(orig)
                    corr_html = _red_to_html(corr)

                    if orig and corr and orig != corr:
                        results.append({
                            "구분": "화면 텍스트",
                            "시간": batch[f_idx]["time_str"],
                            "수정 전": orig_html,
                            "수정 후": corr_html,
                            "교정 사유": reason
                        })

        except Exception as e:
            print(f"화면 맞춤법 검사 오류 (배치 {i // batch_size + 1}): {e}")

    return results


# ─────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────

def _strip_json_fences(text):
    """마크다운 코드 펜스(```json ... ```)를 제거합니다."""
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _red_to_html(text):
    """<red>단어</red> 태그를 인라인 HTML 강조 스타일로 변환합니다."""
    return (text
            .replace("<red>", "<span style='color:red; font-weight:bold;'>")
            .replace("</red>", "</span>"))
