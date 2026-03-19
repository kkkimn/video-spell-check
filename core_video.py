import os
import json
import base64
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from openai import OpenAI

def extract_audio(video_path, audio_path):
    """MP4 영상에서 오디오(.mp3)를 추출하여 저장합니다."""
    try:
        video = VideoFileClip(video_path)
        # 오디오만 추출하여 파일로 저장
        video.audio.write_audiofile(audio_path, logger=None)
        video.close()
        return True
    except Exception as e:
        print(f"오디오 추출 오류: {e}")
        return False

def format_timestamp(seconds):
    """초(float) 단위의 시간을 [00:00:00] 또는 [00:00] 형태의 문자열로 변환합니다."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"
    else:
        return f"[{minutes:02d}:{secs:02d}]"

def transcribe_audio(audio_path, api_key):
    """Whisper API를 사용하여 오디오 파일에서 타임스탬프와 함께 텍스트를 추출합니다."""
    client = OpenAI(api_key=api_key)
    
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    return transcription.segments

def spell_check_segments(segments, api_key):
    """각 세그먼트의 텍스트를 모아 GPT로 맞춤법 검사를 수행하고 결과를 매핑합니다."""
    client = OpenAI(api_key=api_key)
    results = []
    
    batch_text = ""
    for idx, seg in enumerate(segments):
        start_time = format_timestamp(seg.start)
        batch_text += f"{start_time} (ID:{idx}): {seg.text.strip()}\n"
    
    if not batch_text:
        return []

    system_prompt = (
        "아래 텍스트는 동영상에서 추출한 음성의 STT(Speech-to-Text) 결과입니다. (각 줄마다 ID와 시간, 텍스트가 있습니다.)\n"
        "전체적인 문맥을 바탕으로 오타, 맞춤법, 띄어쓰기 오류를 꼼꼼하게 찾으세요.\n"
        "응답 시 오려내지 말고 **원본의 문장(전체)**과 **교정된 문장(전체)**을 반환하되, 반드시 수정이 발생한 지점을 `<red>단어</red>` 모양으로 감싸서 강조하세요.\n"
        "만약 오류가 없다면 해당 문장은 반환하지 않습니다.\n\n"
        "반드시 아래 JSON 형태의 배열로 응답하세요. 백틱(`) 없이 순수 JSON만 출력하세요.\n"
        "[\n"
        "  {\n"
        "    \"id\": 0,\n"
        "    \"original\": \"오늘은 날씨가 참 <red>조습니다</red> 전체문장\",\n"
        "    \"corrected\": \"오늘은 날씨가 참 <red>좋습니다</red> 전체문장\"\n"
        "  },\n"
        "  ...\n"
        "]"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": batch_text}
            ],
            temperature=0.2
        )
        
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
            
        corrections = json.loads(content)
        
        for item in corrections:
            idx = item.get("id")
            if idx is not None and 0 <= idx < len(segments):
                seg = segments[idx]
                start_time = format_timestamp(seg.start)
                orig = item.get("original", "").strip()
                corr = item.get("corrected", "").strip()
                
                # <red> 태그를 HTML 빨간색 표기 태그로 변환
                orig_html = orig.replace("<red>", "<span style='color:red; font-weight:bold;'>").replace("</red>", "</span>")
                corr_html = corr.replace("<red>", "<span style='color:red; font-weight:bold;'>").replace("</red>", "</span>")
                
                if orig and corr and (orig != corr):
                    results.append({
                        "시간": start_time,
                        "수정 전": orig_html,
                        "수정 후": corr_html
                    })
        
        return results
    except Exception as e:
        print(f"음성 맞춤법 검사 오류: {e}")
        return []

def encode_image(image):
    """OpenCV 이미지를 Base64 문자열로 인코딩합니다 (해상도를 유지하여 OCR 인식률 극대화)."""
    # 전송용 이미지는 1920 해상도(FHD)까지만 유지 (너무 크면 자원 낭비, 작으면 인식 불가)
    h, w = image.shape[:2]
    if max(w, h) > 1920:
        scale = 1920 / max(w, h)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
        
    # 이미지 품질 95로 상향 (글자 뭉개짐 방지)
    _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return base64.b64encode(buffer).decode('utf-8')

def extract_and_filter_frames(video_path, sample_rate=1.0, diff_threshold=15.0):
    """
    영상에서 sample_rate(초) 간격으로 프레임을 추출하고,
    이전 캡처본과 비교하여 차이가 작은(화면이 동일한) 경우는 스킵합니다.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    
    frame_interval = int(fps * sample_rate)
    unique_frames = []
    prev_frame_gray = None
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # 두 프레임 간 유사도 비교를 위해 작게 리사이즈된 흑백 이미지만 생성
            small_frame = cv2.resize(frame, (320, 180))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            is_unique = True
            if prev_frame_gray is not None:
                # 픽셀 간 압도적 차이 확인 (MAE 계산)
                diff = cv2.absdiff(gray, prev_frame_gray)
                mean_diff = np.mean(diff)
                
                # 차이가 threshold 이하라면 화면이 안 바뀐 것으로 보고 스킵
                if mean_diff < diff_threshold:
                    is_unique = False
                    
            if is_unique:
                current_time_sec = frame_count / fps
                unique_frames.append({
                    "time": current_time_sec,
                    "time_str": format_timestamp(current_time_sec),
                    "base64": encode_image(frame)
                })
                prev_frame_gray = gray
                
        frame_count += 1
        
    cap.release()
    return unique_frames

def spell_check_frames(frames, api_key):
    """
    고유 프레임들의 Base64 이미지 목록을 GPT-4o Vision에 전달하여
    텍스트 추출(OCR) 및 맞춤법 검사를 수행합니다.
    """
    if not frames:
        return []
        
    client = OpenAI(api_key=api_key)
    results = []
    
    # High Detail 모드는 더 많은 토큰과 네트워크를 소모하므로 API 에러 방지를 위해 3장씩 묶음 전송
    batch_size = 3
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        
        content_items = [
            {
                "type": "text", 
                "text": (
                    "당신은 매우 꼼꼼한 초정밀 OCR(광학 문자 인식) 및 국어 맞춤법 교정 전문가입니다. "
                    "제공된 이미지(비디오 캡처 화면)들을 아주 세밀하게 관찰하세요.\n"
                    "1. (매우 중요) 화면에 존재하는 모든 한국어 텍스트를 토씨 하나 빠뜨리지 말고 먼저 `transcription` 필드에 전부 기록하세요.\n"
                    "2. 추출한 전체 텍스트(`transcription`) 중에서 명백한 오타, 오류가 발생한 문장을 찾아내어 `corrections` 배열에 담으세요.\n"
                    "3. `corrections` 배열 안에서 **문장 전체(문맥을 알 수 있는 충분한 길이)**를 돌려주되, 오류가 발생한 부분만 반드시 `<red>단어</red>`로 감싸서 강조하세요.\n"
                    "4. 화면에 텍스트가 없거나 수정할 게 완벽히 없으면 `corrections` 배열을 빈칸(`[]`)으로 두세요.\n\n"
                    "반드시 아래 JSON 배열 형식으로만 응답하세요. 다른 코멘트 없이 순수 JSON만 반환해야 합니다.\n"
                    "[\n"
                    "  {\n"
                    "    \"id\": 0,\n"
                    "    \"transcription\": \"화면에 적힌 전체 텍스트 내용들을 모두 적음(문맥 파악 용도)\",\n"
                    "    \"corrections\": [\n"
                    "      {\n"
                    "        \"original\": \"오늘은 날씨가 참 <red>조습니다</red> 전체문장\",\n"
                    "        \"corrected\": \"오늘은 날씨가 참 <red>좋습니다</red> 전체문장\"\n"
                    "      }\n"
                    "    ]\n"
                    "  },\n"
                    "  ...\n"
                    "]"
                )
            }
        ]
        
        for idx, f in enumerate(batch_frames):
            content_items.append({"type": "text", "text": f"[{idx}번 이미지]"})
            content_items.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{f['base64']}",
                    "detail": "high"
                }
            })
            
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content_items}],
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
                
            corrections = json.loads(content)
            
            for item in corrections:
                f_idx = item.get("id")
                if f_idx is not None and 0 <= f_idx < len(batch_frames):
                    
                    for corr_item in item.get("corrections", []):
                        orig = corr_item.get("original", "").strip()
                        corr = corr_item.get("corrected", "").strip()
                        
                        orig_html = orig.replace("<red>", "<span style='color:red; font-weight:bold;'>").replace("</red>", "</span>")
                        corr_html = corr.replace("<red>", "<span style='color:red; font-weight:bold;'>").replace("</red>", "</span>")
                        
                        if orig and corr and (orig != corr):
                            results.append({
                                "구분": "화면 텍스트",
                                "시간": batch_frames[f_idx]["time_str"],
                                "수정 전": orig_html,
                                "수정 후": corr_html
                            })
        except Exception as e:
            print(f"화면 맞춤법 검사 오류: {e}")
            
    return results
