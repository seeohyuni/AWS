import streamlit as st
import requests
from PIL import Image
import io
from streamlit_cropper import st_cropper
import base64
import json
import os

# --- 설정 및 경로 ---
LAMBDA_URL = "https://oub5iny2k5lxutxy736drnexoi0ozwyj.lambda-url.us-west-2.on.aws/"
# 스크립트 위치에 상관없이 동일한 위치에 저장되도록 절대 경로 사용
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")

st.set_page_config(page_title="SAM2 프리미엄 스튜디오", layout="wide")

# --- 데이터 저장/불러오기 기능 ---
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(history_item):
    current_history = load_history()
    current_history.insert(0, history_item) # 최신 항목이 위로
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(current_history, f, ensure_ascii=False, indent=2)

def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

# --- 프리미엄 스타일 CSS ---
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .floating-card {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }
    .main-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        background-color: #4f46e5;
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4338ca;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        transform: translateY(-2px);
    }
    img {
        max-width: 100%;
        border-radius: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 헤더 ---
st.markdown("<h1 class='main-title'>✨ AI가 누끼따주는 서비스</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;'>AI를 이용한 가장 정교하고 빠른 배경 제거 서비스</p>", unsafe_allow_html=True)

# --- 사이드바 설정 ---
with st.sidebar:
    st.header("📸 워크스페이스")
    uploaded_file = st.file_uploader("이미지를 업로드하여 시작하세요", type=["jpg", "png", "jpeg"])
    
    st.write("---")
    if st.button("🗑️ 모든 기록 삭제"):
        clear_history()
        st.rerun()

# --- 메인 레이아웃 및 로직 ---
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # 좌우 2컬럼 레이아웃
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("<div class='floating-card'>", unsafe_allow_html=True)
        st.subheader("🎯 영역 지정")
        st.info("누끼를 딸 물체가 사각형 안에 모두 들어오도록 조절해주세요.")
        
        # 크로퍼 도구 (박스 기반)
        box = st_cropper(image, realtime_update=True, box_color='#4f46e5', aspect_ratio=None, return_type='box')
        
        if box:
            x1, y1 = box['left'], box['top']
            x2, y2 = x1 + box['width'], y1 + box['height']
            
            if st.button("🚀 배경 제거 시작"):
                with st.spinner("AI가 배경을 분석 중입니다... 잠시만 기다려주세요."):
                    try:
                        # 이미지 전송 준비
                        buf = io.BytesIO()
                        image.save(buf, format="PNG")
                        image_bytes = buf.getvalue()
                        encoded_string = base64.b64encode(image_bytes).decode('utf-8')
                        
                        # 전송용 페이로드
                        payload = {
                            "image": encoded_string,
                            "box_x1": x1, "box_y1": y1, "box_x2": x2, "box_y2": y2
                        }
                        response = requests.post(LAMBDA_URL, json=payload, timeout=120)
                        
                        if response.status_code == 200:
                            data = response.json()
                            image_url = data.get("image_url")
                            
                            if image_url:
                                # JSON 파일에 영구 저장
                                save_history({"url": image_url})
                                st.rerun() # 새로고침하여 결과 업데이트
                        else:
                            st.error(f"서버 오류: {response.status_code}")
                    except Exception as e:
                        st.error(f"요청 실패: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='floating-card'>", unsafe_allow_html=True)
        st.subheader("🖼️ 최근 작업 결과")
        
        # 최신 결과 가져오기
        history_data = load_history()
        if history_data:
            latest_url = history_data[0]['url']
            st.image(latest_url, caption="배경이 제거된 결과물", width="stretch")
            
            # 다운로드 버튼
            try:
                r = requests.get(latest_url)
                st.download_button(
                    label="📥 투명 배경 사진 다운로드",
                    data=r.content,
                    file_name="cutout_result.png",
                    mime="image/png"
                )
            except:
                st.warning("다운로드 준비 중 오류가 발생했습니다.")
        else:
            st.write("---")
            st.write("배경 제거를 완료하면 여기에 결과가 나타납니다.")
        st.markdown("</div>", unsafe_allow_html=True)

else:
    # 대기 화면 (사진 업로드 전)
    st.markdown("""
        <div style='text-align: center; padding: 5rem; color: #94a3b8;'>
            <h3>👋 안녕하세요!</h3>
            <p>좌측 메뉴에서 사진을 선택하여 누끼 작업을 시작해 보세요.</p>
        </div>
    """, unsafe_allow_html=True)

# --- 작업 내역 갤러리 (항상 표시: F5 눌러도 유지됨) ---
history_all = load_history()
if len(history_all) > 0:
    st.write("---")
    st.header("🕰️ 이전 작업 내역")
    st.write("지금까지 작업한 모든 결과물입니다.")
    
    n_cols = 4
    cols = st.columns(n_cols)
    
    for idx, item in enumerate(history_all):
        with cols[idx % n_cols]:
            st.image(item['url'], width="stretch")
            st.caption(f"작업 번호 #{len(history_all) - idx}")
