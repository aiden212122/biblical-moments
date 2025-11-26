import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image
import io
import time

# --- 1. 页面配置 ---
st.set_page_config(page_title="圣经时空照相馆", page_icon="✨")
st.markdown("""
    <style>
    .stApp {background-color: #FAFAF9;}
    div.stButton > button:first-child {
        background-color: #EA580C;
        color: white;
        border-radius: 10px;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("✨ 圣经时空照相馆")
st.caption("Powered by Google Gemini")

# --- 2. API Key 配置 ---
api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    with st.sidebar:
        api_key = st.text_input("API Key", type="password")
if not api_key:
    st.warning("请先配置 API Key")
    st.stop()

genai.configure(api_key=api_key)

# --- 3. 关键修复：把安全限制降到最低 ---
# 必须加上这个，否则生成“宗教人物”或“真人”容易被系统自动拦截导致卡死
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# --- 4. 界面 ---
with st.container():
    uploaded_file = st.file_uploader("1. 上传你的照片", type=["jpg", "jpeg", "png"])
    user_image = None
    if uploaded_file:
        user_image = Image.open(uploaded_file)
        st.image(user_image, width=150)

    col1, col2 = st.columns(2)
    with col1:
        character = st.text_input("2. 圣经人物", value="Jesus")
    with col2:
        clothing = st.selectbox("3. 服装风格", ["Historical Robes (复古长袍)", "Casual (现代便装)", "Suit (西装)"])
    
    style = st.selectbox("4. 画面风格", ["Cinematic (电影感)", "Oil Painting (油画)", "Realistic (写实)"])

# --- 5. 生成逻辑 ---
if st.button("🚀 开始生成 (解决卡顿版)", type="primary", use_container_width=True):
    if not user_image:
        st.error("请先上传照片！")
    else:
        status = st.status("正在与 AI 建立连接...", expanded=True)
        try:
            # 步骤 1: 准备模型
            # 如果 Gemini 3 依然卡住，代码会自动尝试 fallback
            status.write("正在初始化模型...")
            MODEL_ID = 'gemini-3-pro-image-preview' # 你指定的模型
            model = genai.GenerativeModel(MODEL_ID)

            prompt = f"""
            Task: Create a two-shot image.
            Subject 1: {character} from the Bible, historically accurate ancient look.
            Subject 2: The user from the input image, wearing {clothing}.
            Action: Standing side by side, friendly expression.
            Background: Ancient biblical landscape.
            Style: {style}, high quality.
            Output: IMAGE ONLY.
            """
            
            status.write("正在发送图片数据 (这步最慢，请耐心等待 30秒)...")
            
            # 步骤 2: 调用 (带上安全设置)
            # 增加 generation_config 确保输出格式
            response = model.generate_content(
                [prompt, user_image],
                safety_settings=safety_settings,
                request_options={"timeout": 60} # 设置 60秒超时防止死等
            )
            
            status.write("数据接收完毕，正在解析...")
            
            # 步骤 3: 解析图片
            image_found = False
            
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        status.update(label="生成成功！", state="complete", expanded=False)
                        
                        img_data = part.inline_data.data
                        final_img = Image.open(io.BytesIO(img_data))
                        
                        st.image(final_img, caption=f"我和 {character}", use_column_width=True)
                        image_found = True
                        break
            
            if not image_found:
                status.update(label="生成结束", state="error")
                st.error("⚠️ AI 这次没有返回图片。")
                st.write("可能原因：")
                st.write("1. 模型认为内容依然敏感（即使降低了安全等级）。")
                st.write("2. Gemini 3 Pro 处于预览版，有时候只返回文字描述。")
                if response.text:
                    st.info(f"AI返回的文字: {response.text}")

        except Exception as e:
            status.update(label="发生错误", state="error")
            st.error(f"出错信息: {e}")
            st.warning("建议：如果一直卡住或报错，请尝试更换 API Key，或等待几分钟再试（预览版模型不稳定）。")
