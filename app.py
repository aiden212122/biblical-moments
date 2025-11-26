import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image
import io

# --- 1. 界面深度美化与配置 ---
st.set_page_config(page_title="圣经时光照相馆", page_icon="✝️", layout="centered")

# 自定义 CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #FDFBF7;
        background-image: radial-gradient(#F3E5AB 1px, transparent 1px);
        background-size: 20px 20px;
    }
    h1 {
        color: #78350F;
        font-family: 'Georgia', serif;
        text-align: center;
        padding-bottom: 20px;
    }
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #B45309 0%, #D97706 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 15px 25px;
        font-size: 18px;
        font-weight: 600;
        box-shadow: 0 4px 14px 0 rgba(180, 83, 9, 0.39);
        transition: all 0.2s ease-in-out;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(180, 83, 9, 0.23);
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. 核心配置 ---
api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    with st.sidebar:
        st.header("⚙️ 设置")
        api_key = st.text_input("请输入 Google API Key", type="password")
        st.markdown("[获取免费 Key](https://aistudio.google.com/)")

if not api_key:
    st.warning("请先配置 API Key 才能启动")
    st.stop()

genai.configure(api_key=api_key)

# 彻底解除安全拦截
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# --- 3. UI 布局 ---
st.markdown("<h1 style='text-align: center;'>✝️ 圣经时光照相馆</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #92400E; margin-bottom: 30px;'>穿越千年，与信心伟人同框 | Powered by Gemini AI</p>", unsafe_allow_html=True)

with st.container():
    st.markdown("### 📸 第一步：上传你的照片")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    user_image = None
    if uploaded_file:
        user_image = Image.open(uploaded_file)
        col_img1, col_img2, col_img3 = st.columns([1,2,1])
        with col_img2:
            st.image(user_image, caption="你的照片已就绪", use_column_width=True)

    st.markdown("---")

    st.markdown("### 🎨 第二步：定制合照细节")
    col1, col2 = st.columns(2)
    with col1:
        character_input = st.text_input("🙏 你想见谁？(支持中文)", value="耶稣", placeholder="例如：摩西、大卫")
        clothing_style = st.selectbox("👕 你的着装风格", [
            "保持历史真实感 (穿古希伯来长袍)",
            "现代休闲 (T恤/卫衣)",
            "现代正装 (西装/礼服)",
            "保留我照片里的衣服"
        ])
    with col2:
        art_style = st.selectbox("✨ 画面艺术风格", [
            "史诗电影感 (Cinematic Epic)",
            "文艺复兴油画 (Renaissance Oil)",
            "高清写实 (Photorealistic)",
            "复古胶片 (Vintage Film)",
            "3D 动画风格 (Pixar Style)",
            "素描手绘 (Pencil Sketch)",
            "彩色玻璃窗风格 (Stained Glass)"
        ])
        aspect_ratio = st.selectbox("📐 图片比例", [
            "3:4 (竖屏 - 适合壁纸)",
            "1:1 (正方形 - 适合头像)",
            "16:9 (横屏 - 电影宽幅)"
        ])

# --- 4. 提示词构建 ---
def build_prompt(char, cloth, style, ratio):
    ratio_prompt = ""
    if "16:9" in ratio: ratio_prompt = "Wide angle cinematic shot, 16:9 aspect ratio."
    elif "3:4" in ratio: ratio_prompt = "Portrait shot, vertical composition."
    
    cloth_prompt = "wearing generic clothes"
    if "历史" in cloth: cloth_prompt = "wearing historically accurate ancient Hebrew robes"
    elif "休闲" in cloth: cloth_prompt = "wearing modern casual clothes (t-shirt/jeans)"
    elif "正装" in cloth: cloth_prompt = "wearing a formal modern suit"
    elif "保留" in cloth: cloth_prompt = "wearing exactly the same clothes as in the input image"

    style_prompt = "Photorealistic"
    if "电影" in style: style_prompt = "Cinematic lighting, 8k resolution, epic atmosphere"
    elif "油画" in style: style_prompt = "Classic Renaissance oil painting style, visible brushstrokes"
    elif "胶片" in style: style_prompt = "Vintage Kodak film look, slight grain"
    elif "动画" in style: style_prompt = "Pixar style 3D render, cute"
    elif "玻璃" in style: style_prompt = "Stained glass window art style"

    full_prompt = f"""
    Role: Expert Biblical Photographer.
    Task: Create a collaborative photo (Two-Shot) based on the Input Image.
    
    Subject A (The User): The person from the [Input Image].
    - Attire: {cloth_prompt}.
    - Face: Preserve the face of the user carefully.
    
    Subject B (Biblical Figure): {char}.
    - Appearance: MUST be historically accurate to the Bible era (Ancient Middle Eastern descent). 
    - NO westernized/modernized depictions.
    
    Setting: Authentic biblical landscape matching Subject B.
    Style: {style_prompt}. {ratio_prompt}
    Output format: IMAGE ONLY.
    """
    return full_prompt

# --- 5. 生成执行 (已修改超时时间) ---
if st.button("✨ 开始祈祷并生成合照 ✨"):
    if not user_image:
        st.error("⚠️ 请先上传一张你的照片")
    else:
        status = st.status("🌟 正在连接时空...", expanded=True)
        
        try:
            status.write("正在构建场景描述...")
            final_prompt = build_prompt(character_input, clothing_style, art_style, aspect_ratio)
            
            status.write("正在唤醒 AI 画师 (Gemini 3 Pro)...")
            MODEL_ID = 'gemini-3-pro-image-preview'
            model = genai.GenerativeModel(MODEL_ID)
            
            # 这里修改了提示语，让用户心里有底
            status.write("正在精细绘制 (高清模型较慢，请耐心等待 1-4 分钟)...")
            
            # 发送请求 (关键修改：timeout=240)
            response = model.generate_content(
                [final_prompt, user_image],
                safety_settings=safety_settings,
                request_options={"timeout": 240}  # <--- 这里改成了 240 秒 (4分钟)
            )
            
            status.write("正在显影...")
            
            image_generated = False
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        status.update(label="✅ 生成成功！", state="complete", expanded=False)
                        
                        img_data = part.inline_data.data
                        final_img = Image.open(io.BytesIO(img_data))
                        
                        st.image(final_img, caption=f"我和 {character_input} 的时空合影", use_column_width=True)
                        
                        buf = io.BytesIO()
                        final_img.save(buf, format="PNG")
                        st.download_button(
                            label="📥 下载高清合照",
                            data=buf.getvalue(),
                            file_name=f"bible_photo_{character_input}.png",
                            mime="image/png",
                            type="primary"
                        )
                        image_generated = True
                        break
            
            if not image_generated:
                status.update(label="⚠️ 生成未完成", state="error")
                st.error("未接收到图片数据。")
                if response.text:
                    st.info(f"AI 反馈: {response.text}")

        except Exception as e:
            status.update(label="❌ 发生错误", state="error")
            st.error(f"错误详情: {e}")
            # 增加一个友好的超时提示
            if "408" in str(e) or "deadline" in str(e).lower() or "timeout" in str(e).lower():
                st.warning("请求超时了。Gemini 3 Pro 可能正在忙碌，请稍等几分钟再试，或者尝试刷新页面。")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #aaa; font-size: 12px;'>此应用仅供娱乐与信仰纪念，生成的圣经人物形象为 AI 艺术想象。</div>", unsafe_allow_html=True)
