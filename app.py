import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account
import json
import os
import re

# 1. 页面配置
st.set_page_config(page_title="Biblical Moments - 圣经合影", page_icon="✝️", layout="centered")

# --- 2. 认证逻辑 ---
def init_vertex_ai():
    try:
        if "gcp_service_account" in st.secrets:
            raw_json_str = st.secrets["gcp_service_account"]
            try:
                service_account_info = json.loads(raw_json_str, strict=False)
            except json.JSONDecodeError:
                try:
                    fixed_str = raw_json_str.replace('\n', '\\n')
                    service_account_info = json.loads(raw_json_str, strict=False)
                except:
                    st.error("❌ Secrets 格式严重错误，无法解析。")
                    st.stop()

            credentials = service_account.Credentials.from_service_account_info(service_account_info)
            vertexai.init(project=service_account_info["project_id"], location="us-central1", credentials=credentials)
            return True
        else:
            vertexai.init(location="us-central1")
            return True
    except Exception as e:
        st.error(f"认证出错: {e}")
        return False

if not init_vertex_ai():
    st.stop()

# --- 3. 样式 ---
st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #D4AF37; color: white; border-radius: 20px; height: 50px; font-size: 18px; border: none; }
    h1 { text-align: center; font-family: 'serif'; color: #2C3E50; }
    .caption { text-align: center; color: #888; font-size: 12px; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

st.title("✝️ Biblical Moments")
st.write("上传您的照片，穿越时空与圣经人物合影。")

col1, col2 = st.columns(2)
with col1:
    bible_character = st.text_input("想合照的圣经人物", placeholder="例如：耶稣、大卫、彼得")
with col2:
    clothing_style = st.selectbox("您的服装风格", ["保持我照片里的衣服", "在这个时代的休闲装", "正式西装/礼服", "与圣经人物一样的古装", "工装/户外风格"])

art_style = st.select_slider("选择照片风格", options=["超写实摄影 (Photorealistic)", "电影质感 (Cinematic)", "油画风格 (Oil Painting)", "素描 (Sketch)"], value="超写实摄影 (Photorealistic)")

uploaded_file = st.file_uploader("上传您的自拍/半身照", type=['jpg', 'png', 'jpeg'])

# --- 4. AI 功能 (修改了这里的模型名称) ---

def get_gemini_prompt(user_image_bytes, character, clothing, style):
    # 🔴 修复点：使用具体的版本号 gemini-1.5-pro-001
    model = GenerativeModel("gemini-1.5-pro-001")
    
    image_part = Part.from_data(data=user_image_bytes, mime_type="image/jpeg")
    
    prompt_instruction = f"""
    You are an expert biblical historian and an art director.
    TASK: Analyze the person in the image (face, ethnicity, age, hair) and create an Imagen 3 prompt.
    SCENE: The user and {character} from the Bible. {character} must be historically accurate.
    ACTION: Standing side-by-side, friendly.
    CLOTHING: User wears {clothing}.
    STYLE: {style}, 8k resolution.
    OUTPUT: Just the prompt text.
    """
    response = model.generate_content([image_part, prompt_instruction])
    return response.text

def generate_image(prompt):
    # Imagen 3 模型
    model = ImageGenerationModel.from_pretrained("imagegeneration@006")
    
    images = model.generate_images(
        prompt=prompt, number_of_images=1, language="en", aspect_ratio="3:4",
        safety_filter_level="block_some", person_generation="allow_adult"
    )
    return images[0]

# --- 5. 执行逻辑 ---
if st.button("✨ 生成合照"):
    if not uploaded_file or not bible_character:
        st.warning("请先上传照片并输入人物。")
    else:
        try:
            progress = st.progress(0)
            status = st.empty()
            
            status.text("🙏 Gemini 正在观察照片...")
            img_bytes = uploaded_file.getvalue()
            # 1. Gemini
            prompt = get_gemini_prompt(img_bytes, bible_character, clothing_style, art_style)
            progress.progress(50)
            
            # 2. Imagen
            status.text(f"🎨 正在与 {bible_character} 合影...")
            result = generate_image(prompt)
            progress.progress(100)
            status.text("✨ 完成！")
            
            st.image(result._image_bytes, caption=f"与 {bible_character} 的合影", use_column_width=True)
            
            # 经文生成
            st.markdown("---")
            v_model = GenerativeModel("gemini-1.5-flash-001") # 这里也改用了具体版本
            verse = v_model.generate_content(f"给我一句关于'{bible_character}'的圣经经文(中英对照)。")
            st.info(verse.text)
            
        except Exception as e:
            st.error("生成出错")
            st.expander("错误详情").write(e)
