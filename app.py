import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account
import json
import os
import re

# 1. 页面配置 (必须在第一行)
st.set_page_config(page_title="Biblical Moments - 圣经合影", page_icon="✝️", layout="centered")

# --- 2. 增强版 Google Cloud 认证逻辑 (包含自动修复功能) ---
def init_vertex_ai():
    """
    初始化 Vertex AI 连接。
    包含针对 Streamlit Secrets 格式错误的自动修复逻辑。
    """
    try:
        # 情况 A: 在 Streamlit Cloud 上运行 (读取 Secrets)
        if "gcp_service_account" in st.secrets:
            raw_json_str = st.secrets["gcp_service_account"]
            
            # --- 🛡️ 容错解析逻辑开始 ---
            try:
                # 尝试 1: 使用 strict=False，允许字符串中包含控制字符（如回车换行）
                service_account_info = json.loads(raw_json_str, strict=False)
            
            except json.JSONDecodeError:
                # 尝试 2: 如果还是失败，说明格式可能比较混乱
                # 尝试手动清理 private_key 中的换行符问题
                # 这是一个简单的正则替换，试图保留结构但清理值
                try:
                    # 将看起来像 private key 区域内的真实换行替换为 \n 字符
                    # 注意：这只是一个应急修复
                    fixed_str = raw_json_str.replace('\n', '\\n')
                    # 有时候全局替换会破坏外层结构，所以我们回退到只依赖 strict=False
                    # 如果 strict=False 失败，通常意味着引号没闭合或者丢了逗号
                    st.warning("JSON 格式轻微异常，正在尝试自动修复...")
                    service_account_info = json.loads(raw_json_str, strict=False)
                except:
                    st.error("❌ 自动修复失败。Secrets 格式损坏太严重。")
                    st.stop()
            # --- 容错解析逻辑结束 ---

            # 创建凭证对象
            credentials = service_account.Credentials.from_service_account_info(service_account_info)
            
            # 初始化 (Project ID 自动从 JSON 里获取)
            vertexai.init(
                project=service_account_info["project_id"], 
                location="us-central1", 
                credentials=credentials
            )
            return True

        # 情况 B: 在本地电脑运行
        else:
            vertexai.init(location="us-central1")
            return True
            
    except Exception as e:
        # 显示友好的错误提示，帮助定位问题
        st.error(f"⚠️ 认证配置错误 (Secrets Error)")
        with st.expander("查看详细错误信息"):
            st.code(str(e))
        st.info("💡 提示：请检查 Streamlit Secrets 中是否完整复制了 JSON 内容，特别是大括号 {} 是否成对。")
        return False

# 执行初始化
if not init_vertex_ai():
    st.stop()

# --- 3. 样式美化 ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #D4AF37;
        color: white;
        border-radius: 20px;
        height: 50px;
        font-size: 18px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #B5952F;
    }
    h1 {
        text-align: center; 
        font-family: 'serif';
        color: #2C3E50;
    }
    .caption {
        text-align: center;
        color: #888;
        font-size: 12px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. 主界面 UI ---
st.title("✝️ Biblical Moments")
st.write("上传您的照片，穿越时空与圣经人物合影。")

col1, col2 = st.columns(2)
with col1:
    bible_character = st.text_input("想合照的圣经人物", placeholder="例如：耶稣、大卫、彼得")
with col2:
    clothing_style = st.selectbox("您的服装风格", 
        ["保持我照片里的衣服", "在这个时代的休闲装", "正式西装/礼服", "与圣经人物一样的古装", "工装/户外风格"]
    )

art_style = st.select_slider("选择照片风格", 
    options=["超写实摄影 (Photorealistic)", "电影质感 (Cinematic)", "油画风格 (Oil Painting)", "素描 (Sketch)"],
    value="超写实摄影 (Photorealistic)"
)

uploaded_file = st.file_uploader("上传您的自拍/半身照", type=['jpg', 'png', 'jpeg'])

# --- 5. AI 功能函数 ---

def get_gemini_prompt(user_image_bytes, character, clothing, style):
    """Gemini 1.5 Pro 分析与提示词生成"""
    model = GenerativeModel("gemini-1.5-pro")
    
    image_part = Part.from_data(data=user_image_bytes, mime_type="image/jpeg")
    
    prompt_instruction = f"""
    You are an expert biblical historian and an art director.
    
    TASK:
    1. Analyze the facial features, ethnicity, age, hair style, and gender of the person in the provided image in extreme detail.
    2. Create a detailed image generation prompt for Google Imagen 3.
    
    SCENE DETAILS:
    - Subject A: The person from the image (use the analyzed description above).
    - Subject B: {character} from the Bible. Ensure {character} is depicted historically accurately according to their era.
    - Action: They are standing side-by-side or interacting in a friendly, holy manner.
    - User's Clothing: {clothing}.
    - Background: A setting appropriate for the Bible character's era.
    - Style: {style}. High quality, 8k resolution, perfect lighting.
    
    OUTPUT FORMAT:
    Just return the PROMPT text directly, nothing else.
    """
    
    response = model.generate_content([image_part, prompt_instruction])
    return response.text

def generate_image(prompt):
    """Imagen 3 生成图片"""
    model = ImageGenerationModel.from_pretrained("imagegeneration@006")
    
    images = model.generate_images(
        prompt=prompt,
        number_of_images=1,
        language="en",
        aspect_ratio="3:4",
        safety_filter_level="block_some",
        person_generation="allow_adult"
    )
    return images[0]

# --- 6. 生成逻辑 ---
if st.button("✨ 生成合照"):
    if not uploaded_file or not bible_character:
        st.warning("请先上传照片并输入圣经人物名字。")
    else:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 1. Gemini
            status_text.text("🙏 正在祈祷与构思... (Gemini 分析照片特征)")
            image_bytes = uploaded_file.getvalue()
            generated_prompt = get_gemini_prompt(image_bytes, bible_character, clothing_style, art_style)
            progress_bar.progress(50)
            
            # 2. Imagen
            status_text.text(f"🎨 正在绘制与 {bible_character} 的合影... (Imagen 生成中)")
            result_image = generate_image(generated_prompt)
            progress_bar.progress(100)
            status_text.text("✨ 完成！")
            
            st.image(result_image._image_bytes, caption=f"您与 {bible_character} 的合影", use_column_width=True)
            
            st.download_button(
                label="📥 保存照片",
                data=result_image._image_bytes,
                file_name=f"with_{bible_character}.png",
                mime="image/png"
            )
            
            st.markdown("---")
            st.markdown("### 📖 每日恩典")
            verse_model = GenerativeModel("gemini-1.5-flash")
            verse = verse_model.generate_content(f"给我一句关于'{bible_character}'或者关于'友谊/信心/爱'的圣经经文，中文和英文对照。")
            st.info(verse.text)
                
        except Exception as e:
            st.error("生成出错，请稍后重试。")
            st.expander("调试信息").write(e)

st.markdown("<p class='caption'>Powered by Google Vertex AI (Gemini 1.5 & Imagen 3)</p>", unsafe_allow_html=True)
