import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from vertexai.generative_models import GenerativeModel, Part, Image
import tempfile
import os

# --- 配置页面 ---
st.set_page_config(page_title="Biblical Moments - 与圣经人物合影", page_icon="✝️", layout="centered")

# --- 自定义CSS (手机端优化 & 风格) ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #D4AF37; /* 金色 */
        color: white;
        border-radius: 20px;
        height: 50px;
        font-size: 18px;
    }
    .stTextInput>div>div>input {
        text-align: center;
    }
    h1 {
        text-align: center; 
        font-family: 'Georgia', serif;
        color: #2C3E50;
    }
    .caption {
        text-align: center;
        color: #888;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --- 侧边栏：API 设置 (为了安全，也可以放在 .streamlit/secrets.toml 中) ---
with st.sidebar:
    st.header("⚙️ 设置")
    project_id = st.text_input("Google Cloud Project ID", value="your-project-id")
    location = st.text_input("Region", value="us-central1")
    
    # 初始化 Vertex AI
    if project_id:
        try:
            vertexai.init(project=project_id, location=location)
            st.success("Google Cloud 连接成功")
        except Exception as e:
            st.error(f"连接失败: {e}")

# --- 主界面 ---
st.title("✝️ Biblical Moments")
st.write("上传您的照片，穿越时空与圣经人物合影。")

# 1. 用户输入
col1, col2 = st.columns(2)
with col1:
    bible_character = st.text_input("想合照的圣经人物", placeholder="例如：耶稣、大卫王、摩西")
with col2:
    clothing_style = st.selectbox("您的服装风格", 
        ["保持我照片里的衣服", "在这个时代的休闲装", "正式西装/礼服", "与圣经人物一样的古装", "工装/户外风格"]
    )

art_style = st.select_slider("选择照片风格", 
    options=["超写实摄影 (Photorealistic)", "电影质感 (Cinematic)", "油画风格 (Oil Painting)", "素描 (Sketch)"],
    value="超写实摄影 (Photorealistic)"
)

# 2. 图片上传
uploaded_file = st.file_uploader("上传您的自拍/半身照", type=['jpg', 'png', 'jpeg'])

def get_gemini_prompt(user_image_bytes, character, clothing, style):
    """
    使用 Gemini 1.5 Pro 分析用户照片并生成 Imagen 3 的提示词
    """
    model = GenerativeModel("gemini-1.5-pro-001") # 或最新的 gemini-1.5-pro
    
    image_part = Part.from_data(data=user_image_bytes, mime_type="image/jpeg")
    
    prompt_instruction = f"""
    You are an expert biblical historian and an art director.
    
    TASK:
    1. Analyze the facial features, ethnicity, age, hair style, and gender of the person in the provided image in extreme detail.
    2. Create a detailed image generation prompt for Google Imagen 3.
    
    SCENE DETAILS:
    - Subject A: The person from the image (use the analyzed description above).
    - Subject B: {character} from the Bible. Ensure {character} is depicted historically accurately according to their era (1st century Judea, Old Testament Egypt, etc.). NO Europeanized Jesus if not historically accurate.
    - Action: They are standing side-by-side or interacting in a friendly, holy manner (e.g., talking, walking, selfie).
    - User's Clothing: {clothing}.
    - Background: A setting appropriate for the Bible character's era (e.g., Sea of Galilee, Temple, Desert).
    - Style: {style}. High quality, 8k resolution, perfect lighting.
    
    OUTPUT FORMAT:
    Just return the PROMPT text directly, nothing else.
    """
    
    response = model.generate_content([image_part, prompt_instruction])
    return response.text

def generate_image(prompt):
    """
    调用 Imagen 3 生成图片
    """
    model = ImageGenerationModel.from_pretrained("imagegeneration@006") # imagen-3 版本通常是 006 或 latest
    
    images = model.generate_images(
        prompt=prompt,
        number_of_images=1,
        language="en",
        aspect_ratio="3:4", # 适合手机竖屏
        safety_filter_level="block_some",
        person_generation="allow_adult"
    )
    return images[0]

# 3. 生成逻辑
if st.button("✨ 生成合照"):
    if not uploaded_file or not bible_character:
        st.warning("请先上传照片并输入圣经人物名字。")
    else:
        with st.spinner("正在祈祷与构思... (Gemini 正在分析您的照片)"):
            try:
                # 读取图片数据
                image_bytes = uploaded_file.getvalue()
                
                # 第一阶段：Gemini 编写提示词
                generated_prompt = get_gemini_prompt(image_bytes, bible_character, clothing_style, art_style)
                # st.expander("查看生成的提示词 (调试用)").write(generated_prompt) # 调试时可打开
                
                with st.spinner(f"正在与 {bible_character} 合影... (Imagen 3 正在生成)"):
                    # 第二阶段：Imagen 生成图片
                    result_image = generate_image(generated_prompt)
                    
                    # 展示结果
                    st.image(result_image._image_bytes, caption=f"您与 {bible_character} 的合影", use_column_width=True)
                    
                    # 下载按钮
                    st.download_button(
                        label="📥 保存照片",
                        data=result_image._image_bytes,
                        file_name=f"with_{bible_character}.png",
                        mime="image/png"
                    )
                    
                    # 额外功能：生成一句经文
                    st.markdown("---")
                    st.markdown("### 📖 今日恩典")
                    verse_model = GenerativeModel("gemini-pro")
                    verse = verse_model.generate_content(f"给我一句关于'{bible_character}'或者关于'友谊/陪伴'的圣经经文，中文和英文对照。")
                    st.info(verse.text)
                    
            except Exception as e:
                st.error(f"生成过程中出现错误: {str(e)}")
                st.info("提示：请检查您的 Google Cloud 额度或 API 权限。")

st.markdown("<p class='caption'>Powered by Google Gemini 1.5 & Imagen 3</p>", unsafe_allow_html=True)