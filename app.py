import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account
import json
import os

# 1. 页面配置必须放在第一行
st.set_page_config(page_title="Biblical Moments - 圣经合影", page_icon="✝️", layout="centered")

# --- 2. Google Cloud 核心认证逻辑 (这是修改的重点) ---
def init_vertex_ai():
    """
    初始化 Vertex AI 连接。
    优先从 Streamlit Secrets 读取 Service Account，
    如果在本地运行且没有 Secrets，则尝试读取环境变量。
    """
    try:
        # 情况 A: 在 Streamlit Cloud 上运行 (读取 Secrets)
        if "gcp_service_account" in st.secrets:
            # 1. 解析 Secrets 里的 JSON 字符串
            service_account_info = json.loads(st.secrets["gcp_service_account"])
            
            # 2. 创建凭证对象
            credentials = service_account.Credentials.from_service_account_info(service_account_info)
            
            # 3. 初始化 (Project ID 自动从 JSON 里获取)
            vertexai.init(
                project=service_account_info["project_id"], 
                location="us-central1", 
                credentials=credentials
            )
            return True

        # 情况 B: 在本地电脑运行 (依赖环境变量 GOOGLE_APPLICATION_CREDENTIALS)
        else:
            # 尝试直接初始化 (假设用户本地已配置好 gcloud auth 或环境变量)
            vertexai.init(location="us-central1")
            return True
            
    except Exception as e:
        st.error(f"⚠️ 认证失败: 请检查 Secrets 配置。\n错误详情: {e}")
        return False

# 执行初始化
if not init_vertex_ai():
    st.stop() # 如果认证失败，停止运行后续代码

# --- 3. 样式美化 ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #D4AF37; /* 金色 */
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

# 输入区域
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

# --- 5. 核心 AI 功能函数 ---

def get_gemini_prompt(user_image_bytes, character, clothing, style):
    """
    使用 Gemini 1.5 Pro 分析用户照片并生成 Prompt
    """
    # 加载模型
    model = GenerativeModel("gemini-1.5-pro")
    
    image_part = Part.from_data(data=user_image_bytes, mime_type="image/jpeg")
    
    prompt_instruction = f"""
    You are an expert biblical historian and an art director.
    
    TASK:
    1. Analyze the facial features, ethnicity, age, hair style, and gender of the person in the provided image in extreme detail.
    2. Create a detailed image generation prompt for Google Imagen 3.
    
    SCENE DETAILS:
    - Subject A: The person from the image (use the analyzed description above).
    - Subject B: {character} from the Bible. Ensure {character} is depicted historically accurately according to their era (1st century Judea, Old Testament Egypt, etc.).
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
    """
    调用 Imagen 3 生成图片
    """
    # 加载模型: imagegeneration@006 是目前的 Imagen 3 模型
    model = ImageGenerationModel.from_pretrained("imagegeneration@006")
    
    images = model.generate_images(
        prompt=prompt,
        number_of_images=1,
        language="en",
        aspect_ratio="3:4", # 竖屏适合手机
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
            # 显示进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 第一步：Gemini 分析
            status_text.text("🙏 正在祈祷与构思... (Gemini 分析照片特征)")
            image_bytes = uploaded_file.getvalue()
            generated_prompt = get_gemini_prompt(image_bytes, bible_character, clothing_style, art_style)
            progress_bar.progress(50)
            
            # 第二步：Imagen 作画
            status_text.text(f"🎨 正在绘制与 {bible_character} 的合影... (Imagen 生成中)")
            result_image = generate_image(generated_prompt)
            progress_bar.progress(100)
            status_text.text("✨ 完成！")
            
            # 展示图片
            st.image(result_image._image_bytes, caption=f"您与 {bible_character} 的合影", use_column_width=True)
            
            # 下载按钮
            st.download_button(
                label="📥 保存照片",
                data=result_image._image_bytes,
                file_name=f"with_{bible_character}.png",
                mime="image/png"
            )
            
            # 额外：生成经文
            st.markdown("---")
            st.markdown("### 📖 每日恩典")
            verse_model = GenerativeModel("gemini-1.5-flash") # 使用 Flash 模型速度更快
            verse = verse_model.generate_content(f"给我一句关于'{bible_character}'或者关于'友谊/信心/爱'的圣经经文，中文和英文对照。")
            st.info(verse.text)
                
        except Exception as e:
            st.error("生成过程中出现错误，请稍后再试。")
            with st.expander("查看错误详情 (调试用)"):
                st.write(e)

st.markdown("<p class='caption'>Powered by Google Vertex AI (Gemini 1.5 & Imagen 3)</p>", unsafe_allow_html=True)
