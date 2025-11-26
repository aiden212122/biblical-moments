import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account
import json
import os

# 1. 页面配置
st.set_page_config(page_title="Biblical Moments - AI Gen", page_icon="✝️", layout="centered")

# --- 2. 认证逻辑 (保持稳健的容错机制) ---
def init_vertex_ai():
    try:
        if "gcp_service_account" in st.secrets:
            raw_json_str = st.secrets["gcp_service_account"]
            try:
                # 尝试标准解析
                service_account_info = json.loads(raw_json_str, strict=False)
            except json.JSONDecodeError:
                try:
                    # 尝试自动修复换行符问题
                    fixed_str = raw_json_str.replace('\n', '\\n')
                    service_account_info = json.loads(raw_json_str, strict=False)
                except:
                    st.error("❌ Secrets 格式严重错误，无法解析。")
                    st.stop()

            credentials = service_account.Credentials.from_service_account_info(service_account_info)
            # 强制指定 us-central1 (新模型通常在此区域首发)
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

# --- 3. 样式美化 ---
st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #2E86C1; color: white; border-radius: 20px; height: 50px; font-size: 18px; border: none; }
    h1 { text-align: center; font-family: 'serif'; color: #2C3E50; }
    .caption { text-align: center; color: #888; font-size: 12px; margin-top: 20px; }
    .stSuccess { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("✝️ Biblical Moments (Pro)")
st.caption(f"Engine: Gemini 2.5 Flash + Imagen 4.0")

col1, col2 = st.columns(2)
with col1:
    bible_character = st.text_input("想合照的圣经人物", placeholder="例如：耶稣、大卫、摩西")
with col2:
    clothing_style = st.selectbox("您的服装风格", ["保持我照片里的衣服", "圣经时代的古装长袍", "现代休闲装", "正式西装/礼服"])

uploaded_file = st.file_uploader("上传您的自拍 (将用于特征分析)", type=['jpg', 'png', 'jpeg'])

# --- 4. 核心 AI 逻辑 (使用您指定的特定模型 ID) ---

def get_gemini_prompt_v2(user_image_bytes, character, clothing):
    """
    第一步：使用 Gemini 2.5 Flash Preview 进行超精细视觉分析
    """
    # 🔴 指定模型 ID：gemini-2.5-flash-preview
    model_id = "gemini-2.5-flash-preview"
    
    try:
        model = GenerativeModel(model_id)
    except:
        # 如果预览版 ID 不可用，回退到 1.5 Pro
        print(f"Model {model_id} not found, falling back.")
        model = GenerativeModel("gemini-1.5-pro")

    image_part = Part.from_data(data=user_image_bytes, mime_type="image/jpeg")
    
    # 编写超级 Prompt：要求 Gemini 充当“摄影师导演”，把用户的脸描述得像代码一样精准
    prompt_instruction = f"""
    ROLE: You are an expert AI Image Prompt Engineer.
    
    INPUT: An image of a USER and a target BIBLE CHARACTER: {character}.
    USER CLOTHING GOAL: {clothing}.
    
    TASK: Write a highly detailed, photorealistic prompt for Imagen 4.0 to generate a photo of the USER standing with {character}.
    
    CRITICAL IDENTITY INSTRUCTIONS:
    1. Analyze the USER in the image: Describe their face, ethnicity, age, specific eye shape, nose shape, hair style, and hair color in EXTREME DETAIL.
    2. Do NOT use the user's name, just describe their visual appearance physically so the image generator can reconstruct them.
    
    SCENE INSTRUCTIONS:
    1. Subject B: {character} (Historical accuracy is mandatory).
    2. Background: Realistic biblical era setting (e.g., Jerusalem stone streets, Desert, Sea of Galilee).
    3. Lighting: Cinematic, Golden Hour, Soft lighting.
    4. Style: Award-winning photography, 8k, hyper-realistic.
    
    OUTPUT: Return ONLY the raw prompt text. No markdown, no explanations.
    """
    
    try:
        response = model.generate_content([image_part, prompt_instruction])
        return response.text
    except Exception as e:
        st.error(f"Gemini 分析失败: {str(e)}")
        # 降级备选
        return f"A photo of a person standing with {character} in biblical times."

def generate_image_v4(prompt):
    """
    第二步：使用 Imagen 4.0 Generate 001 进行生成
    """
    # 🔴 指定模型 ID：imagen-4.0-generate-001
    model_name = "imagen-4.0-generate-001"
    
    try:
        model = ImageGenerationModel.from_pretrained(model_name)
        
        images = model.generate_images(
            prompt=prompt,
            number_of_images=1,
            language="en",
            aspect_ratio="3:4", # 竖屏适合手机
            safety_filter_level="block_some",
            person_generation="allow_adult"
        )
        return images[0]
        
    except Exception as e:
        # 如果 4.0 未授权，回退到 3.0 或标准版
        st.warning(f"⚠️ Imagen 4.0 调用受限 ({str(e)})，正在尝试切换至 Imagen 3...")
        fallback_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
        images = fallback_model.generate_images(prompt=prompt, number_of_images=1, aspect_ratio="3:4")
        return images[0]

# --- 5. 执行逻辑 ---
if st.button("✨ 启动生成引擎"):
    if not uploaded_file or not bible_character:
        st.warning("请先上传照片并输入人物。")
    else:
        try:
            progress = st.progress(0)
            status = st.empty()
            
            # 1. 视觉分析
            status.text(f"🧠 Gemini 2.5 Flash 正在解析您的面部特征...")
            img_bytes = uploaded_file.getvalue()
            
            # 获取生成的 Prompt
            generated_prompt = get_gemini_prompt_v2(img_bytes, bible_character, clothing_style)
            
            # 调试模式：如果您想看 Gemini 写了什么提示词，可以把下面这行注释取消
            # st.expander("查看生成的 Prompt").write(generated_prompt)
            
            progress.progress(40)
            
            # 2. 图像生成
            status.text(f"🎨 Imagen 4.0 正在渲染高精度合影...")
            result = generate_image_v4(generated_prompt)
            progress.progress(100)
            status.text("✨ 生成完毕！")
            
            # 展示
            st.image(result._image_bytes, caption=f"您与 {bible_character}", use_column_width=True)
            
            # 下载
            st.download_button(
                label="📥 保存原图", 
                data=result._image_bytes, 
                file_name=f"imagen4_gen.png", 
                mime="image/png"
            )
            
            # 经文彩蛋
            st.markdown("---")
            v_model = GenerativeModel("gemini-1.5-flash") # 经文生成用普通版足够，省钱
            verse = v_model.generate_content(f"Output one Bible verse about {bible_character} or 'Faith', bilingual (Chinese/English).")
            st.info(verse.text)
            
        except Exception as e:
            st.error("生成流程中断")
            with st.expander("查看错误详情"):
                st.code(str(e))

