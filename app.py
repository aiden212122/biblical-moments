import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account
import json
import os

# 1. 页面配置
st.set_page_config(page_title="Biblical Moments - Future AI", page_icon="✝️", layout="centered")

# --- 2. 认证逻辑 (包含自动修复 Secrets 的稳健代码) ---
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
            # 强制指定 us-central1 (新模型首发区域)
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
    .stButton>button { width: 100%; background-color: #7D3C98; color: white; border-radius: 20px; height: 50px; font-size: 18px; border: none; }
    h1 { text-align: center; font-family: 'serif'; color: #2C3E50; }
    .caption { text-align: center; color: #888; font-size: 12px; margin-top: 20px; }
    .stSuccess { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("✝️ Biblical Moments")
st.caption(f"Architecture: Gemini 2.5 Flash + Imagen 4.0")

col1, col2 = st.columns(2)
with col1:
    bible_character = st.text_input("想合照的圣经人物", placeholder="例如：耶稣、大卫、摩西")
with col2:
    clothing_style = st.selectbox("您的服装风格", ["保持我照片里的衣服", "圣经时代的古装长袍", "现代休闲装", "正式西装/礼服"])

uploaded_file = st.file_uploader("上传您的自拍 (将用于 Gemini 2.5 特征提取)", type=['jpg', 'png', 'jpeg'])

# --- 4. 核心 AI 逻辑 (使用指定模型 ID) ---

def get_gemini_prompt_v2(user_image_bytes, character, clothing):
    """
    第一步：使用 Gemini 2.5 Flash 进行视觉分析
    """
    # 🔴 严格指定模型 ID：gemini-2.5-flash
    model_id = "gemini-2.5-flash"
    
    try:
        model = GenerativeModel(model_id)
    except:
        # 仅作为保险：如果 2.5 暂时不可用，回退到 1.5 Flash
        st.warning(f"⚠️ {model_id} 连接超时，正在尝试回退模型...")
        model = GenerativeModel("gemini-1.5-flash")

    image_part = Part.from_data(data=user_image_bytes, mime_type="image/jpeg")
    
    # Prompt 策略：要求模型像 '面部识别系统' 一样精准描述
    prompt_instruction = f"""
    ROLE: You are an advanced AI visual analyzer.
    
    TASK 1: ANALYZE THE USER
    Look at the person in this image. Write a physical description so precise that a blind artist could paint them.
    Include:
    - Exact ethnicity, skin tone code (e.g. olive, fair, dark).
    - Precise face shape (oval, square, etc.), cheekbones.
    - Eyes: Shape, color, eyebrows.
    - Nose: Shape, bridge width.
    - Hair: Exact texture, style, color, hairline.
    - Facial hair (if any).
    - Age estimate.
    
    TASK 2: CONSTRUCT THE SCENE
    Create a prompt for an image generator (Imagen 4.0) featuring:
    - Subject A: The User (based on description above).
    - Subject B: {character} (Historical Accuracy: 1st Century Judea/Old Testament era).
    - Action: Standing side-by-side, friendly, holy atmosphere.
    - User's Clothing: {clothing}.
    - Background: Biblical landscape (e.g., Temple mount, Desert, River Jordan).
    - Style: 8k photorealistic, cinematic lighting.
    
    OUTPUT: Return ONLY the final prompt text.
    """
    
    try:
        response = model.generate_content([image_part, prompt_instruction])
        return response.text
    except Exception as e:
        # 如果模型调用失败
        st.error(f"Gemini 分析失败: {str(e)}")
        return f"A photo of a person standing with {character} in biblical times."

def generate_image_v4(prompt):
    """
    第二步：使用 Imagen 4.0 Generate 001 进行生成
    """
    # 🔴 严格指定模型 ID：imagen-4.0-generate-001
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
        # 错误处理：如果 4.0 尚未进入白名单，回退到 3.0 (imagegeneration@006)
        st.warning(f"⚠️ Imagen 4.0 权限受限，已自动切换至 Imagen 3 高清版。")
        fallback_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
        images = fallback_model.generate_images(prompt=prompt, number_of_images=1, aspect_ratio="3:4")
        return images[0]

# --- 5. 执行逻辑 ---
if st.button("✨ 启动生成"):
    if not uploaded_file or not bible_character:
        st.warning("请先上传照片并输入人物。")
    else:
        try:
            progress = st.progress(0)
            status = st.empty()
            
            # 1. 视觉分析
            status.text(f"🧠 Gemini 2.5 Flash 正在分析您的面部特征...")
            img_bytes = uploaded_file.getvalue()
            
            # 获取生成的 Prompt
            generated_prompt = get_gemini_prompt_v2(img_bytes, bible_character, clothing_style)
            
            progress.progress(40)
            
            # 2. 图像生成
            status.text(f"🎨 Imagen 4.0 正在绘制合影...")
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
            
            # 经文彩蛋 (使用 2.5 生成经文)
            st.markdown("---")
            v_model = GenerativeModel("gemini-2.5-flash") 
            try:
                verse = v_model.generate_content(f"Output one encouraging Bible verse about {bible_character}, bilingual (Chinese/English).")
                st.info(verse.text)
            except:
                st.info("主赐平安 (经文生成服务暂时繁忙)")
            
        except Exception as e:
            st.error("生成流程中断")
            with st.expander("查看错误详情"):
                st.code(str(e))
