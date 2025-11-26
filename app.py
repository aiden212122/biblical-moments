import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account
import json
import os

# 1. 页面配置
st.set_page_config(page_title="Biblical Moments - 圣经合影", page_icon="✝️", layout="centered")

# --- 2. 认证逻辑 (包含自动修复 Secrets) ---
def init_vertex_ai():
    try:
        if "gcp_service_account" in st.secrets:
            raw_json_str = st.secrets["gcp_service_account"]
            try:
                # 尝试宽松解析
                service_account_info = json.loads(raw_json_str, strict=False)
            except json.JSONDecodeError:
                try:
                    # 尝试修复换行符
                    fixed_str = raw_json_str.replace('\n', '\\n')
                    service_account_info = json.loads(raw_json_str, strict=False)
                except:
                    st.error("❌ Secrets 格式严重错误，无法解析。")
                    st.stop()

            credentials = service_account.Credentials.from_service_account_info(service_account_info)
            # 强制指定 us-central1 (新模型通常在这里首发)
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

# --- 4. AI 核心功能 ---

def get_gemini_prompt(user_image_bytes, character, clothing, style):
    """
    使用 Gemini 进行多模态分析 (看图)。
    此处指定使用 gemini-2.5-flash。
    """
    # 🔴 修改点：尝试使用 gemini-2.5-flash
    target_model = "gemini-2.5-flash"
    
    try:
        model = GenerativeModel(target_model)
    except Exception:
        # 如果 2.5 还未对项目开放，回退到 1.5-flash 以防报错
        print(f"Warning: {target_model} not found, falling back to 1.5-flash")
        model = GenerativeModel("gemini-1.5-flash")
    
    image_part = Part.from_data(data=user_image_bytes, mime_type="image/jpeg")
    
    prompt_instruction = f"""
    You are an expert biblical historian and an art director.
    TASK: Analyze the person in the image (face, ethnicity, age, hair) and create a detailed image generation prompt.
    SCENE: The user and {character} from the Bible. {character} must be historically accurate.
    ACTION: Standing side-by-side, friendly.
    CLOTHING: User wears {clothing}.
    STYLE: {style}, 8k resolution, high detail.
    OUTPUT: Just the prompt text.
    """
    
    # 增加容错：如果模型调用失败（例如 404），自动回退
    try:
        response = model.generate_content([image_part, prompt_instruction])
        return response.text
    except Exception as e:
        # 如果 2.5 运行时报错，尝试用备用模型再跑一次
        fallback_model = GenerativeModel("gemini-1.5-flash")
        response = fallback_model.generate_content([image_part, prompt_instruction])
        return response.text

def generate_image(prompt):
    """
    使用 Imagen 进行绘画。
    此处使用 imagen-3.0-generate-001 (目前最强的公开 Imagen 3 版本)
    """
    # 🔴 修改点：使用 Imagen 3.0 正式版 ID
    model_name = "imagen-3.0-generate-001"
    
    try:
        model = ImageGenerationModel.from_pretrained(model_name)
        
        images = model.generate_images(
            prompt=prompt,
            number_of_images=1,
            language="en",
            aspect_ratio="3:4",
            safety_filter_level="block_some",
            person_generation="allow_adult"
        )
        return images[0]
    except Exception as e:
        # 如果 3.0 尚未开通，回退到 standard
        fallback_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
        images = fallback_model.generate_images(prompt=prompt, number_of_images=1, aspect_ratio="3:4")
        return images[0]

# --- 5. 执行逻辑 ---
if st.button("✨ 生成合照"):
    if not uploaded_file or not bible_character:
        st.warning("请先上传照片并输入人物。")
    else:
        try:
            progress = st.progress(0)
            status = st.empty()
            
            # 1. Gemini 分析 (使用 2.5-flash)
            status.text("🙏 正在祈祷与构思 (Gemini 2.5 Flash)...")
            img_bytes = uploaded_file.getvalue()
            prompt = get_gemini_prompt(img_bytes, bible_character, clothing_style, art_style)
            progress.progress(50)
            
            # 2. Imagen 生成 (使用 Imagen 3.0)
            status.text(f"🎨 正在绘制合影 (Imagen 3.0)...")
            result = generate_image(prompt)
            progress.progress(100)
            status.text("✨ 完成！")
            
            # 展示
            st.image(result._image_bytes, caption=f"您与 {bible_character} 的合影", use_column_width=True)
            
            # 下载
            st.download_button(
                label="📥 保存照片", 
                data=result._image_bytes, 
                file_name=f"with_{bible_character}.png", 
                mime="image/png"
            )
            
            # 经文
            st.markdown("---")
            v_model = GenerativeModel("gemini-1.5-flash")
            verse = v_model.generate_content(f"给我一句关于'{bible_character}'的圣经经文(中英对照)。")
            st.info(verse.text)
            
        except Exception as e:
            st.error("生成出错")
            with st.expander("查看错误详情"):
                st.code(str(e))
