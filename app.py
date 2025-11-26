import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image as VertexImage
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account
import json
import importlib.metadata

# 1. 页面配置
st.set_page_config(page_title="Biblical Moments - Final", page_icon="✝️", layout="centered")

# --- 2. 认证逻辑 ---
def init_vertex_ai():
    try:
        if "gcp_service_account" in st.secrets:
            raw_json = st.secrets["gcp_service_account"]
            try:
                info = json.loads(raw_json, strict=False)
            except:
                info = json.loads(raw_json.replace('\n', '\\n'), strict=False)
            
            creds = service_account.Credentials.from_service_account_info(info)
            vertexai.init(project=info["project_id"], location="us-central1", credentials=creds)
            return True
        else:
            vertexai.init(location="us-central1")
            return True
    except Exception as e:
        st.error(f"认证失败: {e}")
        return False

if not init_vertex_ai():
    st.stop()

# --- 3. 界面 ---
st.title("✝️ Biblical Moments")

# 检查库版本 (调试用)
try:
    ver = importlib.metadata.version("google-cloud-aiplatform")
    st.caption(f"System Status: Google Cloud SDK v{ver}")
except:
    st.caption("System Status: SDK version unknown")

col1, col2 = st.columns(2)
with col1:
    bible_character = st.text_input("合照人物", value="Jesus")
with col2:
    clothing = st.selectbox("服装", ["My original clothes", "Biblical Robes", "Modern Suit"])

uploaded_file = st.file_uploader("上传您的自拍", type=['jpg', 'png', 'jpeg'])

# --- 4. 核心逻辑: 智能双模态 ---
def generate_smart(user_image_bytes, character, clothing):
    status_box = st.empty()
    
    # 尝试 A: 直接图片编辑 (Image-to-Image)
    # 只有当 requirements.txt 更新成功且 SDK 版本足够新时，这里才会成功
    try:
        status_box.text("🚀 正在尝试直接图片融合 (Mode A)...")
        model = ImageGenerationModel.from_pretrained("imagegeneration@006")
        source_img = VertexImage(image_bytes=user_image_bytes)
        
        prompt = f"""
        A photorealistic shot of the person in the input image standing side-by-side with {character} from the Bible.
        Background: Realistic biblical landscape.
        User clothing: {clothing}.
        {character} is wearing historical robes.
        Quality: 8k.
        """
        
        # 这一句是关键，如果报错 AttributeError，说明 SDK 版本还是旧的
        images = model.edit_images(
            prompt=prompt,
            base_image=source_img,
            number_of_images=1,
            guidance_scale=60
        )
        return images[0], "直接融合模式 (Best)"

    except AttributeError:
        # 捕捉到您刚才遇到的错误，自动切换
        status_box.warning("⚠️ 云端环境版本较旧，不支持直接编辑。正在自动切换至视觉重构模式 (Mode B)...")
        
        # 尝试 B: 视觉分析 + 重构 (Gemini -> Imagen)
        # 这是永远不会崩的保底方案
        return generate_fallback(user_image_bytes, character, clothing, status_box)

    except Exception as e:
        st.error(f"Mode A 报错: {e}")
        status_box.text("切换至 Mode B...")
        return generate_fallback(user_image_bytes, character, clothing, status_box)

def generate_fallback(user_image_bytes, character, clothing, status_box):
    """
    保底模式：先看图，再画图。
    """
    try:
        # 1. 视觉分析
        status_box.text("👀 正在分析面部特征...")
        try:
            gemini = GenerativeModel("gemini-1.5-flash")
        except:
            gemini = GenerativeModel("gemini-pro-vision")
            
        img_part = Part.from_data(data=user_image_bytes, mime_type="image/jpeg")
        desc = gemini.generate_content([img_part, "Describe this person's face, hair, age, and ethnicity in detail for an image generator."]).text
        
        # 2. 生成图像
        status_box.text("🎨 正在绘制合照...")
        model = ImageGenerationModel.from_pretrained("imagegeneration@006")
        prompt = f"A photo of {desc} standing with {character} (Bible figure). Biblical background. {clothing}. 8k resolution."
        
        images = model.generate_images(prompt=prompt, number_of_images=1, aspect_ratio="3:4")
        return images[0], "视觉重构模式 (Backup)"
        
    except Exception as e:
        raise RuntimeError(f"所有模式均失败: {e}")

# --- 5. 执行 ---
if st.button("✨ 生成合照") and uploaded_file:
    try:
        progress = st.progress(0)
        img_bytes = uploaded_file.getvalue()
        
        result, method = generate_smart(img_bytes, bible_character, clothing)
        
        progress.progress(100)
        st.success("生成成功！")
        st.image(result._image_bytes, caption=f"Result ({method})", use_column_width=True)
        st.download_button("📥 下载图片", result._image_bytes, "bible_photo.png", "image/png")
        
    except Exception as e:
        st.error("生成失败")
        st.code(str(e))
