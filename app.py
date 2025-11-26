import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image as VertexImage
from google.oauth2 import service_account
import json

# 1. 页面配置
st.set_page_config(page_title="Biblical Moments - Direct", page_icon="✝️", layout="centered")

# --- 2. 认证逻辑 ---
def init_vertex_ai():
    try:
        if "gcp_service_account" in st.secrets:
            raw_json_str = st.secrets["gcp_service_account"]
            try:
                service_account_info = json.loads(raw_json_str, strict=False)
            except json.JSONDecodeError:
                fixed_str = raw_json_str.replace('\n', '\\n')
                service_account_info = json.loads(fixed_str, strict=False)
            
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
    .stButton>button { width: 100%; background-color: #17202A; color: white; border-radius: 20px; height: 50px; font-size: 18px; border: none; }
    h1 { text-align: center; font-family: 'serif'; color: #2C3E50; }
</style>
""", unsafe_allow_html=True)

st.title("✝️ Biblical Moments")
st.caption("Model: Imagen 2 (The King of Direct Image Editing)")

col1, col2 = st.columns(2)
with col1:
    bible_character = st.text_input("想合照的圣经人物", placeholder="例如：耶稣")
with col2:
    clothing_style = st.selectbox("您的服装风格", ["保持原样", "换成圣经时代长袍", "工装风格"])

uploaded_file = st.file_uploader("上传您的自拍 (直接喂给模型)", type=['jpg', 'png', 'jpeg'])

# --- 4. 核心逻辑：直接喂图 ---

def generate_direct_edit(user_image_bytes, character, clothing):
    """
    使用 imagegeneration@006 (Imagen 2) 的 edit_images 接口。
    这是目前唯一支持直接 'Image-to-Image' 的稳定模型 ID。
    """
    
    # 🔴 关键：使用支持编辑的模型 ID
    model_name = "imagegeneration@006"
    
    try:
        model = ImageGenerationModel.from_pretrained(model_name)
        source_image = VertexImage(image_bytes=user_image_bytes)
        
        # 编写编辑指令
        # 这里的 Prompt 不是描述画面，而是告诉模型怎么"改"
        prompt = f"""
        Keep the person in the foreground exactly as they are.
        Change the background to a photorealistic biblical scene featuring {character}.
        Ensure {character} is standing next to the person in a friendly way.
        The person should be wearing {clothing}.
        High resolution, cinematic lighting, 8k.
        """
        
        # 🔴 调用 edit_images
        # base_image 就是您上传的图，模型会基于这张图进行修改
        images = model.edit_images(
            prompt=prompt,
            base_image=source_image,
            number_of_images=1,
            guidance_scale=60, # 较高的约束力，防止画面跑偏
            safety_filter_level="block_some",
            person_generation="allow_adult"
        )
        return images[0]
        
    except Exception as e:
        raise RuntimeError(f"模型调用失败: {str(e)}")

# --- 5. 执行 ---
if st.button("✨ 生成合照"):
    if not uploaded_file or not bible_character:
        st.warning("请先上传照片并输入人物。")
    else:
        try:
            progress = st.progress(0)
            status = st.empty()
            
            status.text(f"🎨 正在将照片直接喂给 Imagen 2...")
            img_bytes = uploaded_file.getvalue()
            
            # 直接调用
            result = generate_direct_edit(img_bytes, bible_character, clothing_style)
            
            progress.progress(100)
            status.text("✨ 完成！")
            
            st.image(result._image_bytes, caption=f"您与 {bible_character}", use_column_width=True)
            
            st.download_button(label="📥 保存照片", data=result._image_bytes, file_name="result.png", mime="image/png")
            
        except Exception as e:
            st.error("生成失败")
            st.code(str(e))
            st.info("提示：请确保您的 Google Cloud 项目已启用 Vertex AI API。")
