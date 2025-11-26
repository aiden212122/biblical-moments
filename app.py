import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image as VertexImage
from google.oauth2 import service_account
import json

# 1. 页面配置
st.set_page_config(page_title="Biblical Moments - Stable", page_icon="✝️", layout="centered")

# --- 2. 认证逻辑 (Secrets) ---
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
st.caption("Engine: Google Imagen 2 (Production Stable)")

col1, col2 = st.columns(2)
with col1:
    bible_character = st.text_input("合照人物", value="Jesus")
with col2:
    clothing = st.selectbox("服装", ["My original clothes", "Biblical Robes", "Modern Suit"])

uploaded_file = st.file_uploader("上传您的自拍 (直接合成)", type=['jpg', 'png', 'jpeg'])

# --- 4. 核心逻辑: 使用 Imagen 2 进行编辑 ---
if st.button("✨ 生成合照") and uploaded_file:
    try:
        progress = st.progress(0)
        status = st.empty()
        status.text("正在连接 Google Cloud...")
        
        # 1. 加载模型
        # imagegeneration@006 是目前唯一支持 edit_images 的稳定版模型 ID
        model = ImageGenerationModel.from_pretrained("imagegeneration@006")
        
        # 2. 准备图片
        source_img = VertexImage(image_bytes=uploaded_file.getvalue())
        
        # 3. 编写提示词
        # 这里的技巧是：告诉模型“背景变了，身边多了个人，但原来的主体保持不变”
        prompt = f"""
        A photorealistic shot of the person in the input image standing side-by-side with {bible_character} from the Bible.
        Background: A realistic biblical landscape (Desert or Ancient City).
        Lighting: Cinematic, soft, warm sunlight.
        Quality: 8k, highly detailed.
        User's clothing: {clothing}.
        {bible_character} is wearing historically accurate robes.
        """
        
        status.text("正在进行图像融合 (Image-to-Image)...")
        
        # 4. 调用 edit_images
        # base_image 参数就是您的“喂图”
        images = model.edit_images(
            prompt=prompt,
            base_image=source_img,
            number_of_images=1,
            guidance_scale=60, # 较高的引导值，强制模型听从Prompt修改背景
            language="en"
        )
        
        progress.progress(100)
        status.success("生成成功！")
        
        # 5. 展示结果
        result = images[0]
        st.image(result._image_bytes, caption=f"With {bible_character}", use_column_width=True)
        
        # 下载
        st.download_button("📥 保存图片", result._image_bytes, "bible_photo.png", "image/png")

    except Exception as e:
        st.error("生成失败")
        st.error(f"错误信息: {str(e)}")
        st.info("提示：请确保您的 Google Cloud 项目已启用 Vertex AI API。")
