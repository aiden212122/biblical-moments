import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image as VertexImage
from google.oauth2 import service_account
import json

# 1. 页面配置
st.set_page_config(page_title="Nano Banana Direct", page_icon="🍌", layout="centered")

# --- 2. 认证逻辑 (Secrets 读取) ---
def init_vertex_ai():
    try:
        if "gcp_service_account" in st.secrets:
            # 读取 Secrets
            raw_json = st.secrets["gcp_service_account"]
            # 简单容错处理
            try:
                info = json.loads(raw_json, strict=False)
            except:
                # 尝试修复换行符
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
st.title("🍌 Nano Banana Direct")
st.caption("Target Model: gemini-2.5-flash-image")

col1, col2 = st.columns(2)
with col1:
    bible_character = st.text_input("合照人物", value="Jesus")
with col2:
    clothing = st.selectbox("服装", ["My original clothes", "Biblical Robes", "Modern Suit"])

uploaded_file = st.file_uploader("直接上传图片 (Feed Image)", type=['jpg', 'png', 'jpeg'])

# --- 4. 核心调用逻辑 ---
if st.button("🚀 Call Nano Banana API") and uploaded_file:
    try:
        status = st.empty()
        status.text("正在连接 gemini-2.5-flash-image...")
        
        # 1. 加载模型 (直接指定 Nano Banana ID)
        model_id = "gemini-2.5-flash-image"
        model = ImageGenerationModel.from_pretrained(model_id)
        
        # 2. 准备图片
        source_img = VertexImage(image_bytes=uploaded_file.getvalue())
        
        # 3. 编写提示词
        prompt = f"""
        Edit this image.
        Task: Place the person in the image standing next to {bible_character}.
        Setting: Realistic biblical era background.
        Clothing: The person wears {clothing}.
        Style: Photorealistic, 8k.
        Keep the person's face identical to the input image.
        """
        
        # 4. 直接调用 edit_images
        # 这是"图生图"的标准接口
        status.text("正在生成 (Image-to-Image)...")
        response = model.edit_images(
            prompt=prompt,
            base_image=source_img,  # <--- 核心：直接喂图
            number_of_images=1,
            guidance_scale=60,      # 较高的引导值
            language="en"
        )
        
        # 5. 展示结果
        result = response[0]
        st.image(result._image_bytes, caption="Nano Banana Output", use_column_width=True)
        
        # 下载
        st.download_button("📥 下载图片", result._image_bytes, "nano_output.png", "image/png")
        status.success("调用成功！")

    except Exception as e:
        st.error("API 调用失败")
        st.error(f"错误详情: {str(e)}")
        st.info("提示：如果报 '404 Not Found'，说明您的 Google Cloud 项目尚未获得该预览版模型的白名单权限。")
