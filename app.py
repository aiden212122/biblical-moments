import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image as VertexImage
from google.oauth2 import service_account
import json

# 1. 页面配置
st.set_page_config(page_title="Biblical Moments - Preview", page_icon="✨", layout="centered")

# --- 2. 认证逻辑 (自动修复 Secrets) ---
def init_vertex_ai():
    try:
        if "gcp_service_account" in st.secrets:
            raw_json_str = st.secrets["gcp_service_account"]
            try:
                service_account_info = json.loads(raw_json_str, strict=False)
            except json.JSONDecodeError:
                try:
                    fixed_str = raw_json_str.replace('\n', '\\n')
                    service_account_info = json.loads(raw_json_str, strict=False)
                except:
                    st.error("❌ Secrets 格式严重错误")
                    st.stop()
            
            credentials = service_account.Credentials.from_service_account_info(service_account_info)
            # 强制指定 us-central1
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
    .stButton>button { width: 100%; background-color: #6C3483; color: white; border-radius: 20px; height: 50px; font-size: 18px; border: none; }
    h1 { text-align: center; font-family: 'serif'; color: #2C3E50; }
    .caption { text-align: center; color: #888; font-size: 12px; margin-top: 20px; }
    .stInfo { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("✝️ Biblical Moments")
st.caption("Engine: gemini-2.5-flash-image-preview")

col1, col2 = st.columns(2)
with col1:
    bible_character = st.text_input("想合照的圣经人物", placeholder="例如：耶稣、大卫")
with col2:
    clothing_style = st.selectbox("您的服装风格", ["保持原图着装", "圣经时代的古装长袍", "现代正装"])

uploaded_file = st.file_uploader("上传您的自拍 (直接用于合成)", type=['jpg', 'png', 'jpeg'])

# --- 4. 核心逻辑: Preview 模型直接编辑 ---

def generate_with_preview_model(user_image_bytes, character, clothing):
    """
    使用 gemini-2.5-flash-image-preview 直接进行图像编辑。
    """
    
    # 🔴 指定您要求的具体模型 ID
    model_name = "gemini-2.5-flash-image-preview"
    
    try:
        # 加载模型
        model = ImageGenerationModel.from_pretrained(model_name)
        
        # 准备图片对象
        source_image = VertexImage(image_bytes=user_image_bytes)
        
        # 编写编辑指令 (Prompt)
        # 这里的 Prompt 同时包含"视觉指令"和"生成指令"
        prompt = f"""
        Function: Edit Image.
        Task: Create a photorealistic photo of the person in this image standing side-by-side with {character} (Bible Figure).
        Details:
        - Keep the person's facial features EXACTLY the same.
        - Change the background to a realistic biblical landscape (e.g., Jerusalem, Desert).
        - Person's clothing: {clothing}.
        - {character}'s clothing: Historically accurate.
        - Lighting: Cinematic, soft, 8k resolution.
        """
        
        # 调用 edit_images
        # 这是"喂图"的关键步骤：base_image = source_image
        images = model.edit_images(
            prompt=prompt,
            base_image=source_image,
            number_of_images=1,
            guidance_scale=60, # 较高的引导值，强迫模型融合圣经人物
            language="en"
        )
        return images[0], "Gemini 2.5 Preview"

    except Exception as e:
        # === 智能兜底 ===
        # 如果 Preview 版暂未对您的 Project ID 开放，或者 API 签名不同
        # 自动无缝切换到目前最稳的 Imagen 2 编辑模式
        st.warning(f"⚠️ 预览版模型 ({model_name}) 响应异常: {str(e)}。已自动切换至标准高清编辑模式。")
        
        fallback_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
        source_image = VertexImage(image_bytes=user_image_bytes)
        images = fallback_model.edit_images(
            prompt=prompt,
            base_image=source_image,
            number_of_images=1
        )
        return images[0], "Standard Backup"

# --- 5. 执行逻辑 ---
if st.button("✨ 立即生成合照"):
    if not uploaded_file or not bible_character:
        st.warning("请先上传照片并输入人物。")
    else:
        try:
            progress = st.progress(0)
            status = st.empty()
            
            status.text(f"🚀 正在调用 {bible_character} 并处理图像...")
            img_bytes = uploaded_file.getvalue()
            
            # 调用生成函数
            result_image, method = generate_with_preview_model(img_bytes, bible_character, clothing_style)
            
            progress.progress(100)
            status.text("✨ 完成！")
            
            # 展示
            st.image(result_image._image_bytes, caption=f"合照 ({method})", use_column_width=True)
            
            # 下载
            st.download_button(
                label="📥 保存照片", 
                data=result_image._image_bytes, 
                file_name=f"gemini_2.5_preview_{bible_character}.png", 
                mime="image/png"
            )
            
        except Exception as e:
            st.error("生成失败，请稍后重试。")
            with st.expander("查看详情"):
                st.code(str(e))
