import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image as VertexImage
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account
import json
import io

# 1. 页面配置
st.set_page_config(page_title="Biblical Moments - Direct Gen", page_icon="✝️", layout="centered")

# --- 2. 认证逻辑 (保持不变) ---
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
                    st.error("❌ Secrets 格式严重错误，无法解析。")
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

# --- 3. 样式美化 ---
st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #1F618D; color: white; border-radius: 20px; height: 50px; font-size: 18px; border: none; }
    h1 { text-align: center; font-family: 'serif'; color: #2C3E50; }
    .caption { text-align: center; color: #888; font-size: 12px; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

st.title("✝️ Biblical Moments")
st.caption("Mode: Direct Image Injection (No Text Analysis)")

col1, col2 = st.columns(2)
with col1:
    bible_character = st.text_input("想合照的圣经人物", placeholder="例如：耶稣、大卫")
with col2:
    clothing_style = st.selectbox("您的服装风格", ["保持原样", "换成圣经时代长袍", "换成现代西装"])

uploaded_file = st.file_uploader("上传您的自拍 (直接用于合成)", type=['jpg', 'png', 'jpeg'])

# --- 4. 核心 AI 逻辑 (直接合成模式) ---

def generate_direct_blend(user_image_bytes, character, clothing):
    """
    直接合成逻辑：
    1. 不调用 Gemini 分析图片。
    2. 直接将图片传给 Imagen 模型作为 base_image (底图)。
    3. 通过 Prompt 指导模型修改场景和添加人物。
    """
    
    # 🔴 指定模型 ID
    model_name = "imagen-4.0-generate-001"
    
    try:
        # 加载模型
        model = ImageGenerationModel.from_pretrained(model_name)
        
        # 将上传的字节流直接转为 Vertex AI Image 对象
        source_image = VertexImage(image_bytes=user_image_bytes)
        
        # 编写“合成指令”而非“描述指令”
        # 我们不再描述"那个人长什么样"，而是说"把这个人放在..."
        prompt = f"""
        A photorealistic image of the person from the input image standing side-by-side with {character} (Bible Figure).
        The scene is set in a historical biblical landscape.
        User's clothing: {clothing}.
        {character} is wearing historically accurate clothing.
        Cinematic lighting, high detail, 8k.
        """
        
        # 🔴 关键步骤：使用 edit_images (或类似接口) 直接传入图片
        # 注意：如果 4.0 API 的 generate 接口支持 reference_image 参数，也可以用 generate_images
        # 这里使用最通用的 edit_images 逻辑，将原图作为输入
        images = model.edit_images(
            prompt=prompt,
            base_image=source_image,  # <--- 这里直接把图喂给模型
            number_of_images=1,
            language="en",
            guidance_scale=60, # 较高的引导值，确保模型听从指令修改背景
            safety_filter_level="block_some",
            person_generation="allow_adult"
        )
        return images[0]
        
    except Exception as e:
        # 如果 4.0 暂时不支持直接喂图 (Edit模式)，回退到 3.0 (imagegeneration@006)
        st.warning(f"⚠️ 模型 {model_name} 的直接图片输入接口暂未就绪，已切换至 Imagen 3 Direct Edit 模式。")
        fallback_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
        source_image = VertexImage(image_bytes=user_image_bytes)
        images = fallback_model.edit_images(
            prompt=prompt,
            base_image=source_image,
            number_of_images=1
        )
        return images[0]

# --- 5. 执行逻辑 ---
if st.button("✨ 直接合成"):
    if not uploaded_file or not bible_character:
        st.warning("请先上传照片并输入人物。")
    else:
        try:
            progress = st.progress(0)
            status = st.empty()
            
            # 直接进入生成阶段，没有 Gemini 分析步骤了
            status.text(f"🎨 正在将您的照片直接传送给 Imagen 4.0...")
            img_bytes = uploaded_file.getvalue()
            
            # 调用直接合成函数
            result = generate_direct_blend(img_bytes, bible_character, clothing_style)
            
            progress.progress(100)
            status.text("✨ 合成完毕！")
            
            # 展示
            st.image(result._image_bytes, caption=f"您与 {bible_character}", use_column_width=True)
            
            # 下载
            st.download_button(
                label="📥 保存照片", 
                data=result._image_bytes, 
                file_name=f"direct_blend_{bible_character}.png", 
                mime="image/png"
            )
            
            # 经文 (仅保留文本功能)
            st.markdown("---")
            v_model = GenerativeModel("gemini-1.5-flash")
            verse = v_model.generate_content(f"One Bible verse about {bible_character}, bilingual.")
            st.info(verse.text)
            
        except Exception as e:
            st.error("合成失败")
            st.code(str(e))
