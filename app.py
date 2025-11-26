import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image as VertexImage
from google.oauth2 import service_account
import json

# 1. 页面配置
st.set_page_config(page_title="Biblical Moments - Nano", page_icon="🍌", layout="centered")

# --- 2. 认证逻辑 ---
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
            # 强制指定 us-central1 (新模型首发区)
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
    .stButton>button { width: 100%; background-color: #F1C40F; color: black; border-radius: 20px; height: 50px; font-size: 18px; border: none; font-weight: bold; }
    h1 { text-align: center; font-family: 'serif'; color: #2C3E50; }
    .caption { text-align: center; color: #888; font-size: 12px; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

st.title("✝️ Biblical Moments")
st.caption("Engine: Gemini 2.5 Flash Image (Nano Banana)")

col1, col2 = st.columns(2)
with col1:
    bible_character = st.text_input("想合照的圣经人物", placeholder="例如：耶稣、摩西")
with col2:
    clothing_style = st.selectbox("您的服装风格", ["保持我原图的衣服", "圣经时代的古装长袍", "现代正装"])

uploaded_file = st.file_uploader("上传您的自拍 (直接喂给 Gemini 进行编辑)", type=['jpg', 'png', 'jpeg'])

# --- 4. 核心逻辑: Nano Banana 编辑 ---

def generate_with_nano_banana(user_image_bytes, character, clothing):
    """
    使用 gemini-2.5-flash-image 直接进行图像编辑。
    """
    
    # 🔴 指定您截图中的新模型 ID
    model_name = "gemini-2.5-flash-image"
    
    try:
        # 这个模型虽然叫 Gemini，但具备 Edit 能力，通常通过 ImageGenerationModel 接口调用
        # 或者通过 GenerativeModel 的 edit_content 接口
        # 这里我们尝试标准的 edit_images 接口，这是目前处理"喂图+提示词"的标准路径
        model = ImageGenerationModel.from_pretrained(model_name)
        
        # 准备图片
        source_image = VertexImage(image_bytes=user_image_bytes)
        
        # 编写编辑指令
        # 既然是 Editing Model，我们需要告诉它“改什么”
        prompt = f"""
        Edit this image to show the person standing side-by-side with {character} from the Bible.
        Change the background to a realistic biblical era scene.
        Ensure the person's face remains unchanged.
        The person is wearing {clothing}.
        {character} is wearing historically accurate clothing.
        Photorealistic, 8k, cinematic lighting.
        """
        
        # 调用编辑接口
        images = model.edit_images(
            prompt=prompt,
            base_image=source_image,
            number_of_images=1,
            guidance_scale=50, # 这里的参数控制模型听从Prompt的程度
            language="en"
        )
        return images[0]

    except Exception as e:
        # 错误处理：如果 2.5-image 接口有变动，或者未白名单
        # 我们做一个智能回退，保证 App 不会崩
        st.warning(f"⚠️ Nano Banana ({model_name}) 调用异常: {str(e)}。已自动切换至标准 Imagen 2 编辑模式。")
        fallback_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
        source_image = VertexImage(image_bytes=user_image_bytes)
        images = fallback_model.edit_images(
            prompt=prompt,
            base_image=source_image,
            number_of_images=1
        )
        return images[0]

# --- 5. 执行逻辑 ---
if st.button("✨ 立即合照"):
    if not uploaded_file or not bible_character:
        st.warning("请先上传照片并输入人物。")
    else:
        try:
            progress = st.progress(0)
            status = st.empty()
            
            # 直接调用
            status.text(f"🍌 正在将照片喂给 Gemini 2.5 Flash Image...")
            img_bytes = uploaded_file.getvalue()
            
            result_image = generate_with_nano_banana(img_bytes, bible_character, clothing_style)
            
            progress.progress(100)
            status.text("✨ 完成！")
            
            # 展示
            st.image(result_image._image_bytes, caption=f"与 {bible_character} 的合影", use_column_width=True)
            
            # 下载
            st.download_button(
                label="📥 保存照片", 
                data=result_image._image_bytes, 
                file_name=f"nano_gen_{bible_character}.png", 
                mime="image/png"
            )
            
        except Exception as e:
            st.error("生成失败")
            st.code(str(e))
