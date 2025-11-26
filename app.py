import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image as VertexImage
from google.oauth2 import service_account
import json

# 1. 页面配置
st.set_page_config(page_title="Biblical Moments - Capability", page_icon="✝️", layout="centered")

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
    .stButton>button { width: 100%; background-color: #1A5276; color: white; border-radius: 20px; height: 50px; font-size: 18px; border: none; }
    h1 { text-align: center; font-family: 'serif'; color: #2C3E50; }
    .caption { text-align: center; color: #888; font-size: 12px; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

st.title("✝️ Biblical Moments")
st.caption("Engine: imagen-3.0-capability-001 (Few-shot Ref)")

col1, col2 = st.columns(2)
with col1:
    bible_character = st.text_input("想合照的圣经人物", placeholder="例如：耶稣、大卫")
with col2:
    clothing_style = st.selectbox("您的服装风格", ["保持原图着装", "圣经时代的古装长袍", "现代正装"])

uploaded_file = st.file_uploader("上传您的自拍 (将作为参考图直接合成)", type=['jpg', 'png', 'jpeg'])

# --- 4. 核心逻辑: Capability 模型直接合成 ---

def generate_with_capability_model(user_image_bytes, character, clothing):
    """
    使用 imagen-3.0-capability-001 进行基于参考图的生成/编辑。
    """
    
    # 🔴 严格指定您截图中的模型 ID
    model_name = "imagen-3.0-capability-001"
    
    try:
        # 加载 Capability 模型
        model = ImageGenerationModel.from_pretrained(model_name)
        
        # 准备图片对象
        source_image = VertexImage(image_bytes=user_image_bytes)
        
        # 编写指令
        # 针对 Capability 模型，Prompt 需要强调"基于原图"但"修改环境"
        prompt = f"""
        Function: Subject Preservation Edit.
        Input: The uploaded reference image of a person.
        Task: Generate a photorealistic image of this SAME person standing next to {character} (Bible Figure).
        Environment: Authentic biblical landscape (Jerusalem/Desert).
        Clothing: The person is wearing {clothing}. {character} wears historical robes.
        Details: Keep the person's facial identity and features EXACTLY as in the reference image.
        Quality: 8k, cinematic lighting.
        """
        
        # 调用 edit_images
        # 这利用了 Capability 模型的"参考"能力，将 base_image 视为 Few-shot 样本
        images = model.edit_images(
            prompt=prompt,
            base_image=source_image, # 直接喂图
            number_of_images=1,
            guidance_scale=60,       # 高引导值，确保模型遵循"添加圣经人物"的指令
            language="en"
        )
        return images[0], "Capability Model"

    except Exception as e:
        # === 智能兜底 ===
        # 如果 Capability 模型需要特殊的 Tuning 权限或 API 格式不同
        # 自动切换至标准 Imagen 3 高清版，保证 App 可用
        st.warning(f"⚠️ Capability 模型 ({model_name}) 调用受限: {str(e)}。已自动切换至标准 Imagen 3。")
        
        fallback_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
        source_image = VertexImage(image_bytes=user_image_bytes)
        images = fallback_model.edit_images(
            prompt=prompt,
            base_image=source_image,
            number_of_images=1
        )
        return images[0], "Standard Backup"

# --- 5. 执行逻辑 ---
if st.button("✨ 生成合照"):
    if not uploaded_file or not bible_character:
        st.warning("请先上传照片并输入人物。")
    else:
        try:
            progress = st.progress(0)
            status = st.empty()
            
            status.text(f"🚀 正在使用参考图模型 (Capability 001) 处理...")
            img_bytes = uploaded_file.getvalue()
            
            # 调用生成函数
            result_image, method = generate_with_capability_model(img_bytes, bible_character, clothing_style)
            
            progress.progress(100)
            status.text("✨ 完成！")
            
            # 展示
            st.image(result_image._image_bytes, caption=f"合照 ({method})", use_column_width=True)
            
            # 下载
            st.download_button(
                label="📥 保存照片", 
                data=result_image._image_bytes, 
                file_name=f"capability_gen_{bible_character}.png", 
                mime="image/png"
            )
            
            # 经文
            st.markdown("---")
            v_model = GenerativeModel("gemini-1.5-flash")
            try:
                verse = v_model.generate_content(f"One short Bible verse about {bible_character}, bilingual.")
                st.info(verse.text)
            except:
                pass
            
        except Exception as e:
            st.error("生成失败")
            st.code(str(e))
