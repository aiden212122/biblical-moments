import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image as VertexImage
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account
import json

# 1. 页面配置
st.set_page_config(page_title="Biblical Moments - Ultra", page_icon="✝️", layout="centered")

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
    .caption { text-align: center; color: #888; font-size: 12px; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

st.title("✝️ Biblical Moments")
st.caption("Core: Auto-Switch (Direct Edit ➡️ Visual Synthesis)")

col1, col2 = st.columns(2)
with col1:
    bible_character = st.text_input("想合照的圣经人物", placeholder="例如：耶稣、大卫")
with col2:
    clothing_style = st.selectbox("您的服装风格", ["保持原样", "圣经时代长袍", "现代正装"])

uploaded_file = st.file_uploader("上传您的自拍", type=['jpg', 'png', 'jpeg'])

# --- 4. 核心功能：双模态引擎 ---

def generate_smart_hybrid(user_image_bytes, character, clothing):
    """
    智能混合生成器：
    1. 优先尝试直接图片编辑 (Direct Edit)。
    2. 如果失败 (AttributeError/不支持)，自动降级为 Gemini 2.5 + Imagen 4.0 (Visual Synthesis)。
    """
    status_box = st.empty()
    
    # === 尝试 1: 直接图片编辑 (最符合您需求的模式) ===
    try:
        status_box.text("🚀 尝试 Mode A: 直接图片融合...")
        
        # 目前最稳定的编辑模型是 Imagen 2
        model = ImageGenerationModel.from_pretrained("imagegeneration@006")
        source_image = VertexImage(image_bytes=user_image_bytes)
        
        prompt = f"""
        A photorealistic shot of the person in this image standing next to {character} (Bible Figure).
        Background: Ancient biblical landscape.
        User clothing: {clothing}.
        High quality, 8k.
        """
        
        # 核心：尝试调用 edit_images
        # 如果库版本旧，这里会直接抛出 AttributeError，触发下方的 except
        images = model.edit_images(
            prompt=prompt,
            base_image=source_image,
            number_of_images=1,
            guidance_scale=60
        )
        status_box.text("✅ Mode A 成功！")
        return images[0], "Direct Edit"

    except (AttributeError, Exception) as e:
        # === 尝试 2: 视觉合成 (兜底模式) ===
        # 当直接编辑失败时，我们不报错，而是立刻切换到 Gemini 2.5 + Imagen 4.0
        # 这能保证用户 100% 拿到结果。
        
        print(f"Direct edit failed: {e}. Switching to fallback.")
        status_box.text(f"⚠️ 直接融合不可用，自动切换 Mode B: Gemini 2.5 + Imagen 4.0...")
        
        # 2.1: Gemini 2.5 视觉分析
        try:
            gemini_model = GenerativeModel("gemini-2.5-flash-preview-09-2025") # 优先用最新
        except:
            gemini_model = GenerativeModel("gemini-1.5-flash")

        image_part = Part.from_data(data=user_image_bytes, mime_type="image/jpeg")
        
        analysis_prompt = f"""
        Analyze the person in this image. Write a detailed physical description for an image generator prompt:
        - Ethnicity, Face shape, Age, Skin tone.
        - Exact Hair style & color, Facial features.
        Output ONLY the description.
        """
        
        try:
            desc_response = gemini_model.generate_content([image_part, analysis_prompt])
            user_desc = desc_response.text
        except:
            user_desc = "A person"

        # 2.2: Imagen 4.0 生成
        gen_model_name = "imagen-4.0-generate-001"
        try:
            gen_model = ImageGenerationModel.from_pretrained(gen_model_name)
            
            final_prompt = f"""
            A photorealistic photo of {user_desc} standing side-by-side with {character} (Bible Character).
            Scene: Biblical era, holy atmosphere.
            User clothing: {clothing}.
            Quality: 8k, cinematic.
            """
            
            images = gen_model.generate_images(prompt=final_prompt, number_of_images=1, aspect_ratio="3:4")
            status_box.text("✅ Mode B 成功！(Imagen 4.0)")
            return images[0], "Visual Synthesis"
            
        except Exception as final_e:
            # 如果连 Imagen 4 都挂了，最后尝试 Imagen 3
            fallback_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
            images = fallback_model.generate_images(prompt=final_prompt, number_of_images=1, aspect_ratio="3:4")
            return images[0], "Backup Gen"

# --- 5. 执行逻辑 ---
if st.button("✨ 开始合成"):
    if not uploaded_file or not bible_character:
        st.warning("请先上传照片并输入人物。")
    else:
        try:
            progress = st.progress(0)
            
            img_bytes = uploaded_file.getvalue()
            
            # 调用智能混合函数
            result_image, method_used = generate_smart_hybrid(img_bytes, bible_character, clothing_style)
            
            progress.progress(100)
            
            # 展示结果
            st.image(result_image._image_bytes, caption=f"合影完成 ({method_used})", use_column_width=True)
            
            # 下载
            st.download_button(
                label="📥 保存照片", 
                data=result_image._image_bytes, 
                file_name=f"bible_moment_{bible_character}.png", 
                mime="image/png"
            )
            
            st.markdown("---")
            st.info(f"技术说明：本次生成使用了 {method_used} 模式。")
            
        except Exception as e:
            st.error("生成过程发生严重错误，请检查网络或配额。")
            st.code(str(e))
