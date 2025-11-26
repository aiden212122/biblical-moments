import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image as VertexImage
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account
import json
import io

# 1. 页面配置
st.set_page_config(page_title="Biblical Moments - 圣经合影", page_icon="✝️", layout="centered")

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
                    st.error("❌ Secrets 格式严重错误。")
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

# --- 3. 样式美化 ---
st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #D4AF37; color: white; border-radius: 20px; height: 50px; font-size: 18px; border: none; }
    h1 { text-align: center; font-family: 'serif'; color: #2C3E50; }
    .caption { text-align: center; color: #888; font-size: 12px; margin-top: 20px; }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("✝️ Biblical Moments")
st.write("上传您的照片，AI 将保留您的样貌，生成与圣经人物的合影。")

col1, col2 = st.columns(2)
with col1:
    bible_character = st.text_input("想合照的圣经人物", placeholder="例如：耶稣、摩西、彼得")
with col2:
    clothing_style = st.selectbox("您的服装风格", ["保持我照片里的衣服", "圣经时代的古装长袍", "现代休闲装", "正式礼服"])

uploaded_file = st.file_uploader("上传您的自拍/半身照 (尽量正面，光线清晰)", type=['jpg', 'png', 'jpeg'])

# --- 4. 核心 AI 逻辑 (升级版) ---

def get_gemini_analysis(user_image_bytes):
    """
    使用 Gemini 2.5/1.5 Flash 提取面部特征。
    这一步是为了给 Imagen 提供文字辅助，确保“像”上加“像”。
    """
    try:
        model = GenerativeModel("gemini-2.5-flash") # 尝试最新模型
    except:
        model = GenerativeModel("gemini-1.5-flash")

    image_part = Part.from_data(data=user_image_bytes, mime_type="image/jpeg")
    
    # 指令：只提取面部特征，越详细越好
    prompt = """
    Analyze the person in this image. Describe ONLY their physical appearance in extreme detail for an image generator prompt:
    - Gender, Age, Ethnicity, Skin tone.
    - Exact Hair style, hair color, beard/facial hair.
    - Specific facial features (eye shape, nose shape, smile).
    - Glasses or accessories if any.
    Output just the description text.
    """
    
    try:
        response = model.generate_content([image_part, prompt])
        return response.text
    except:
        return "A person" # 降级处理

def generate_with_identity_preservation(user_image_bytes, character, clothing, user_description):
    """
    核心升级：使用 edit_images 接口，将原图作为 base_image 喂给模型。
    这样模型是基于原图进行“修改/重绘”，而不是凭空创造，从而最大程度保留五官。
    """
    # 注意：目前 Imagen 的 edit_images 功能在 imagegeneration@006 (Imagen 2) 上最稳定可用
    # Imagen 3 的编辑功能 API 尚未完全对所有项目开放，因此这里使用 006 以确保代码不报错
    model_name = "imagegeneration@006"
    
    try:
        model = ImageGenerationModel.from_pretrained(model_name)
        
        # 1. 将上传的字节流转换为 Vertex AI Image 对象
        source_image = VertexImage(image_bytes=user_image_bytes)
        
        # 2. 构建“编辑”提示词
        # 我们告诉模型：保持这个人不变，但是把背景换成圣经场景，并在旁边加上圣经人物
        full_prompt = f"""
        A photorealistic shot of {user_description} standing side-by-side with {character} from the Bible.
        The user is wearing {clothing}.
        {character} is wearing historically accurate clothing from the biblical era.
        Background is a realistic scene from ancient Israel/Judea.
        Cinematic lighting, 8k resolution.
        IMPORTANT: Keep the facial features of the person from the original image exactly as they are.
        """
        
        # 3. 调用 edit_images (图生图)
        # 不传 mask 参数时，模型会尝试基于全图进行调整 (Image-to-Image / Variation)
        images = model.edit_images(
            prompt=full_prompt,
            base_image=source_image,
            number_of_images=1,
            language="en",
            # guidance_scale 控制对 Prompt 的遵循程度，21-60 之间通常比较好
            guidance_scale=60, 
            safety_filter_level="block_some",
            person_generation="allow_adult"
        )
        return images[0]
        
    except Exception as e:
        raise RuntimeError(f"图生图模型调用失败: {e}")

# --- 5. 执行逻辑 ---
if st.button("✨ 生成合照 (保真模式)"):
    if not uploaded_file or not bible_character:
        st.warning("请先上传照片并输入人物。")
    else:
        try:
            progress = st.progress(0)
            status = st.empty()
            
            # 读取图片
            img_bytes = uploaded_file.getvalue()
            
            # 第一步：Gemini 分析面部 (30%)
            status.text("👀 正在分析您的五官特征...")
            user_desc = get_gemini_analysis(img_bytes)
            progress.progress(30)
            # st.caption(f"识别到的特征: {user_desc[:50]}...") # 调试用
            
            # 第二步：Imagen 图生图生成 (100%)
            status.text(f"🎨 正在保留您的肖像并邀请 {bible_character} 入镜...")
            result = generate_with_identity_preservation(img_bytes, bible_character, clothing_style, user_desc)
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
            verse = v_model.generate_content(f"给我一句关于'{bible_character}'的圣经经文(中英对照)，简短有力。")
            st.info(verse.text)
            
        except Exception as e:
            st.error("生成出错")
            with st.expander("查看技术详情"):
                st.code(str(e))

st.markdown("<p class='caption'>Powered by Google Vertex AI (Gemini Flash + Imagen 2 Edit)</p>", unsafe_allow_html=True)
