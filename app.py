import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, Image as VertexImage
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account
import json
import io
from PIL import Image, ImageOps

# 1. 页面配置
st.set_page_config(page_title="Biblical Moments - 真人合影版", page_icon="✝️", layout="centered")

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

# --- 3. 样式 ---
st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #D4AF37; color: white; border-radius: 20px; height: 50px; font-size: 18px; border: none; }
    h1 { text-align: center; font-family: 'serif'; color: #2C3E50; }
    .stSelectbox label { font-size: 16px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("✝️ Biblical Moments (真人合影版)")
st.write("上传您的照片，AI 将在您身旁“变”出圣经人物，**保留您的真实样貌**。")

# --- 4. 辅助函数：处理图像与蒙版 ---

def process_image_and_mask(uploaded_file, position="left"):
    """
    1. 读取用户图片
    2. 调整大小以符合模型要求
    3. 生成蒙版：保留一边，另一边涂白（让AI重绘）
    """
    # 读取原始图片
    original_image = Image.open(uploaded_file).convert("RGB")
    
    # 调整大小 (Imagen 编辑模式建议不超过 1MB 且为标准比例，这里统一缩放到 1024x1024 以内)
    original_image.thumbnail((1024, 1024))
    
    width, height = original_image.size
    
    # 创建蒙版 (L mode, 0=黑=保留原图, 255=白=重绘区域)
    # 默认全黑（保留原图）
    mask = Image.new("L", (width, height), 0)
    
    # 计算遮罩区域 (假设占宽度的 40%)
    mask_width = int(width * 0.45) 
    
    if position == "圣经人物在左边":
        # 把左边涂白 (AI 将在左边画人)
        mask.paste(255, (0, 0, mask_width, height))
    else:
        # 把右边涂白 (AI 将在右边画人)
        mask.paste(255, (width - mask_width, 0, width, height))
        
    return original_image, mask

def pil_to_bytes(pil_img):
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

# --- 5. AI 核心功能 ---

def get_gemini_prompt(user_image_bytes, character, position):
    """使用 Gemini 分析图片并写提示词，强调位置关系"""
    # 优先尝试 2.5-flash，失败回退
    target_model = "gemini-2.5-flash"
    try:
        model = GenerativeModel(target_model)
    except:
        model = GenerativeModel("gemini-1.5-flash")
    
    image_part = Part.from_data(data=user_image_bytes, mime_type="image/jpeg")
    
    pos_desc = "on the left" if position == "圣经人物在左边" else "on the right"
    
    prompt_instruction = f"""
    You are an expert art director.
    TASK: Look at the uploaded photo. Describe the environment and lighting briefly.
    GOAL: Write a prompt to EDIT this photo. We want to ADD {character} from the Bible {pos_desc} of the user.
    
    CRITICAL DETAILS:
    1. Keep the user (the real person) exactly as they are (do not mention changing the user).
    2. Describe {character} accurately (clothing, era, appearance).
    3. Ensure {character} is interacting naturally (standing next to, looking at, or walking with).
    4. Match the lighting and style of the original photo so the blend is seamless.
    
    OUTPUT: Just the prompt text for the image editor.
    """
    
    try:
        response = model.generate_content([image_part, prompt_instruction])
        return response.text
    except:
        # 回退逻辑
        fallback = GenerativeModel("gemini-1.5-flash")
        response = fallback.generate_content([image_part, prompt_instruction])
        return response.text

def edit_image_with_model(base_img_pil, mask_img_pil, prompt):
    """
    调用 Vertex AI 的 edit_images 接口
    """
    # 将 PIL 图片转为 Vertex AI 需要的 Bytes
    base_bytes = pil_to_bytes(base_img_pil)
    mask_bytes = pil_to_bytes(mask_img_pil)
    
    base_img_vertex = VertexImage(base_bytes)
    mask_img_vertex = VertexImage(mask_bytes)
    
    # 使用支持编辑的模型
    # imagegeneration@006 (Imagen 2/3) 支持编辑
    # imagen-3.0-capability-001 可能支持，但 @006 最稳
    model_name = "imagegeneration@006" 
    
    try:
        model = ImageGenerationModel.from_pretrained(model_name)
        
        # 调用编辑接口
        images = model.edit_images(
            base_image=base_img_vertex,
            mask=mask_img_vertex,
            prompt=prompt,
            number_of_images=1,
            language="en",
            # product_mode=False # 这是一个通用场景，非商品图
        )
        return images[0]
    except Exception as e:
        raise RuntimeError(f"编辑失败: {e}")

# --- 6. UI 交互 ---

col1, col2 = st.columns(2)
with col1:
    bible_character = st.text_input("想合照的圣经人物", placeholder="例如：耶稣、大卫")
with col2:
    # 让用户选择合成位置，这决定了哪里被遮罩
    edit_position = st.selectbox("圣经人物出现的位置", ["圣经人物在右边", "圣经人物在左边"])

uploaded_file = st.file_uploader("上传您的自拍 (请确保您在画面的一侧，留出空位)", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # 预览蒙版区域，让用户知道哪里会被重绘
    img_preview, mask_preview = process_image_and_mask(uploaded_file, edit_position)
    
    st.caption("📷 预览 (红色区域将被 AI 重绘为圣经人物，其他区域保留原样):")
    
    # 合成一个预览图显示蒙版区域
    overlay = Image.new('RGB', img_preview.size, (255, 0, 0))
    preview_comp = Image.composite(overlay, img_preview, mask_preview)
    st.image(preview_comp, width=300)

if st.button("✨ 开始合成 (保留原脸)"):
    if not uploaded_file or not bible_character:
        st.warning("请上传照片并输入人物名字")
    else:
        try:
            progress = st.progress(0)
            status = st.empty()
            
            # 1. 准备图片和蒙版
            status.text("✂️ 正在处理图片和蒙版...")
            base_pil, mask_pil = process_image_and_mask(uploaded_file, edit_position)
            base_bytes = pil_to_bytes(base_pil) # 用于 Gemini 分析
            progress.progress(20)
            
            # 2. Gemini 写提示词
            status.text("🧠 Gemini 正在观察环境并构思...")
            prompt = get_gemini_prompt(base_bytes, bible_character, edit_position)
            # st.info(f"提示词: {prompt}") # 调试用
            progress.progress(50)
            
            # 3. Imagen 编辑/重绘
            status.text(f"🎨 正在将 {bible_character} 绘制到照片中...")
            result = edit_image_with_model(base_pil, mask_pil, prompt)
            progress.progress(100)
            status.text("✨ 完成！")
            
            # 展示
            st.image(result._image_bytes, caption=f"您与 {bible_character} 的跨时空合影", use_column_width=True)
            
            # 下载
            st.download_button(
                label="📥 保存照片", 
                data=result._image_bytes, 
                file_name=f"with_{bible_character}_real.png", 
                mime="image/png"
            )
            
            # 经文
            st.markdown("---")
            v_model = GenerativeModel("gemini-1.5-flash")
            verse = v_model.generate_content(f"给我一句关于'{bible_character}'的经文(中英对照)。")
            st.info(verse.text)
            
        except Exception as e:
            st.error("生成出错")
            st.expander("详情").write(e)
