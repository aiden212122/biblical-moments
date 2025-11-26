import streamlit as st
import google.generativeai as genai
from PIL import Image
import io

# --- 1. 页面基础配置 ---
st.set_page_config(page_title="Nano Banana 圣经照相馆", page_icon="🍌")

# 美化界面：隐藏多余菜单
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {background-color: #FAFAF9;}
    /* 调整一下按钮样式 */
    div.stButton > button:first-child {
        background-color: #F59E0B;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🍌 Nano Banana 圣经合影")
st.caption("Powered by Google Gemini 3 Pro (Image Preview)")

# --- 2. 获取 API Key ---
# 优先读取 Streamlit Secrets
api_key = st.secrets.get("GOOGLE_API_KEY")

# 如果没配置，允许侧边栏输入（方便调试）
if not api_key:
    with st.sidebar:
        api_key = st.text_input("请输入 API Key", type="password")
        st.info("建议将 Key 配置在 Streamlit Secrets 中以保证安全。")

if not api_key:
    st.warning("👈 请先配置 API Key 才能开始")
    st.stop()

# 配置 Google AI
genai.configure(api_key=api_key)

# --- 3. 界面交互 ---
with st.container():
    st.subheader("1. 上传照片")
    uploaded_file = st.file_uploader("请上传正面清晰照", type=["jpg", "jpeg", "png"])
    
    user_image = None
    if uploaded_file:
        # 读取并展示用户图片
        user_image = Image.open(uploaded_file)
        st.image(user_image, caption="已上传", width=150)

    st.subheader("2. 设定合照")
    col1, col2 = st.columns(2)
    with col1:
        character = st.text_input("圣经人物", value="Jesus", placeholder="例如: Jesus, David, Moses")
    with col2:
        clothing = st.selectbox("你的服装", [
            "Biblical Robes (Historical) - 圣经时代长袍", 
            "Modern Casual (T-shirt) - 现代休闲", 
            "Suit & Tie - 正装"
        ])
    
    # 提取服装的英文描述，发给 AI
    clothing_prompt = clothing.split(" - ")[0]
    
    style = st.select_slider("风格强度", options=["Realistic", "Cinematic", "Oil Painting"], value="Cinematic")

# --- 4. 核心生成逻辑 (修复版) ---
if st.button("🚀 生成合照", type="primary", use_container_width=True):
    if not user_image or not character:
        st.error("❌ 请先上传照片并填写人物名字！")
    else:
        status_text = st.empty()
        bar = st.progress(0)
        
        try:
            status_text.text("正在连接 Gemini 3 Pro...")
            bar.progress(20)

            # 指定模型 ID (截图中的模型)
            MODEL_ID = 'gemini-3-pro-image-preview'
            model = genai.GenerativeModel(MODEL_ID)

            # 构建提示词
            prompt = f"""
            Task: Edit the input image to create a two-person photo.
            1. Keep the user from the input image on the right.
            2. Add {character} from the Bible on the left.
            3. {character} must look historically accurate (ancient Middle Eastern appearance).
            4. The user should be wearing {clothing_prompt}.
            5. Background: Ancient biblical landscape.
            6. Style: {style}, 8k resolution, photorealistic.
            7. Output Format: IMAGE ONLY. Do not describe the image, just generate it.
            """

            bar.progress(50)
            status_text.text("AI 正在绘图，请稍候 (约10-20秒)...")

            # 发送请求
            response = model.generate_content([prompt, user_image])
            
            bar.progress(90)
            status_text.text("正在解析数据...")

            # --- 核心修复：解析图片数据 ---
            # 这里的逻辑专门处理你截图里的 GenerateContentResponse 结构
            
            image_generated = False

            if response.candidates:
                for candidate in response.candidates:
                    for part in candidate.content.parts:
                        # 检查是否有二进制图片数据 (inline_data)
                        if part.inline_data:
                            try:
                                # 1. 获取 bytes 数据
                                img_bytes = part.inline_data.data
                                # 2. 转换为 PIL Image 对象
                                final_image = Image.open(io.BytesIO(img_bytes))
                                
                                # 3. 展示成功界面
                                st.balloons() # 撒花庆祝
                                st.success("✨ 合照生成成功！")
                                st.image(final_image, caption=f"我和 {character} 的合照", use_column_width=True)
                                image_generated = True
                                break # 找到图了就退出循环
                            except Exception as img_err:
                                st.error(f"解析图片失败: {img_err}")
                        
                        # 如果没有图片，检查是不是返回了文字
                        elif part.text:
                            # 有时候模型还是会忍不住说话
                            print("Model Text Response:", part.text)

            if not image_generated:
                st.warning("⚠️ 生成完成，但未检测到图片。")
                st.write("可能原因：")
                st.write("1. 模型认为图片内容敏感（Google安全过滤非常严格）。")
                st.write("2. API Key 权限不足。")
                # 打印出 AI 到底说了什么，方便调试
                if response.text:
                    st.info(f"AI 回复内容: {response.text}")

        except Exception as e:
            st.error(f"发生错误: {str(e)}")
            st.info("提示：如果是 404 Not Found，请将代码中的 MODEL_ID 改为 'gemini-1.5-flash' 再试。")
        
        finally:
            bar.empty()
            status_text.empty()
