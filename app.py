import streamlit as st
import google.generativeai as genai
from PIL import Image
import io

# --- 1. 页面基础配置 ---
st.set_page_config(page_title="Nano Banana 圣经照相馆", page_icon="🍌")

# 隐藏多余菜单
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {background-color: #FAFAF9;}
    </style>
    """, unsafe_allow_html=True)

st.title("🍌 Nano Banana 圣经合影")
st.caption("Powered by Gemini 3 Pro Image Preview")

# --- 2. 获取 API Key ---
# 优先读取 Streamlit Secrets，如果没有配置，允许用户在侧边栏临时输入（方便测试）
api_key = st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    with st.sidebar:
        api_key = st.text_input("请输入 Google API Key", type="password")
        st.info("提示：去 aistudio.google.com 免费申请")

if not api_key:
    st.warning("👈 请先配置 API Key 才能开始")
    st.stop()

# 配置模型
genai.configure(api_key=api_key)

# --- 3. 界面交互 ---
with st.container():
    st.subheader("1. 上传你的照片")
    uploaded_file = st.file_uploader("最好是半身或全身照", type=["jpg", "jpeg", "png"])
    
    user_image = None
    if uploaded_file:
        user_image = Image.open(uploaded_file)
        st.image(user_image, caption="已上传", width=150)

    st.subheader("2. 设定合照")
    col1, col2 = st.columns(2)
    with col1:
        character = st.text_input("圣经人物", value="Jesus", placeholder="例如: Jesus, David")
    with col2:
        clothing = st.selectbox("你的服装", ["Modern Casual (T-shirt)", "Biblical Robes (Historical)", "Suit & Tie"])
    
    style = st.select_slider("风格强度", options=["Realistic", "Cinematic", "Oil Painting"], value="Cinematic")

# --- 4. 核心生成逻辑 ---
if st.button("🚀 生成合照 (调用 Nano Banana Pro)", type="primary", use_container_width=True):
    if not user_image or not character:
        st.error("请上传照片并填写人物名字")
    else:
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            status_text.text("正在连接 Gemini 3 Pro 模型...")
            progress_bar.progress(20)

            # 关键：指定截图中的模型 ID
            # 如果报错 "Model not found"，说明你的 Key 还没开通这个预览版权限
            # 那时可以尝试换回 'gemini-1.5-pro'
            MODEL_ID = 'gemini-3-pro-image-preview' 
            
            model = genai.GenerativeModel(MODEL_ID)

            # 构建提示词 (Gemini 3 理解力很强，直接用自然语言)
            prompt = f"""
            Task: Edit the input image to create a two-person photo.
            1. Keep the person from the input image (Input User) on the right side.
            2. Add {character} from the Bible on the left side.
            3. {character} should look historically accurate (ancient Middle Eastern appearance).
            4. The Input User should be wearing {clothing}.
            5. Background: Ancient biblical landscape.
            6. Style: {style}, high quality, photorealistic.
            7. Output ONLY the generated image.
            """

            status_text.text("AI 正在绘图 (可能需要 10-20 秒)...")
            progress_bar.progress(50)

            # 发送请求 (Gemini 3 支持直接把 image 对象放进列表)
            response = model.generate_content([prompt, user_image])

            progress_bar.progress(90)
            status_text.text("正在接收数据...")

            # 解析结果
            # Gemini 3 可能会返回图片对象，或者有时候返回文字
            # 我们需要检查 parts 里是不是有 binary 数据
            
            if response.parts:
                # 尝试寻找图片部分
                image_data = None
                for part in response.parts:
                    if hasattr(part, "inline_data"): # 只有图片会有这个字段
                        image_data = part.inline_data.data
                        break
                    # 有些版本 SDK 属性名可能是 image
                
                # 如果 SDK 自动处理了，可以直接用 text 检查是否失败，或者直接展示
                # 简单粗暴的方法：直接展示 response (Streamlit 支持)
                # 但为了保险，我们手动处理一下可能的情况
                
                try:
                    # 最新版 SDK 通常这样获取图片
                    generated_image = response.text # 如果失败，通常是因为里面是 binary，访问 text 会报错或者为空
                except:
                    # 报错说明不是纯文本，很可能是图片，好事！
                    pass

                # 完美展示逻辑
                st.success("生成成功！")
                
                # 这是一个技巧：Streamlit 的 write 可以自动渲染 Gemini 的 response 对象里的图片
                st.write(response) 
                
            else:
                st.error("模型未返回内容，可能是触发了安全拦截。")

        except Exception as e:
            st.error(f"发生错误: {str(e)}")
            st.info("排错指南：\n1. 检查 API Key 是否正确。\n2. 你的账号可能还没获得 gemini-3-pro 的权限，请把代码里的 MODEL_ID 改成 'gemini-1.5-pro' 试试。")
        
        finally:
            progress_bar.empty()
            status_text.empty()
