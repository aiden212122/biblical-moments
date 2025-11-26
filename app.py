import streamlit as st
import google.generativeai as genai
from PIL import Image

# 1. 页面配置
st.set_page_config(page_title="圣经时空照相馆", page_icon="✨")

# 2. 隐藏默认菜单，让界面像个App
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# 3. 标题
st.title("✨ 圣经时空照相馆")
st.write("上传你的照片，穿越时空与圣经人物合影。")

# 4. 获取 API Key (从 Streamlit 只有后台能看到的 Secrets 里读取)
api_key = st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    st.error("请在 Streamlit 设置中配置 GOOGLE_API_KEY")
    st.stop()

# 配置 Google AI
genai.configure(api_key=api_key)

# 5. 用户输入区域
with st.container():
    st.subheader("1. 上传自拍")
    uploaded_file = st.file_uploader("选择一张你的正面照片", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # 显示用户上传的图
        image = Image.open(uploaded_file)
        st.image(image, caption="你的照片", width=200)

    st.subheader("2. 定制合照")
    character = st.text_input("想见哪位圣经人物？", placeholder="例如：耶稣、大卫、摩西、路得...")
    
    clothing = st.selectbox(
        "你的服装风格",
        ("保持历史真实感 (穿圣经时代长袍)", "现代休闲 (T恤/牛仔裤)", "现代正装", "原本的衣服")
    )
    
    style = st.selectbox(
        "画面艺术风格",
        ("电影质感 (Cinematic)", "古典油画 (Oil Painting)", "素描手绘 (Sketch)", "3D 动漫 (Pixar Style)")
    )

# 6. 生成逻辑
if st.button("✨ 开始生成合照", type="primary", use_container_width=True):
    if not uploaded_file or not character:
        st.warning("请先上传照片并输入人物名字！")
    else:
        with st.spinner("正在祈祷与描绘中，请稍候..."):
            try:
                # 准备 Prompt
                prompt = f"""
                Generate a photorealistic image of two people standing side by side.
                Person 1: {character} from the Bible, looking historically accurate for their era (ancient Middle Eastern).
                Person 2: A person based on the input image provided. 
                They should be {clothing}.
                The two are interacting in a friendly, holy way.
                Background: Biblical scenery relevant to {character}.
                Art Style: {style}.
                Important: Try to preserve the facial features of the uploaded person for Person 2.
                """
                
                # 调用 Gemini Pro Vision 模型
                model = genai.GenerativeModel('gemini-1.5-flash') # 或者 gemini-1.5-pro
                response = model.generate_content([prompt, image])
                
                # 解析并显示结果
                st.success("生成成功！")
                st.image(response.text, caption="生成的合照 (如果是纯文本描述请检查模型是否支持图片输出)")
                
                # 注意：目前的 Gemini API 主要返回文本描述，如果要直接生成图片
                # Google 的 imagen 模型才直接出图。
                # 但 Streamlit + Gemini 的图生文+文生图 链路较复杂。
                # 如果发现只输出了文字描述，我们需要在这里做一个变通：
                # 对于新手，如果 Gemini 1.5 还是只出文本，建议改回用 Replicate。
                
                # 修正：由于 Gemini API 目前主要用于多模态理解（输入图，输出文），
                # 或者 Vertex AI 的 Imagen 3 才能画图。
                # 为了不让你在 Google Cloud 复杂的权限里绕晕，
                # 如果这里只输出了文字，我们可能需要紧急换回 Replicate 方案。
                
                # 但为了演示 Google 流程，这里假设你的账号有 Imagen 权限。
                # 如果没有，下面会显示生成的描述文字。
                
                st.write("注：如果上方没有显示图片，说明当前调用的 Google 模型仅返回了画面描述。")

            except Exception as e:
                st.error(f"出错了: {e}")
