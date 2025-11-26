import React, { useState, useEffect } from 'react';
import { initializeApp } from 'firebase/app';
import { getAuth, signInAnonymously, onAuthStateChanged } from 'firebase/auth';

// ⚠️ 配置区域：请在此处填入您的真实信息
// 在实际生产中，API Key 应该放在环境变量中 (.env)
const GOOGLE_API_KEY = process.env.REACT_APP_GOOGLE_API_KEY || "在此处粘贴您的_GOOGLE_API_KEY";

// Firebase 配置 (如果没有，可以留空，Auth 功能将失效但不影响生图)
const firebaseConfig = {
    apiKey: "YOUR_FIREBASE_API_KEY",
    authDomain: "YOUR_PROJECT.firebaseapp.com",
    projectId: "YOUR_PROJECT_ID",
    storageBucket: "YOUR_PROJECT.appspot.com",
    messagingSenderId: "...",
    appId: "..."
};

const CLOTHING_CHOICES = [
    { label: "保持我照片中的原样服饰", value: "keep_original" },
    { label: "换成该圣经人物时代的服饰 (长袍/麻衣等)", value: "biblical_era_clothing" },
    { label: "换成现代休闲服", value: "modern_casual" },
    { label: "换成正式工装/西装", value: "formal_workwear" },
];

const STYLE_THEMES = [
    { label: "写实电影质感 (Cinematic Realistic)", value: "highly detailed, photorealistic, cinematic lighting, 8k resolution" },
    { label: "油画艺术风格 (Oil Painting)", value: "oil painting style, brush strokes, classical art" },
    { label: "柔和插画风 (Soft Illustration)", value: "digital illustration, soft lighting, warm colors" },
    { label: "复古胶片感 (Vintage Film)", value: "vintage film photography, grain, warm nostalgia" },
];

const App = () => {
    // Firebase State
    const [userId, setUserId] = useState(null);

    // Inputs
    const [biblicalFigureName, setBiblicalFigureName] = useState('');
    const [uploadedImageBase64, setUploadedImageBase64] = useState(null);
    const [uploadedImageMimeType, setUploadedImageMimeType] = useState(null);
    const [imagePreviewUrl, setImagePreviewUrl] = useState(null);
    const [selectedClothing, setSelectedClothing] = useState(CLOTHING_CHOICES[0].value);
    const [selectedStyle, setSelectedStyle] = useState(STYLE_THEMES[0].value);

    // Generation State
    const [generatedImageUrl, setGeneratedImageUrl] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [statusMessage, setStatusMessage] = useState("等待输入...");

    // 1. Firebase Auth (带容错处理)
    useEffect(() => {
        // 如果没有配置 Firebase，跳过初始化，仅作为演示使用
        if (!firebaseConfig.apiKey || firebaseConfig.apiKey === "YOUR_FIREBASE_API_KEY") {
            console.warn("Firebase 未配置，跳过身份验证。");
            setUserId("guest_user");
            return;
        }

        try {
            const app = initializeApp(firebaseConfig);
            const auth = getAuth(app);
            signInAnonymously(auth).catch(e => console.error("Auth Error:", e));
            
            const unsubscribe = onAuthStateChanged(auth, (user) => {
                if (user) setUserId(user.uid);
            });
            return () => unsubscribe();
        } catch (e) {
            console.error("Firebase Init Failed:", e);
        }
    }, []);

    // Helper: Retry Fetch
    const fetchWithRetry = async (url, options, maxRetries = 2) => {
        for (let i = 0; i < maxRetries; i++) {
            try {
                const response = await fetch(url, options);
                // 尝试解析错误信息
                if (!response.ok) {
                    const errData = await response.json().catch(() => ({}));
                    throw new Error(errData.error?.message || `HTTP Error ${response.status}`);
                }
                return await response.json();
            } catch (error) {
                if (i === maxRetries - 1) throw error;
                console.log(`Retrying... (${i + 1}/${maxRetries})`);
                await new Promise(resolve => setTimeout(resolve, 1500));
            }
        }
    };

    // Handle Image Upload
    const handleImageUpload = (event) => {
        const file = event.target.files[0];
        if (!file) return;
        
        if (file.size > 5 * 1024 * 1024) {
            setError("图片大小不能超过 5MB");
            return;
        }

        const reader = new FileReader();
        reader.onloadend = () => {
            const result = reader.result;
            // 提取 Base64 纯数据 (去掉 data:image/jpg;base64, 前缀)
            const [metadata, base64Data] = result.split(',');
            const mimeType = metadata.match(/:(.*?);/)[1];
            
            setUploadedImageBase64(base64Data);
            setUploadedImageMimeType(mimeType);
            setImagePreviewUrl(result);
            setError(null);
        };
        reader.readAsDataURL(file);
    };

    // 2. Direct Image Synthesis
    const handleDirectSynthesis = async () => {
        if (!uploadedImageBase64 || !biblicalFigureName) {
            setError("请确保上传了照片并输入了圣经人物名字。");
            return;
        }

        if (GOOGLE_API_KEY.includes("YOUR_")) {
            setError("请先在代码中配置有效的 Google API Key。");
            return;
        }

        setIsLoading(true);
        setError(null);
        setGeneratedImageUrl(null);
        setStatusMessage("正在连接 AI 模型...");

        try {
            // ⚠️ 关键点：模型名称
            // 如果 2.5-preview 不可用，您可以尝试换回 'gemini-1.5-pro' 进行测试（虽然它主要返回文本）
            // 或者如果您有 Imagen 3 的访问权限，API 路径可能不同
            const modelName = "gemini-1.5-flash"; // 暂时使用通用模型演示，如果您的账号有 2.5 权限请改回
            const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:generateContent?key=${GOOGLE_API_KEY}`;

            // 构建 Prompt
            const clothingPrompt = selectedClothing === 'keep_original' 
                ? "Keep the person's original clothing from the input image."
                : `Change the person's clothing to match the biblical era of ${biblicalFigureName}.`;

            const promptText = `
                [Direct Image Generation Request]
                Input Image: Provided.
                Task: Generate a high-quality image of the person in the input photo standing next to ${biblicalFigureName}.
                Style: ${selectedStyle}.
                Clothing: ${clothingPrompt}.
                Important: Maintain the user's face fidelity.
                NOTE: If you cannot generate an image directly, please describe the scene in extreme detail instead.
            `;

            const payload = {
                contents: [{
                    parts: [
                        { text: promptText },
                        {
                            inlineData: {
                                mimeType: uploadedImageMimeType,
                                data: uploadedImageBase64
                            }
                        }
                    ]
                }]
                // 注意：标准的 Gemini 1.5 API 默认返回文本。
                // 如果您有特定的 Image Output 权限，取消下面这行的注释
                // generationConfig: { responseModalities: ["IMAGE"] } 
            };

            setStatusMessage("AI 正在思考与绘制...");

            const result = await fetchWithRetry(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            console.log("AI Response:", result);

            // ⚠️ 处理响应：
            // 情况 A: 模型直接返回了图片 (少见，除非是特定 Imagen 端点)
            let outputBase64 = result.candidates?.[0]?.content?.parts?.find(p => p.inlineData)?.inlineData?.data;
            
            // 情况 B: 模型返回了文本描述 (Gemini 1.5 默认行为)
            // 在这种情况下，通常需要由前端再调用一次绘图 API (如 DALL-E 或 Imagen)
            // 这里为了演示，我们检查是否有图片，如果没有，显示错误提示
            if (!outputBase64) {
                const textOutput = result.candidates?.[0]?.content?.parts?.[0]?.text;
                if (textOutput) {
                    throw new Error("模型仅返回了文本描述，未返回图像。请确认您的 API Key 是否有 Imagen 权限或使用 Python 后端转发。描述：" + textOutput.slice(0, 50) + "...");
                }
                throw new Error("模型未返回任何数据。");
            }

            setGeneratedImageUrl(`data:image/jpeg;base64,${outputBase64}`);
            setStatusMessage("合成成功！");

        } catch (e) {
            console.error("Synthesis error:", e);
            setError(`生成失败: ${e.message}`);
            setStatusMessage("发生错误");
        } finally {
            setIsLoading(false);
        }
    };

    // Download Handler
    const handleDownload = () => {
        if (generatedImageUrl) {
            const link = document.createElement('a');
            link.href = generatedImageUrl;
            link.download = `HolyCoop_${biblicalFigureName}_${Date.now()}.jpg`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    };

    // UI Components
    const Section = ({ title, children }) => (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-5 mb-4">
            <h2 className="text-lg font-bold text-gray-800 mb-4 border-l-4 border-indigo-500 pl-3">{title}</h2>
            {children}
        </div>
    );

    return (
        <div className="min-h-screen bg-gray-50 font-sans pb-10">
            {/* Header */}
            <header className="bg-gradient-to-r from-indigo-600 to-purple-700 text-white p-6 rounded-b-3xl shadow-lg mb-6">
                <div className="max-w-xl mx-auto text-center">
                    <h1 className="text-3xl font-extrabold tracking-tight">✝️ 圣经合影合成器</h1>
                    <p className="text-indigo-100 mt-2 text-sm opacity-90">AI 驱动 • 跨越时空的相遇</p>
                </div>
            </header>

            <main className="max-w-xl mx-auto px-4">

                {/* 1. Upload Section */}
                <Section title="1. 上传您的照片">
                    <div className="flex flex-col items-center justify-center w-full">
                        <label className="flex flex-col items-center justify-center w-full h-48 border-2 border-indigo-300 border-dashed rounded-xl cursor-pointer bg-indigo-50 hover:bg-indigo-100 transition relative overflow-hidden group">
                            {imagePreviewUrl ? (
                                <>
                                    <img src={imagePreviewUrl} alt="Preview" className="h-full w-full object-contain rounded-xl z-10 relative" />
                                    <div className="absolute inset-0 bg-black bg-opacity-40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition z-20">
                                        <p className="text-white font-bold">点击更换</p>
                                    </div>
                                </>
                            ) : (
                                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                                    <span className="text-4xl mb-2">📸</span>
                                    <p className="mb-2 text-sm text-gray-500"><span className="font-semibold">点击上传</span> 自拍/半身照</p>
                                </div>
                            )}
                            <input type="file" className="hidden" accept="image/*" onChange={handleImageUpload} />
                        </label>
                    </div>
                </Section>

                {/* 2. Figure Input */}
                <Section title="2. 输入圣经人物">
                    <div className="mb-2">
                        <label className="block text-sm font-medium text-gray-700 mb-1">您想与哪位人物合影？</label>
                        <input 
                            type="text" 
                            className="w-full p-4 border border-gray-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition shadow-sm text-lg"
                            placeholder="例如：大卫、参孙、彼得、路得..."
                            value={biblicalFigureName}
                            onChange={(e) => setBiblicalFigureName(e.target.value)}
                        />
                    </div>
                </Section>

                {/* 3. Settings */}
                <Section title="3. 风格与服饰设置">
                    <div className="grid gap-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">您的服装选择</label>
                            <select 
                                className="w-full p-3 bg-white border border-gray-200 rounded-lg text-gray-700 focus:ring-indigo-500 focus:border-indigo-500"
                                value={selectedClothing}
                                onChange={(e) => setSelectedClothing(e.target.value)}
                            >
                                {CLOTHING_CHOICES.map(c => <option key={c.value} value={c.value}>{c.label}</option>)}
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">画面艺术风格</label>
                            <select 
                                className="w-full p-3 bg-white border border-gray-200 rounded-lg text-gray-700 focus:ring-indigo-500 focus:border-indigo-500"
                                value={selectedStyle}
                                onChange={(e) => setSelectedStyle(e.target.value)}
                            >
                                {STYLE_THEMES.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
                            </select>
                        </div>
                    </div>
                </Section>

                {/* 4. Action */}
                <button
                    onClick={handleDirectSynthesis}
                    disabled={isLoading || !uploadedImageBase64 || !biblicalFigureName}
                    className={`w-full py-4 text-lg font-bold rounded-xl shadow-lg transform transition active:scale-95 flex items-center justify-center ${
                        isLoading || ! uploadedImageBase64 || !biblicalFigureName
                        ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                        : 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white hover:from-indigo-700 hover:to-purple-700'
                    }`}
                >
                    {isLoading ? (
                        <>
                            <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                            {statusMessage}
                        </>
                    ) : (
                        "✨ 开始合成合照"
                    )}
                </button>

                {/* 5. Result */}
                {error && (
                    <div className="mt-4 p-4 bg-red-50 text-red-600 rounded-xl border border-red-200 text-sm animate-fade-in">
                        ❌ {error}
                    </div>
                )}

                {generatedImageUrl && (
                    <div className="mt-6 bg-white p-2 rounded-2xl shadow-xl border-4 border-indigo-100 animate-fade-in-up">
                        <img src={generatedImageUrl} alt="Generated Result" className="w-full rounded-xl" />
                        <div className="p-4">
                            <h3 className="text-center font-bold text-gray-800 mb-3">✅ 合照已生成！</h3>
                            <button 
                                onClick={handleDownload}
                                className="w-full py-3 bg-green-500 text-white font-bold rounded-lg shadow-md hover:bg-green-600 transition flex items-center justify-center gap-2"
                            >
                                下载图片
                            </button>
                        </div>
                    </div>
                )}
            </main>

            <footer className="mt-10 text-center text-gray-400 text-xs pb-4">
                 ID: {userId ? userId.slice(0, 8) + '...' : 'Guest'} • Powered by Google AI
            </footer>
        </div>
    );
};

export default App;
