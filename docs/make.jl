using Documenter

makedocs(
    sitename = "9Docs", 
    pages = [
        "Home" => "index.md",
        # "机器视觉" => Any[
        #     "对比度增强" => "CV/contrast.md",
        #     "暗通道先验去雾" => "CV/dark_channel_prior.md",
        # ],
        "深度学习" => Any[
        #     "特征工程" => "AI/FE.md",
        #     "机器学习" => "AI/ML.md",
        #     "神经网络" => "AI/NN.md",
            "卷积神经网络" => "DL/CNN.md",
            "立体匹配" => "DL/stereo.md",
        #     "循环神经网络" => "AI/RNN.md",
        #     "图神经网络" => "AI/GNN.md",
        #     "生成对抗网络" => "AI/GAN.md",
        #     "计算机视觉" => "AI/CV.md",
        #     "自然语言处理" => "AI/NLP.md",
        #     "Transformer" => "AI/Transformer.md",
        ],
        # "其他" => Any[
        #     "推荐书籍大全" => "books.md",
        # ]
    ],
)


deploydocs(
    repo = "github.com/strongnine/9Docs.git",
    target = "build",
    devbranch = "main",
)
