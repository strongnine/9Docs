using Documenter

makedocs(
    sitename = "9Docs", 
    pages = [
        "Home" => "index.md",
        "人工智能" => Any[
            "特征工程" => "AI/FE.md",
            "机器学习" => "AI/ML.md",
            "神经网络" => "AI/NN.md",
            "卷积神经网络" => "AI/CNN.md",
            "循环神经网络" => "AI/RNN.md",
            "图神经网络" => "AI/GNN.md",
            "生成对抗网络" => "AI/GAN.md",
            "计算机视觉" => "AI/CV.md",
            "自然语言处理" => "AI/NLP.md",
        ],
        "编程语言" => Any[
            "Python" => "lang/Python.md",
            "C++" => "lang/Cpp.md",
            "Julia" => "lang/Julia.md",
        ],
        "Git" => "git.md",
        "Docker" => "docker.md",
        "Linux" => "Linux.md",
        "Vim" => "vim.md",
    ],
)

deploydocs(
    repo = "github.com/strongnine/9Docs.git",
    target = "build",
    devbranch = "main",
)
