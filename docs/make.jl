using Documenter

makedocs(
    sitename = "9Docs", 
    pages = [
        "Home" => "index.md",
        "特征工程" => "AI/FE.md",
        "机器学习" => "AI/ML.md",
        "神经网络" => "AI/NN.md",
        "卷积神经网络" => "AI/CNN.md",
        "循环神经网络" => "AI/RNN.md",
        "图神经网络" => "AI/GNN.md",
#        "自然语言处理" => "AI/NLP.md",
        "Git 学习笔记" => "git/git_notebook.md",
    ],
)

deploydocs(
    repo = "github.com/strongnine/9Docs.git",
    target = "build",
    devbranch = "main",
)
