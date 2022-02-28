using Documenter

makedocs(
    sitename = "9Docs", 
    pages = [
        "Home" => "index.md",
        "数据结构与算法" => "algorithm/algo.md",
        "机器学习" => Any[
            "图神经网络" => "machine_learning/GNN.md",
        ],
        "产品思维" => "product/product_manager.md",
        "个人推荐" => Any[
            "书籍推荐" => "library/book.md",
            "其他推荐" => "library/recommend.md"
        ],
        "Git 学习笔记" => "git/git_notebook.md", 
    ],
)

deploydocs(
    repo = "github.com/strongnine/9Docs.git",
    target = "build",
    devbranch = "main",
)
