using Documenter

makedocs(
    sitename = "9Docs", 
    pages = [
        "Home" => "index.md",
        "机器学习" => Any[
            "图神经网络" => "machine_learning/GNN.md",
        ],
        "Git 学习笔记" => "git/git_notebook.md", 
        "LeetCode 刷题" => Any[
            "数据结构" => "leetcode/data_structure.md",
            "算法总结" => "leetcode/algorithm.md",
            "刷题记录" => "leetcode/leetcoding.md",
        ],
        "产品" => "product/product_manager.md",
    ],
)

deploydocs(
    repo = "github.com/strongnine/9Docs.git",
    target = "build",
    devbranch = "main",
)
