using Documenter

makedocs(
    sitename = "9Docs", 
    pages = [
        "Home" => "index.md",
        "Git 学习笔记" => "git/git_notebook.md",
        "LeetCode 刷题" => "leetcode/leetcode.md",
    ],
)

deploydocs(
    repo = "github.com/strongnine/9Docs.git",
    target = "build",
    devbranch = "main",
)
