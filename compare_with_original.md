# 未來如何比對與更新？

因為你同時保留了原作者的連結 (origin) 和你自己的連結 (my-repo)，未來的操作會非常方便：

## 抓取原作者的最新代碼：
```bash
git fetch origin
```

## 比對你的進度 vs 原作者的進度：
```bash
git diff pitt..origin/main
```

## 如果想要把原作者的更新拉進你的分支：
```bash
# 確保你目前在 pitt 分支
git merge origin/main  # 或是使用 git rebase origin/main
```
