# HOW TO USE GIT #
****
## init ##
创建新的文件夹 输入命令

    git init 
    git config --global user.name 'gitname'
    git config --global user.email 'git_email_address'

分别为初始化本地仓库  设置用户名和邮箱

## 工作流 ##

本地仓库由三个部分组成 第一个是工作目录  第二个是暂存区（Index）

第三个是本地仓库（HEAD) 在工作目录完成文件的修改后提交到暂存区，
将暂存区的文件提交到HEAD后，HEAD能提交到远端仓库。
    
    git add filename # 此命令将工作目录文件提交到暂存区
    git commit -m 'message'  #此命令将暂存区文件提交到本地仓库并添加说明

## branch ##
    git checkout -b feature_x # 创建一个叫feature_x的分支并切换过去
    git checkout master # 切回主分支
    git branch -d feature_x # 删除分支
    git push origin <branch> # 将此分支推送到远端仓库 不推送此分支不为人所见

## 更新与合并 ##
    git pull # 将本地仓库更新至最新
    git merge <branch> # 将其他分支合并到此分支

    git diff <sourse_branch> <target_branch> # 比较目标分支和本分支的差别
## 将本地仓库推送到远程仓库 ##

    git push origin master

    git clone <gitaddress> # 将远程仓库克隆到本地

## 替换本地改动 ##


    git checkout -- <filename>
操作失误时使用此命令可将本地仓库中此文件的最新版本替换工作目录中的
此文件

## 后续有用到的其它功能再添加 ##



### [借鉴于521xueweihan/git-tips 非常感谢](https://github.com/zhuangzhuangsun/git-tips) ###
