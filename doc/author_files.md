# Author Feature Files
- https://drive.google.com/drive/folders/1CLJ2a3H_oTX5b_4GLurVYoCpZUXQVpFr
- Command:
```bash
scp -P 9413 \
  "/home/ting/Downloads/CIFAR10-20220914T064003Z-001.zip" \
  "tingweijing@ssh.jiuntian.com:/data/weijing/author-DI/_author_files/CIFAR10-20220914T064003Z-001.zip"

unzip CIFAR10-20220914T064003Z-001.zip -d ./

zip -r reproduce_files.zip "./_reproduce_files_01"
zip -r reproduce_models.zip "./reproduce_models"

scp -P 9413 \
  "tingweijing@ssh.jiuntian.com:/data/weijing/author-DI/reproduce_files.zip" \
  "/home/ting/Downloads/reproduce_files.zip"
scp -P 9413 \
  "tingweijing@ssh.jiuntian.com:/data/weijing/author-DI/reproduce_models.zip" \
  "/home/ting/Downloads/reproduce_models.zip"
```