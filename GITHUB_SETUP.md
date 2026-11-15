# Upload the project to GitHub

After downloading the UCI dataset, running the training script, and checking that the Streamlit app works:

1. Create a new **public or private** repository on your GitHub account. Do not initialize it with an extra README if you plan to push this folder as-is.
2. Open a terminal in `Customer_Segmentation_Project/` and run:

```bash
git init
git add .
git status
git commit -m "Complete Customer Segmentation ML project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

3. Replace the remote URL with your own repository URL.
4. Copy the repository link into your college submission form/report if required.

The UCI workbook is about 22.6 MB, below GitHub's usual 100 MB per-file limit. Include it in the repository if your college expects the dataset folder to contain the actual dataset; otherwise keep the downloader and dataset-source instructions so the project can reproduce the data locally.
