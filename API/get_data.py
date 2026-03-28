import requests
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
REPO = "microsoft/vscode" 
AUTH_TOKEN =os.getenv("API_KEY") 
headers = {"Authorization": f"token {AUTH_TOKEN}"}

def get_issues(repo, num_pages=5):
    issues = []
    for page in range(1, num_pages + 1):
        url = f"https://api.github.com/repos/{repo}/issues?state=closed&per_page=100&page={page}"
        response = requests.get(url, headers=headers,timeout=30)
        if response.status_code == 200:
            data = response.json()
            for item in data:
                if "pull_request" not in item:
                    labels = [l['name'] for l in item['labels']]
                    if labels:
                        issues.append({
                            "title": item["title"],
                            "body": item["body"],
                            "label": labels[0]
                        })
        print(f"Finished page {page}")
    return pd.DataFrame(issues)

df = get_issues(REPO, num_pages=100)
df.to_csv(r"Data\Scraped_data\github_issues.csv", index=False)