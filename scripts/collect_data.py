"""
GitHub Bug Report Collector for Game Repositories
Scrapes bug reports from major open-source game projects
"""

from github import Github
import os
import json
import time
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd

# Load environment variables
load_dotenv()

# Game repositories to scrape
GAME_REPOS = [
    "godotengine/godot",           # Game engine with 70k+ stars
    "bevyengine/bevy",              # Rust game engine
    "minetest/minetest",            # Voxel game engine
    "OpenRCT2/OpenRCT2",           # RollerCoaster Tycoon 2
]

# Bug-related labels to search for
BUG_LABELS = ["bug", "defect", "crash", "issue", "error"]

def collect_bug_reports(repo_name, max_issues=500, token=None):
    """
    Collect bug reports from a GitHub repository
    
    Args:
        repo_name: Repository in format "owner/repo"
        max_issues: Maximum number of issues to collect
        token: GitHub personal access token
    
    Returns:
        List of bug report dictionaries
    """
    print(f"\n📦 Collecting from {repo_name}...")
    
    # Initialize GitHub API
    if token:
        g = Github(token)
    else:
        g = Github()  # Unauthenticated (limited to 60 requests/hour)
        print("⚠️  No GitHub token found. Rate limited to 60 requests/hour.")
        print("   Set GITHUB_TOKEN in .env for higher limits (5000/hour)")
    
    try:
        repo = g.get_repo(repo_name)
        bugs = []
        
        # Search for issues with bug-related labels
        issues = repo.get_issues(state='all', labels=[])
        
        count = 0
        for issue in tqdm(issues, desc=f"Fetching issues", total=max_issues):
            if count >= max_issues:
                break
            
            # Filter for bug-related issues
            issue_labels = [label.name.lower() for label in issue.labels]
            is_bug = any(bug_label in ' '.join(issue_labels) for bug_label in BUG_LABELS)
            
            # Also check title and body for bug keywords
            title_and_body = f"{issue.title} {issue.body or ''}".lower()
            if not is_bug:
                is_bug = any(keyword in title_and_body for keyword in ['crash', 'bug', 'error', 'broken'])
            
            if is_bug and issue.body and len(issue.body) > 50:
                bug_data = {
                    'repository': repo_name,
                    'issue_number': issue.number,
                    'title': issue.title,
                    'body': issue.body[:2000],  # Limit body length
                    'state': issue.state,
                    'labels': [label.name for label in issue.labels],
                    'created_at': issue.created_at.isoformat(),
                    'comments_count': issue.comments,
                    'is_pull_request': issue.pull_request is not None,
                }
                bugs.append(bug_data)
                count += 1
        
        print(f"✅ Collected {len(bugs)} bug reports from {repo_name}")
        return bugs
    
    except Exception as e:
        print(f"❌ Error collecting from {repo_name}: {e}")
        return []

def main():
    print("="*60)
    print("🎮 Game Bug Report Collector")
    print("="*60)
    
    # Get GitHub token
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        print("\n⚠️  Warning: No GITHUB_TOKEN found in .env")
        print("   You'll be limited to 60 API requests per hour")
        print("   Get a token at: https://github.com/settings/tokens")
        response = input("\n   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Please set GITHUB_TOKEN in .env file")
            return
    
    # Create data directory
    os.makedirs('data/raw', exist_ok=True)
    
    # Collect from all repos
    all_bugs = []
    for repo_name in GAME_REPOS:
        bugs = collect_bug_reports(repo_name, max_issues=500, token=github_token)
        all_bugs.extend(bugs)
        time.sleep(2)  # Be nice to GitHub API
    
    print(f"\n{'='*60}")
    print(f"📊 Total bugs collected: {len(all_bugs)}")
    print(f"{'='*60}\n")
    
    # Save to files
    if all_bugs:
        # Save as JSON
        json_path = 'data/raw/game_bug_reports.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_bugs, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved to {json_path}")
        
        # Save as CSV for easy viewing
        df = pd.DataFrame(all_bugs)
        csv_path = 'data/raw/game_bug_reports.csv'
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved to {csv_path}")
        
        # Print statistics
        print(f"\n📈 Dataset Statistics:")
        print(f"   Total issues: {len(all_bugs)}")
        print(f"   Repositories: {df['repository'].nunique()}")
        print(f"   Open issues: {sum(1 for b in all_bugs if b['state'] == 'open')}")
        print(f"   Closed issues: {sum(1 for b in all_bugs if b['state'] == 'closed')}")
        print(f"   Issues per repo:")
        for repo in GAME_REPOS:
            count = sum(1 for b in all_bugs if b['repository'] == repo)
            print(f"      {repo}: {count}")
    else:
        print("❌ No bugs collected. Check your internet connection and GitHub token.")

if __name__ == "__main__":
    main()
