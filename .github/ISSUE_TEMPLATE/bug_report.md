---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

## 🐛 Bug Description
A clear and concise description of what the bug is.

## 🔄 To Reproduce
Steps to reproduce the behavior:
1. Start agents with '...'
2. Run client with '...'
3. See error

## ✅ Expected Behavior
A clear and concise description of what you expected to happen.

## 📱 Environment
- OS: [e.g. macOS 14.0, Ubuntu 22.04]
- Python version: [e.g. 3.12.1]
- UV version: [e.g. 0.5.0]
- Project version: [e.g. 0.1.0]

## 📋 Agent Configuration
```toml
# Paste relevant sections from config/config.toml
```

## 📄 Command Output
```bash
# Paste the full command and output
uv run agents status
# output here...
```

## 📝 Log Files
```
# Paste relevant log entries from ./logs/
```

## 📷 Screenshots
If applicable, add screenshots to help explain your problem.

## 🔧 Additional Context
Add any other context about the problem here.

## 🔍 Attempted Solutions
- [ ] Restarted agents with `uv run agents restart`
- [ ] Checked port availability with `lsof -i :8001`
- [ ] Verified AWS credentials (for voice modes)
- [ ] Checked logs for errors
- [ ] Tried with different agent configurations 