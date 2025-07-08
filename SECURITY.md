# Security Policy

## 🔒 Supported Versions

We currently support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## 🛡️ Security Considerations

### Credential Management
- **Never commit AWS credentials** to the repository
- Use environment variables or AWS credential files
- Rotate credentials regularly
- Use IAM roles when possible (e.g., on EC2 instances)

### Network Security
- Agents run on localhost by default (ports 8000-8099)
- Web interfaces bind to `0.0.0.0` but should be firewalled in production
- Voice modes require internet access for Amazon Nova services

### Input Validation
- All user inputs are processed by LLM agents
- File operations are restricted to configured directories
- Web content fetching is limited by MCP server capabilities

## 🚨 Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

### How to Report
1. **Email**: Send details to [your-security-email] (replace with actual email)
2. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Any suggested fixes

### What to Expect
- **Initial Response**: Within 48 hours
- **Status Updates**: Every 72 hours during investigation
- **Resolution Timeline**: Security issues will be prioritized based on severity

### Responsible Disclosure
- Allow reasonable time for fix development and testing
- Coordinate public disclosure timing
- Credit will be given for responsible reporting

## 🔐 Security Best Practices

### For Users
```bash
# Use environment variables for AWS credentials
export AWS_ACCESS_KEY_ID=your_access_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_access_key

# Check file permissions
chmod 600 ~/.aws/credentials

# Run with minimal required permissions
# Avoid running as root/administrator
```

### For Developers
- Validate all configuration file inputs
- Sanitize file paths and names
- Use secure random generation for secrets
- Implement proper error handling to avoid information leakage
- Keep dependencies updated

### For Production Deployments
- Use proper firewall rules
- Implement authentication for web interfaces
- Use HTTPS for all web traffic
- Monitor logs for suspicious activity
- Regular security assessments

## 🛠️ Security Tools

### Static Analysis
```bash
# Install security scanning tools
uv add --dev bandit safety

# Run security checks
uv run bandit -r .
uv run safety check
```

### Dependency Monitoring
- UV automatically tracks dependency versions in `uv.lock`
- Regular updates help maintain security
- Monitor for security advisories

## 📋 Security Checklist

### Before Production
- [ ] Remove all debug logging
- [ ] Implement proper authentication
- [ ] Configure HTTPS/TLS
- [ ] Set up monitoring and alerting
- [ ] Review firewall rules
- [ ] Test backup and recovery procedures
- [ ] Document security architecture

### Regular Maintenance
- [ ] Update dependencies monthly
- [ ] Review access logs weekly
- [ ] Rotate credentials quarterly
- [ ] Conduct security reviews annually

## 📚 Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Guidelines](https://python.org/dev/security/)
- [AWS Security Best Practices](https://aws.amazon.com/security/security-resources/)
- [A2A Security Considerations](https://google.github.io/A2A/security/)

## 🏷️ Security Labels

We use the following GitHub labels for security-related issues:
- `security` - General security improvements
- `vulnerability` - Confirmed security vulnerabilities
- `security-enhancement` - Proactive security improvements

Thank you for helping keep the Strands A2A Demo project secure! 