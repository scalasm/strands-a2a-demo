# Contributing to Strands A2A Demo

Thank you for your interest in contributing to the Strands A2A Demo project! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- [UV package manager](https://docs.astral.sh/uv/getting-started/installation/)
- Git

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/strands-a2a-demo.git
cd strands-a2a-demo

# Install dependencies
uv sync

# Set up pre-commit hooks (optional but recommended)
uv run pre-commit install
```

## 🛠️ Development Workflow

### 1. Fork and Branch
```bash
# Fork the repository on GitHub, then:
git checkout -b feature/your-feature-name
```

### 2. Make Changes
- Write clear, concise commit messages
- Follow Python PEP 8 style guidelines
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes
```bash
# Run the test suite
uv run a2a-client test

# Start agents and test manually
uv run agents start
uv run a2a-client web  # Test web interface
uv run a2a-client interactive  # Test CLI
```

### 4. Submit Pull Request
- Push your changes to your fork
- Create a pull request with a clear description
- Reference any related issues

## 📋 Contribution Types

### Bug Reports
- Use the GitHub issue template
- Include steps to reproduce
- Provide system information (OS, Python version, etc.)
- Include relevant log files from `./logs/`

### Feature Requests
- Describe the use case and benefit
- Consider backward compatibility
- Discuss implementation approach

### Code Contributions
- **New Agents**: Follow the configuration patterns in `config/config.toml`
- **Client Modes**: Extend `a2a_client.py` following existing patterns
- **Tools Integration**: Add to `strands_tools_mapping.py`
- **Documentation**: Update README.md and relevant sections

## 🏗️ Architecture Guidelines

### Adding New Agents
1. Define in `config/config.toml`:
   ```toml
   [configurable_agent.newagent]
   description = "Description of new agent"
   port = 8XXX
   system_prompt = "Agent behavior definition"
   strands_tools = ["tool1", "tool2"]
   ```

2. Test the agent:
   ```bash
   uv run configurable-agent newagent
   uv run a2a-client test --agent newagent
   ```

### Adding New Client Modes
1. Create interface module (e.g., `new_interface.py`)
2. Add mode to `a2a_client.py` main function
3. Update README.md with usage instructions
4. Add tests to `test_interface.py`

### Code Style
- Follow PEP 8
- Use type hints where possible
- Write docstrings for public functions
- Keep functions focused and small
- Use descriptive variable names

## 🧪 Testing

### Running Tests
```bash
# All tests
uv run a2a-client test

# Specific agent
uv run a2a-client test --agent calculator

# Manual testing
uv run agents start
# Test each interface mode
```

### Adding Tests
- Add test cases to `test_interface.py`
- Test both success and failure scenarios
- Include edge cases
- Test with different agent configurations

## 📚 Documentation

### Required Documentation Updates
- Update README.md for new features
- Add code comments for complex logic
- Update API reference for new functions
- Include examples in docstrings

### Documentation Style
- Use clear, concise language
- Include code examples
- Add diagrams for complex flows
- Keep tutorials beginner-friendly

## 🐛 Debugging

### Useful Commands
```bash
# Enable debug logging
export STRANDS_LOG_LEVEL=DEBUG

# View logs
tail -f ./logs/*.log

# Check agent status
uv run agents status

# Test connectivity
curl http://localhost:8001/.well-known/agent.json
```

## 🔒 Security

### Reporting Security Issues
- **DO NOT** create public issues for security vulnerabilities
- Email security issues to: [security-email]
- Include detailed reproduction steps
- Allow time for fix before public disclosure

### Security Guidelines
- Never commit credentials or API keys
- Use environment variables for sensitive configuration
- Validate all user inputs
- Follow secure coding practices

## 📝 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## 🤝 Code of Conduct

### Our Standards
- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers learn and contribute
- Maintain professional communication

### Enforcement
- Report issues to project maintainers
- Violations may result in temporary or permanent ban
- See full Code of Conduct (link when available)

## 💬 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and community chat
- **Documentation**: Start with README.md
- **Examples**: Check the tutorials in README.md

## 🙏 Recognition

Contributors are recognized in:
- GitHub contributors list
- Release notes for significant contributions
- Special mentions for exceptional help

Thank you for contributing to the Strands A2A Demo project! 