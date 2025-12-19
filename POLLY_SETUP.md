# AWS Polly TTS Setup Guide

## Quick Start

### 1. Install Dependencies
```bash
python3 -m pip install boto3>=1.34.0
```

### 2. Configure Environment Variables

Edit your `.env` file and add:

```bash
# TTS Provider - set to "polly" to use AWS Polly
TTS_PROVIDER=polly

# AWS Credentials (REQUIRED for Polly)
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1

# AWS Polly Voice Settings (Optional)
POLLY_VOICE_ID=Salli
POLLY_OUTPUT_FORMAT=mp3
```

### 3. Test the Integration

```bash
python3 test_polly.py
```

This will:
- Test AWS Polly API connection
- Test the AWSPollyTTSHandler class
- Generate sample audio files for verification

### 4. Run Your Server

```bash
python3 run_socket_server.py
```

The server will automatically use AWS Polly for TTS based on your `TTS_PROVIDER` setting.

## Available Voices

AWS Polly supports many voices. Popular options:

- **Salli** (Female, US English) - Default
- **Joanna** (Female, US English)
- **Matthew** (Male, US English)
- **Joey** (Male, US English)
- **Kendra** (Female, US English)
- **Kimberly** (Female, US English)

[Full voice list](https://docs.aws.amazon.com/polly/latest/dg/voicelist.html)

## Switching Between TTS Providers

To switch back to the custom TTS endpoint:

```bash
TTS_PROVIDER=custom
```

To use AWS Polly:

```bash
TTS_PROVIDER=polly
```

## Security Notes

⚠️ **IMPORTANT**: 
- Never commit AWS credentials to version control
- Use environment variables or AWS IAM roles
- Rotate credentials regularly
- Use least-privilege IAM policies

## Troubleshooting

### Error: "AWS credentials are required"
- Ensure `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are set in `.env`
- Check that the `.env` file is in the project root

### Error: "Text is too long"
- AWS Polly has a 3000 character limit per request
- Consider breaking long texts into smaller chunks

### Error: "Service failure"
- Check your AWS account status
- Verify your credentials are valid
- Check AWS service health status
