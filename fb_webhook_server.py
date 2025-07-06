from flask import Flask, request, jsonify
import hashlib
import hmac
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

VERIFY_TOKEN = os.environ.get('VERIFY_TOKEN')
PAGE_ACCESS_TOKEN = os.environ.get('PAGE_ACCESS_TOKEN')
APP_SECRET = os.environ.get('APP_SECRET')

# Debug: Print environment variables (remove in production)
print(f"VERIFY_TOKEN: {'SET' if VERIFY_TOKEN else 'NOT SET'}")
print(f"PAGE_ACCESS_TOKEN: {'SET' if PAGE_ACCESS_TOKEN else 'NOT SET'}")
print(f"APP_SECRET: {'SET' if APP_SECRET else 'NOT SET'}")

@app.route('/webhook', methods=['GET'])
def verify_webhook():
    """
    Webhook verification endpoint
    Facebook will call this to verify your webhook
    """
    print("=== WEBHOOK VERIFICATION REQUEST ===")
    
    # Parse params from the webhook verification request
    mode = request.args.get('hub.mode')
    token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')
    
    print(f"Mode: {mode}")
    print(f"Token: {token}")
    print(f"Challenge: {challenge}")
    print(f"Expected token: {VERIFY_TOKEN}")
    
    # Check if a token and mode were sent
    if mode and token:
        # Check the mode and token sent are correct
        if mode == 'subscribe' and token == VERIFY_TOKEN:
            # Respond with 200 OK and challenge token from the request
            print('WEBHOOK_VERIFIED')
            return challenge
        else:
            # Respond with '403 Forbidden' if verify tokens do not match
            print('VERIFICATION FAILED: Token mismatch')
            return 'Verification token mismatch', 403
    
    print('VERIFICATION FAILED: Missing parameters')
    return 'Missing parameters', 400

@app.route('/webhook', methods=['POST'])
def handle_message():
    """
    Handle incoming messages from Facebook Messenger
    """
    print("=== INCOMING WEBHOOK POST ===")
    print(f"Headers: {dict(request.headers)}")
    
    # Verify the request signature
    signature = request.headers.get('X-Hub-Signature-256')
    print(f"Signature: {signature}")
    
    if not verify_signature(request.data, signature):
        print("SIGNATURE VERIFICATION FAILED")
        return 'Invalid signature', 403
    
    print("Signature verified successfully")
    
    # Parse the request body
    try:
        data = request.get_json()
        print(f"Received data: {json.dumps(data, indent=2)}")
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return 'Invalid JSON', 400
    
    # Make sure this is a page subscription
    if data.get('object') == 'page':
        print(f"Processing {len(data.get('entry', []))} entries")
        
        # Iterate over each entry (there may be multiple if batched)
        for entry in data.get('entry', []):
            print(f"Processing entry: {entry}")
            
            # Get the webhook event
            webhook_event = entry.get('messaging', [])
            print(f"Found {len(webhook_event)} messaging events")
            
            # Iterate over each messaging event
            for event in webhook_event:
                print(f"Processing event: {event}")
                sender_id = event['sender']['id']
                
                # Check if the event contains a message
                if 'message' in event:
                    print(f"Handling message event from {sender_id}")
                    handle_message_event(sender_id, event['message'])
                
                # Check if the event contains a postback
                elif 'postback' in event:
                    print(f"Handling postback event from {sender_id}")
                    handle_postback_event(sender_id, event['postback'])
                
                else:
                    print(f"Unknown event type: {list(event.keys())}")
    else:
        print(f"Not a page subscription: {data.get('object')}")
    
    return 'OK', 200

def verify_signature(payload, signature):
    """
    Verify that the request came from Facebook
    """
    if not signature:
        print("No signature provided")
        return False
    
    if not APP_SECRET:
        print("APP_SECRET not configured")
        return False
    
    # Remove 'sha256=' prefix
    signature = signature.replace('sha256=', '')
    
    # Create hash using app secret
    expected_signature = hmac.new(
        APP_SECRET.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    print(f"Expected signature: {expected_signature}")
    print(f"Received signature: {signature}")
    
    is_valid = hmac.compare_digest(signature, expected_signature)
    print(f"Signature valid: {is_valid}")
    
    return is_valid

def handle_message_event(sender_id, message):
    """
    Handle incoming message events
    """
    print(f"=== HANDLING MESSAGE FROM {sender_id} ===")
    print(f"Message: {json.dumps(message, indent=2)}")
    
    # Get message text
    message_text = message.get('text', '')
    print(f"Message text: '{message_text}'")
    
    # Simple echo bot - you can customize this logic
    if message_text:
        response_text = f"You said: {message_text}"
        print(f"Sending response: {response_text}")
        send_message(sender_id, response_text)
    
    # Handle attachments
    if 'attachments' in message:
        print("Message contains attachments")
        send_message(sender_id, "Thanks for the attachment!")

def handle_postback_event(sender_id, postback):
    """
    Handle postback events (button clicks, etc.)
    """
    payload = postback.get('payload')
    print(f"=== HANDLING POSTBACK FROM {sender_id} ===")
    print(f"Postback payload: {payload}")
    
    # Handle different postback payloads
    if payload == 'GET_STARTED':
        send_message(sender_id, "Hello! Welcome to my bot!")
    else:
        send_message(sender_id, f"You clicked: {payload}")

def send_message(recipient_id, message_text):
    """
    Send a text message to a recipient
    """
    if not PAGE_ACCESS_TOKEN:
        print("ERROR: PAGE_ACCESS_TOKEN not configured")
        return None
    
    url = f"https://graph.facebook.com/v18.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
    
    data = {
        'recipient': {'id': recipient_id},
        'message': {'text': message_text}
    }
    
    print(f"Sending message to {recipient_id}: {message_text}")
    print(f"API URL: {url}")
    print(f"Payload: {json.dumps(data, indent=2)}")
    
    try:
        response = requests.post(url, json=data)
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")
        
        if response.status_code == 200:
            print(f"Message sent successfully to {recipient_id}")
        else:
            print(f"Failed to send message: {response.text}")
    
    except Exception as e:
        print(f"Error sending message: {e}")
        response = None
    
    return response

def send_quick_replies(recipient_id, text, quick_replies):
    """
    Send a message with quick reply buttons
    """
    url = f"https://graph.facebook.com/v18.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
    
    data = {
        'recipient': {'id': recipient_id},
        'message': {
            'text': text,
            'quick_replies': quick_replies
        }
    }
    
    response = requests.post(url, json=data)
    return response

def send_button_template(recipient_id, text, buttons):
    """
    Send a message with button template
    """
    url = f"https://graph.facebook.com/v18.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
    
    data = {
        'recipient': {'id': recipient_id},
        'message': {
            'attachment': {
                'type': 'template',
                'payload': {
                    'template_type': 'button',
                    'text': text,
                    'buttons': buttons
                }
            }
        }
    }
    
    response = requests.post(url, json=data)
    return response

# Add a simple health check endpoint
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'ok',
        'env_vars': {
            'VERIFY_TOKEN': 'SET' if VERIFY_TOKEN else 'NOT SET',
            'PAGE_ACCESS_TOKEN': 'SET' if PAGE_ACCESS_TOKEN else 'NOT SET',
            'APP_SECRET': 'SET' if APP_SECRET else 'NOT SET'
        }
    })

if __name__ == '__main__':
    print("Starting Facebook Webhook Server...")
    print("Available endpoints:")
    print("  GET  /webhook - Webhook verification")
    print("  POST /webhook - Message handling")
    print("  GET  /health  - Health check")
    
    # For development only - use a proper WSGI server for production
    app.run(debug=True, port=5000, host='0.0.0.0')