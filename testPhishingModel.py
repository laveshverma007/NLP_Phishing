import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(model_path='./phishing_model'):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer

def predict_email(email_text, model, tokenizer, max_length=128):
    # Prepare device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Tokenize input
    inputs = tokenizer(
        email_text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(predictions, dim=1).item()
        confidence = predictions[0][predicted_class].item()

    return {
        'is_phishing': bool(predicted_class),
        'confidence': confidence,
        'prediction': 'Phishing Email' if predicted_class else 'Legitimate Email'
    }

def main():
    # Load the model
    print("Loading model...")
    model, tokenizer = load_model()
    
    # Test emails
    test_emails = [
            """Dear valued customer, Your account security needs immediate attention. 
        Click here to verify your details: http://suspicious-link.com""",

        """Hi team, Here's the agenda for tomorrow's meeting at 10 AM:
        1. Project updates
        2. Budget review
        3. Q4 planning
        Best regards,
        John""", 
        """
It’s time to put these claims to the test. SquareX has developed a free Web Security Posture Assessment Tool, which evaluates an organisation’s level of protection against common web-based attacks across many actors

To start,
Make sure you are on your corporate network where all the security tools are activated on your device
Run the test!
        """
        ]
    
    print("\nTesting sample emails...")
    for i, email in enumerate(test_emails, 1):
        print(f"\nTest Email #{i}:")
        print("-" * 50)
        print(f"Content: {email[:100]}...")
        
        result = predict_email(email, model, tokenizer)
        
        print("\nResults:")
        print(f"Classification: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("-" * 50)

    # Interactive testing
    print("\nEnter your own email text to test (type 'quit' to exit):")
    while True:
        user_input = input("\nEnter email text: ")
        if user_input.lower() == 'quit':
            break
            
        result = predict_email(user_input, model, tokenizer)
        
        print("\nResults:")
        print(f"Classification: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")

if __name__ == "__main__":
    main()
