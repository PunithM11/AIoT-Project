import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- Configuration ---
SENDER_EMAIL = "punithm11122001@gmail.com"
SENDER_PASSWORD = "bdur xrlk hkmk giao" # 👈 Paste your 16-character App Password here
RECIPIENT_EMAIL = "punithm.4si24scs05@gmail.com" # 👈 An email address you can check
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
# --------------------

print(f"Attempting to send a test email from {SENDER_EMAIL} to {RECIPIENT_EMAIL}...")

# Create the email message
message = MIMEMultipart("alternative")
message["Subject"] = "Test Email from Python Script"
message["From"] = SENDER_EMAIL
message["To"] = RECIPIENT_EMAIL
text = "This is a test email sent from the Python script to confirm SMTP is working."
message.attach(MIMEText(text, "plain"))

try:
    # Connect to the server and send the email
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()  # Secure the connection
    server.login(SENDER_EMAIL, SENDER_PASSWORD)
    server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, message.as_string())
    server.quit()
    print("✅ Success! The test email was sent. Please check your inbox.")
    print("This confirms your credentials and connection are working.")
except Exception as e:
    print("\n❌ FAILED to send email. See the error below:")
    print(f"Error: {e}")