import pyjapc
import smtplib
from email.mime.text import MIMEText

# === CONFIGURATION ===

JAPC_DEVICE_PROPERTY = "ACCCON_SYS.ANYDATA_QUEUE/LHC_PAGE1#value"  # e.g., LHC.BCTDC.A6R4/Acquisition#value



# === EMAIL FUNCTION ===

def send_email(content: str):
    msg = MIMEText(content, "html")
    email_adds = ["jan.pucek@cern.ch"]
    msg['From'] = "lhc.comments@cern.ch"
    msg['To'] = ', '.join(email_adds)
    msg["Subject"] = "Page1 comment Changed"

    smtp = smtplib.SMTP('localhost')
    smtp.sendmail(msg['From'], email_adds, msg.as_string())
    smtp.quit()


# === JAPC SETUP ===

def main():
    japc = pyjapc.PyJapc(selector='', incaAcceleratorName='')
    previous_value = None

    def callback(name, value):
        nonlocal previous_value
        if value != previous_value:
            print(f"Change detected: {value}")
            send_email(f"JAPC variable {JAPC_DEVICE_PROPERTY} changed to: {value}")
            previous_value = value

    print(f"Subscribing to {JAPC_DEVICE_PROPERTY}...")
    japc.subscribeParam(JAPC_DEVICE_PROPERTY, callback)
    japc.startSubscriptions()

    print("Listening for changes. Press Ctrl+C to exit.")
    try:
        while True:
            pass  # Keep the script running
    except KeyboardInterrupt:
        print("Stopping...")
        japc.stopSubscriptions()

if __name__ == "__main__":
    main()
