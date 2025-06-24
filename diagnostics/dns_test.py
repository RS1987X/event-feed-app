from flask import Flask
import socket

app = Flask(__name__)

@app.route('/')
def check_dns():
    hosts = ['imap.gmail.com', 'smtp.gmail.com', 'google.com']
    results = {}
    for host in hosts:
        try:
            ip = socket.gethostbyname(host)
            results[host] = f"Resolved to {ip}"
        except socket.gaierror as e:
            results[host] = f"Failed: {e}"
    return results

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)