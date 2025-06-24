import socket

hosts = ['imap.gmail.com', 'smtp.gmail.com', 'google.com']

for host in hosts:
    try:
        ip = socket.gethostbyname(host)
        print(f"{host} resolved successfully to {ip}")
    except socket.gaierror as e:
        print(f"DNS resolution error for {host}: {e}")