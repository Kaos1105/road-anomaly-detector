from server import create_app

app = create_app()

def start_server():
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    # threading.Thread(target=start_server, daemon=True).start()
    start_server()
