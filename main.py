from server import create_app
from kivy.uix.label import Label
from kivy.app import App
from threading import Thread

app = create_app()

def start_server():
    app.run(host='0.0.0.0', port=5000, debug=False)

class MyApp(App):
    def build(self):
        return Label(text="Flask is running!")
    
if __name__ == '__main__':
    Thread(target=start_server, daemon=True).start()
    MyApp().run()