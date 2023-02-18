from tkinter import *
from chat import get_response, botname

bg1="#8dd9d9"
bg2="#ffffff"
txtcol="#000000"

font1="Helvetica 14"
boldf="Helvetica 13 bold"

class ChatApp:

    def __init__(self):
        self.window=Tk()
        self._setup_main_window()

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("Chat")
        self.window.resizable(width=False,height=False)
        self.window.configure(width=470,height=550,bg=bg1)

        #head label
        head_label=Label(self.window,bg=bg1,fg=txtcol, text="Welcome!", font=boldf, pady=10)
        head_label.place(relwidth=1)

        #tiny divider
        line=Label(self.window,bg=bg2,width=450)
        line.place(relwidth=1,rely=0.07, relheight=0.012)

        #text widget
        self.text_widget=Text(self.window,width=20,height=2, bg=bg1, fg=txtcol, font=font1,padx=5,pady=5)
        self.text_widget.place(relheight=0.745,relwidth=1,rely=0.08)
        self.text_widget.configure(cursor="arrow",state=DISABLED)

        #scroll bar
        scrollbar=Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)

        #bottom label
        bottom_label=Label(self.window,bg=bg2,height=80)
        bottom_label.place(relwidth=1,rely=0.825)

        #message entry box
        self.msg_entry=Entry(bottom_label,bg="#ffffff",fg=txtcol, font=font1)
        self.msg_entry.place(relwidth=0.74,relheight=0.06,rely=0.008,relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>",self._on_enter_pressed)

        #send button
        send_button=Button(bottom_label,bg=bg2,fg=txtcol,font=boldf,text="Send", width=20, command=lambda: self._on_enter_pressed(N))
        send_button.place(relwidth=0.22, relheight=0.06, relx=0.77, rely=0.008)


    def _on_enter_pressed(self,event):
        msg=self.msg_entry.get()
        self._insert_message(msg, "You")

    def _insert_message(self, msg, sender):
        if not msg:
            return
        
        self.msg_entry.delete(0,END)
        msg1=f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END,msg1)
        self.text_widget.configure(cursor="arrow",state=DISABLED)

        msg2=f"{botname}: {get_response(msg)}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(cursor="arrow",state=DISABLED)

        self.text_widget.see(END)

if __name__=="__main__":
    app=ChatApp()
    app.run()