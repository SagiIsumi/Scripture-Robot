import threading
import queue
import time
class thread_parameter_get():
    def __init__(self):
        self.thread=None
    def inner_getter(self,query,func,*args):
        query.put(func(args))
    def independent_thread(self,*args):
        self.thread=threading.Thread(target=self.inner_getter,args=args,daemon=True)
        self.thread.start()
def func1(*args):
  a= 5
  print(args)
  for i in args[0]:
    print(i)
  return a

query1=queue.Queue()
query2=queue.Queue()
thread_1=thread_parameter_get()
thread_2=thread_parameter_get()
try:
  thread_1.independent_thread(query1,func1,1,2,3,4,5,6,7,8,9,10)
  thread_2.independent_thread(query2,func1,6,7,8,9,10,1,2,3,4,5,999,1000)
  print('hi')
except Exception as e:
  pass
  # thread_1.thread.join()
  # thread_2.thread.join()
  # print(query1.get())
  # print(query2.get())
finally:
  print(query1.get())
  print(query2.get())
  print('hello, world')
print('over')
time.sleep(1)