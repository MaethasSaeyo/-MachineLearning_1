import pyrebase

#Initialize Firebase
firebaseConfig={  "apiKey": "AIzaSyDc-8VFnDPLkAPKoBCYlo-ffd7XVjUNjlM",
  "authDomain": "test-666a1.firebaseapp.com",
  "databaseURL": "https://test-666a1-default-rtdb.asia-southeast1.firebasedatabase.app",
  "projectId": "test-666a1",
  "storageBucket": "test-666a1.appspot.com",
  "messagingSenderId": "423503459287",
  "appId": "1:423503459287:web:40d825a46c58ea03595192"}

firebase=pyrebase.initialize_app(firebaseConfig)

db=firebase.database()
direction = None
db.child("Branch").set(1)
